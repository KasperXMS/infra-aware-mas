"""Integration test for SDK Coordinator delegation through the distributed runtime."""

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any, cast

import httpx
from agents import Agent
from agents.agent_output import AgentOutputSchemaBase
from agents.handoffs import Handoff
from agents.items import (
    ModelResponse,
    TResponseInputItem,
    TResponseOutputItem,
    TResponseStreamEvent,
)
from agents.model_settings import ModelSettings
from agents.models.interface import Model, ModelTracing
from agents.tool import Tool
from agents.usage import Usage
from openai.types.responses import (
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
)
from openai.types.responses.response_prompt_param import ResponsePromptParam

from infra_mas.core.agent import AgentSpec
from infra_mas.core.executor import ExecutorSpec
from infra_mas.core.model import ModelRequest, ModelResult, ModelSpec
from infra_mas.core.trace import TraceEvent
from infra_mas.execution.executor_registry import ExecutorRegistry
from infra_mas.execution.manager import ExecutionManager
from infra_mas.execution.transfer import TransferManager
from infra_mas.execution.worker_client import WorkerClient
from infra_mas.planner.context import ArtifactCatalog, PlannerContext
from infra_mas.planner.coordinator import Coordinator
from infra_mas.runtime.agent_registry import AgentRegistry
from infra_mas.runtime.model_registry import ModelRegistry
from infra_mas.runtime.runtime import AgentRuntime
from infra_mas.scheduler.fixed import FixedScheduler
from infra_mas.tracing.recorder import TraceRecorder
from infra_mas.worker.artifact_store import ArtifactStore
from infra_mas.worker.backends.mock import MockBackend
from infra_mas.worker.server import create_app
from infra_mas.worker.service import WorkerExecutor, WorkerService


class EvidenceReasonerBackend:
    """Consume the transferred evidence artifact on the reasoning worker."""

    async def infer(self, request: ModelRequest) -> ModelResult:
        evidence = await asyncio.to_thread(
            Path(request.input_paths[0]).read_text,
            encoding="utf-8",
        )
        return ModelResult(output_text=f"answer from {evidence}", latency_ms=5)


class CapturingBackend:
    """Capture the dynamic role's worker-local request."""

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    async def infer(self, request: ModelRequest) -> ModelResult:
        self.requests.append(request)
        return ModelResult(output_text="dynamic finding", latency_ms=2)


class ScriptedCoordinatorModel(Model):
    """Drive two function calls and one inspection without an external API."""

    def __init__(self) -> None:
        self._turn = 0
        self.instructions_seen: str | None = None

    async def get_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Handoff[Any, Agent[Any]]],
        tracing: ModelTracing,
        *,
        previous_response_id: str | None,
        conversation_id: str | None,
        prompt: ResponsePromptParam | None,
    ) -> ModelResponse:
        del (
            input,
            model_settings,
            tools,
            output_schema,
            handoffs,
            tracing,
            previous_response_id,
            conversation_id,
            prompt,
        )
        self.instructions_seen = system_instructions
        self._turn += 1
        return ModelResponse(
            output=self._output_for_turn(self._turn),
            usage=Usage(),
            response_id=f"response-{self._turn}",
        )

    async def stream_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Handoff[Any, Agent[Any]]],
        tracing: ModelTracing,
        *,
        previous_response_id: str | None,
        conversation_id: str | None,
        prompt: ResponsePromptParam | None,
    ) -> AsyncIterator[TResponseStreamEvent]:
        del (
            system_instructions,
            input,
            model_settings,
            tools,
            output_schema,
            handoffs,
            tracing,
            previous_response_id,
            conversation_id,
            prompt,
        )
        if False:
            yield cast(TResponseStreamEvent, {})

    @staticmethod
    def _output_for_turn(turn: int) -> list[TResponseOutputItem]:
        if turn == 1:
            return [
                ResponseFunctionToolCall(
                    arguments=json.dumps(
                        {
                            "agent": "vision_extractor",
                            "task": "Extract relevant evidence.",
                            "input_artifact_ids": [],
                        }
                    ),
                    call_id="call-delegate-vision",
                    name="delegate",
                    type="function_call",
                )
            ]
        if turn == 2:
            return [
                ResponseFunctionToolCall(
                    arguments=json.dumps(
                        {
                            "agent": "reasoner",
                            "task": "Answer from the evidence.",
                            "input_artifact_ids": ["run-001/evidence-vision-request"],
                        }
                    ),
                    call_id="call-delegate-reasoner",
                    name="delegate",
                    type="function_call",
                )
            ]
        if turn == 3:
            return [
                ResponseFunctionToolCall(
                    arguments=json.dumps({"artifact_id": "run-001/answer-reasoning-request"}),
                    call_id="call-inspect-answer",
                    name="inspect_artifact",
                    type="function_call",
                )
            ]
        return [
            ResponseOutputMessage(
                id="message-final",
                content=[
                    ResponseOutputText(
                        annotations=[],
                        text="answer from evidence",
                        type="output_text",
                    )
                ],
                role="assistant",
                status="completed",
                type="message",
            )
        ]


class DynamicCoordinatorModel(ScriptedCoordinatorModel):
    """Create a novel role that is absent from every static AgentRegistry."""

    @staticmethod
    def _output_for_turn(turn: int) -> list[TResponseOutputItem]:
        if turn == 1:
            return [
                ResponseFunctionToolCall(
                    arguments=json.dumps(
                        {
                            "model_id": "general-llm",
                            "role": "counterexample_hunter",
                            "instructions": "Find decisive counterexamples and state uncertainty.",
                            "task": "Stress-test the proposed claim.",
                            "input_artifact_ids": [],
                        }
                    ),
                    call_id="call-spawn-dynamic",
                    name="spawn_agent",
                    type="function_call",
                )
            ]
        return [
            ResponseOutputMessage(
                id="message-dynamic-final",
                content=[
                    ResponseOutputText(
                        annotations=[],
                        text="dynamic finding",
                        type="output_text",
                    )
                ],
                role="assistant",
                status="completed",
                type="message",
            )
        ]


async def test_coordinator_performs_two_blind_delegations(tmp_path: Path) -> None:
    store_a = ArtifactStore(tmp_path / "worker-a", "worker-a")
    worker_a = WorkerService(
        "worker-a",
        store_a,
        [
            WorkerExecutor(
                "worker-a-vision",
                "visual_understanding",
                MockBackend(output="evidence", latency_ms=10),
            )
        ],
        artifact_id_factory=lambda request: f"run-001/evidence-{request.request_id}",
    )
    store_b = ArtifactStore(tmp_path / "worker-b", "worker-b")

    async with AsyncExitStack() as stack:
        http_a = await stack.enter_async_context(
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=create_app(worker_a)),
                base_url="http://worker-a.test",
            )
        )
        worker_b = WorkerService(
            "worker-b",
            store_b,
            [WorkerExecutor("worker-b-reasoner", "reasoning", EvidenceReasonerBackend())],
            artifact_id_factory=lambda request: f"run-001/answer-{request.request_id}",
            transfer_client=http_a,
        )
        http_b = await stack.enter_async_context(
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=create_app(worker_b)),
                base_url="http://worker-b.test",
            )
        )
        clients = {
            "worker-a": WorkerClient(client=http_a),
            "worker-b": WorkerClient(client=http_b),
        }
        agents = AgentRegistry(
            [
                AgentSpec(
                    name="vision_extractor",
                    capability="visual_understanding",
                    instructions="Extract visual evidence.",
                ),
                AgentSpec(
                    name="reasoner",
                    capability="reasoning",
                    instructions="Reason over supplied evidence.",
                ),
            ]
        )
        executors = [
            ExecutorSpec(
                id="worker-a-vision",
                capability="visual_understanding",
                worker_id="worker-a",
                model_id="mock-vlm",
                device="edge-device",
                site="edge",
            ),
            ExecutorSpec(
                id="worker-b-reasoner",
                capability="reasoning",
                worker_id="worker-b",
                model_id="mock-llm",
                device="remote-device",
                site="remote",
            ),
        ]
        executor_registry = ExecutorRegistry(
            executors,
            {
                "worker-a": "http://worker-a.test",
                "worker-b": "http://worker-b.test",
            },
        )
        models = ModelRegistry(
            [
                ModelSpec(
                    model_id="mock-vlm",
                    description="Visual evidence model.",
                    input_modalities=["text", "image"],
                    output_modalities=["text"],
                    context_window=8192,
                ),
                ModelSpec(
                    model_id="mock-llm",
                    description="Reasoning model.",
                    input_modalities=["text"],
                    output_modalities=["text"],
                    context_window=8192,
                ),
            ]
        )
        trace = TraceRecorder(tmp_path / "runs", "run-001")
        transfer = TransferManager(clients, trace)
        manager = ExecutionManager(clients, transfer, trace)
        request_ids = iter(["vision-request", "reasoning-request"])
        runtime = AgentRuntime(
            agents,
            FixedScheduler(
                executor_registry,
                {
                    "visual_understanding": "worker-a-vision",
                    "reasoning": "worker-b-reasoner",
                },
            ),
            manager,
            trace,
            request_id_factory=lambda: next(request_ids),
            model_registry=models,
        )
        catalog = ArtifactCatalog(clients, tmp_path / "inspection", trace)
        context = PlannerContext(runtime, agents, catalog, trace, model_registry=models)
        model = ScriptedCoordinatorModel()

        await trace.start({"test": "coordinator"})
        answer = await Coordinator(context, model=model).run(
            "Analyze the visual information and answer the question."
        )
        await trace.end({"success": True, "answer": answer})

    assert answer == "answer from evidence"
    assert [artifact.id for artifact in catalog.list()] == [
        "run-001/evidence-vision-request",
        "run-001/answer-reasoning-request",
    ]
    assert model.instructions_seen is not None
    assert "worker-a" not in model.instructions_seen
    assert "worker-b" not in model.instructions_seen
    assert "edge-device" not in model.instructions_seen
    assert "remote-device" not in model.instructions_seen

    events = [
        TraceEvent.model_validate_json(line)
        for line in trace.path.read_text(encoding="utf-8").splitlines()
    ]
    assert [event.event_type for event in events].count("planner.delegate") == 2
    assert [event.event_type for event in events].count("planner.inspect_artifact") == 1
    assert [event.event_type for event in events].count("planner.llm.start") == 4
    assert [event.event_type for event in events].count("planner.llm.end") == 4
    llm_end = next(event for event in events if event.event_type == "planner.llm.end")
    assert llm_end.model_extra is not None
    assert llm_end.model_extra["turn"] == 1
    assert llm_end.model_extra["latency_ms"] >= 0
    assert llm_end.model_extra["token_usage"] == {
        "requests": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }
    planner_finish = events[-2]
    assert planner_finish.model_extra is not None
    assert planner_finish.model_extra["turn_count"] == 4
    assert events[-2].event_type == "planner.finish"
    assert events[-1].event_type == "run.end"


async def test_dynamic_planner_invokes_unregistered_role(tmp_path: Path) -> None:
    backend = CapturingBackend()
    worker = WorkerService(
        "worker-a",
        ArtifactStore(tmp_path / "worker-a", "worker-a"),
        [WorkerExecutor("worker-a-llm", "reasoning", backend)],
        artifact_id_factory=lambda request: f"run-dynamic/output-{request.request_id}",
    )
    executors = ExecutorRegistry(
        [
            ExecutorSpec(
                id="worker-a-llm",
                capability="reasoning",
                worker_id="worker-a",
                model_id="general-llm",
                device="hidden-device",
                site="hidden-site",
            )
        ],
        {"worker-a": "http://worker-a.test"},
    )
    models = ModelRegistry(
        [
            ModelSpec(
                model_id="general-llm",
                description="General reasoning and critique model.",
                input_modalities=["text"],
                output_modalities=["text"],
                context_window=16384,
            )
        ]
    )
    trace = TraceRecorder(tmp_path / "runs", "run-dynamic")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_app(worker)),
        base_url="http://worker-a.test",
    ) as http_client:
        clients = {"worker-a": WorkerClient(client=http_client)}
        runtime = AgentRuntime(
            None,
            FixedScheduler(executors, {"general-llm": "worker-a-llm"}),
            ExecutionManager(clients, TransferManager(clients, trace), trace),
            trace,
            request_id_factory=lambda: "dynamic-request",
            model_registry=models,
        )
        catalog = ArtifactCatalog(clients, tmp_path / "inspection-dynamic", trace)
        context = PlannerContext(
            runtime,
            None,
            catalog,
            trace,
            model_registry=models,
            planner_mode="dynamic_models",
        )
        planner_model = DynamicCoordinatorModel()

        await trace.start({"planner_mode": "dynamic_models"})
        answer = await Coordinator(context, model=planner_model).run("Challenge this claim.")
        await trace.end({"success": True, "answer": answer})

    assert answer == "dynamic finding"
    assert len(backend.requests) == 1
    assert backend.requests[0].instructions == (
        "Find decisive counterexamples and state uncertainty."
    )
    assert backend.requests[0].task == "Stress-test the proposed claim."
    assert planner_model.instructions_seen is not None
    assert "general-llm" in planner_model.instructions_seen
    assert "counterexample_hunter" not in planner_model.instructions_seen
    assert "worker-a" not in planner_model.instructions_seen
    assert "hidden-device" not in planner_model.instructions_seen

    events = [
        TraceEvent.model_validate_json(line)
        for line in trace.path.read_text(encoding="utf-8").splitlines()
    ]
    assert [event.event_type for event in events].count("planner.spawn_agent") == 1
    spawn_event = next(event for event in events if event.event_type == "planner.spawn_agent")
    assert spawn_event.model_extra is not None
    assert spawn_event.model_extra["instructions"] == (
        "Find decisive counterexamples and state uncertainty."
    )
    execution_request = next(
        event for event in events if event.event_type == "execution.request"
    )
    assert execution_request.model_extra is not None
    assert execution_request.model_extra["agent"] == "counterexample_hunter"
    assert execution_request.model_extra["model_id"] == "general-llm"
