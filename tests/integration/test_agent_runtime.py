"""Integration test for the complete semantic-to-physical runtime boundary."""

import asyncio
from contextlib import AsyncExitStack
from pathlib import Path

import httpx
import pytest

from infra_mas.core.agent import AgentSpec
from infra_mas.core.artifact import ArtifactRef
from infra_mas.core.errors import ExecutionFailedError
from infra_mas.core.execution import ExecutionRequest, InvocationSpec
from infra_mas.core.executor import ExecutorSpec
from infra_mas.core.model import ModelRequest, ModelResult, ModelSpec
from infra_mas.core.trace import TraceEvent
from infra_mas.execution.executor_registry import ExecutorRegistry
from infra_mas.execution.manager import ExecutionManager
from infra_mas.execution.transfer import TransferManager
from infra_mas.execution.worker_client import WorkerClient
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
    """Read the physical evidence file supplied by the runtime."""

    async def infer(self, request: ModelRequest) -> ModelResult:
        evidence = await asyncio.to_thread(
            Path(request.input_paths[0]).read_text,
            encoding="utf-8",
        )
        return ModelResult(output_text=f"answer from {evidence}", latency_ms=5)


class InstructionCapturingBackend:
    """Capture model requests to verify semantic instructions cross every boundary."""

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    async def infer(self, request: ModelRequest) -> ModelResult:
        self.requests.append(request)
        return ModelResult(output_text="done", latency_ms=1)


class SchedulingReached(RuntimeError):
    """Signal that modality validation completed before physical selection."""


class CountingScheduler:
    """Observe whether an invocation reaches scheduling."""

    def __init__(self) -> None:
        self.calls = 0

    async def select(self, invocation: InvocationSpec) -> ExecutorSpec:
        del invocation
        self.calls += 1
        raise SchedulingReached

    def preset_model_id(self, capability: str) -> str:
        del capability
        return "test-model"


async def test_manual_two_agent_runtime_workflow(tmp_path: Path) -> None:
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
        executor_registry = ExecutorRegistry(
            [
                ExecutorSpec(
                    id="worker-a-vision",
                    capability="visual_understanding",
                    worker_id="worker-a",
                    model_id="mock-vlm",
                    device="cpu",
                    site="edge",
                ),
                ExecutorSpec(
                    id="worker-b-reasoner",
                    capability="reasoning",
                    worker_id="worker-b",
                    model_id="mock-llm",
                    device="cpu",
                    site="remote",
                ),
            ],
            {
                "worker-a": "http://worker-a.test",
                "worker-b": "http://worker-b.test",
            },
        )
        scheduler = FixedScheduler(
            executor_registry,
            {
                "visual_understanding": "worker-a-vision",
                "reasoning": "worker-b-reasoner",
            },
        )
        trace = TraceRecorder(tmp_path / "runs", "run-001")
        await trace.start(
            {
                "scheduler": "fixed",
                "agents": ["vision_extractor", "reasoner"],
            }
        )
        transfer_manager = TransferManager(
            clients,
            trace,
        )
        manager = ExecutionManager(clients, transfer_manager, trace)
        request_ids = iter(["vision-request", "reasoning-request"])
        runtime = AgentRuntime(
            AgentRegistry(
                [
                    AgentSpec(
                        name="vision_extractor",
                        capability="visual_understanding",
                        instructions="Extract evidence.",
                    ),
                    AgentSpec(
                        name="reasoner",
                        capability="reasoning",
                        instructions="Reason over evidence.",
                    ),
                ]
            ),
            scheduler,
            manager,
            trace,
            request_id_factory=lambda: next(request_ids),
            model_registry=ModelRegistry(
                [
                    ModelSpec(
                        model_id="mock-vlm",
                        description="Vision model.",
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
            ),
        )

        vision_result = await runtime.execute(
            "vision_extractor",
            "Extract relevant evidence.",
            [],
        )
        reasoning_result = await runtime.execute(
            "reasoner",
            "Answer from the evidence.",
            vision_result.output_artifacts,
            parent_action_id="vision-request",
        )
        await trace.end(
            {"final_artifacts": [artifact.id for artifact in reasoning_result.output_artifacts]}
        )

    assert vision_result.executor_id == "worker-a-vision"
    assert vision_result.output_artifacts[0].locations == ["worker-a", "worker-b"]
    assert reasoning_result.executor_id == "worker-b-reasoner"
    assert reasoning_result.transfer_ms > 0
    answer_path = await store_b.get_path(reasoning_result.output_artifacts[0].id)
    assert answer_path.read_text(encoding="utf-8") == "answer from evidence"

    events = [
        TraceEvent.model_validate_json(line)
        for line in trace.path.read_text(encoding="utf-8").splitlines()
    ]
    assert [event.event_type for event in events] == [
        "run.start",
        "execution.request",
        "executor.selected",
        "worker.execution.start",
        "worker.execution.end",
        "artifact.created",
        "execution.request",
        "executor.selected",
        "artifact.transfer.start",
        "artifact.transfer.end",
        "worker.execution.start",
        "worker.execution.end",
        "artifact.created",
        "run.end",
    ]
    reasoning_request = events[6]
    assert reasoning_request.action_id == "reasoning-request"
    assert reasoning_request.parent_action_id == "vision-request"
    assert reasoning_request.model_extra == {
        "request_id": "reasoning-request",
        "agent": "reasoner",
        "model_id": "mock-llm",
        "capability": "reasoning",
        "task": "Answer from the evidence.",
        "input_artifacts": [vision_result.output_artifacts[0].id],
    }
    assert trace.config_path.exists()
    assert trace.result_path.exists()


async def test_failed_worker_execution_records_end_event(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "worker", "worker")
    worker = WorkerService(
        "worker",
        store,
        [WorkerExecutor("worker-llm", "reasoning", MockBackend(fail=True))],
    )
    trace = TraceRecorder(tmp_path / "runs", "run-001")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_app(worker)),
        base_url="http://worker.test",
    ) as http_client:
        clients = {"worker": WorkerClient(client=http_client)}
        manager = ExecutionManager(
            clients,
            TransferManager(clients, trace),
            trace,
        )
        with pytest.raises(ExecutionFailedError, match="mock model execution failed"):
            await manager.execute(
                ExecutionRequest(
                    request_id="request-failed",
                    agent="reasoner",
                    model_id="mock",
                    capability="reasoning",
                    instructions="Reason over evidence.",
                    task="Reason.",
                    inputs=[],
                ),
                ExecutorSpec(
                    id="worker-llm",
                    capability="reasoning",
                    worker_id="worker",
                    model_id="mock",
                    device="cpu",
                    site="local",
                ),
            )

    events = [
        TraceEvent.model_validate_json(line)
        for line in trace.path.read_text(encoding="utf-8").splitlines()
    ]
    assert [event.event_type for event in events] == [
        "worker.execution.start",
        "worker.execution.end",
    ]
    failure_details = events[-1].model_extra
    assert failure_details is not None
    assert failure_details["success"] is False


async def test_distinct_agent_instructions_reach_backend(tmp_path: Path) -> None:
    backend = InstructionCapturingBackend()
    worker = WorkerService(
        "worker",
        ArtifactStore(tmp_path / "worker", "worker"),
        [WorkerExecutor("worker-llm", "reasoning", backend)],
    )
    trace = TraceRecorder(tmp_path / "runs", "run-instructions")
    registry = ExecutorRegistry(
        [
            ExecutorSpec(
                id="worker-llm",
                capability="reasoning",
                worker_id="worker",
                model_id="mock",
                device="cpu",
                site="local",
            )
        ],
        {"worker": "http://worker.test"},
    )
    agents = AgentRegistry(
        [
            AgentSpec(name="critic", capability="reasoning", instructions="Critique claims."),
            AgentSpec(name="solver", capability="reasoning", instructions="Solve precisely."),
        ]
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_app(worker)),
        base_url="http://worker.test",
    ) as http_client:
        clients = {"worker": WorkerClient(client=http_client)}
        runtime = AgentRuntime(
            agents,
            FixedScheduler(registry, {"reasoning": "worker-llm"}),
            ExecutionManager(clients, TransferManager(clients, trace), trace),
            trace,
            model_registry=ModelRegistry(
                [
                    ModelSpec(
                        model_id="mock",
                        description="Reasoning model.",
                        input_modalities=["text"],
                        output_modalities=["text"],
                        context_window=8192,
                    )
                ]
            ),
        )
        await runtime.execute("critic", "Review this.", [])
        await runtime.execute("solver", "Answer this.", [])

    assert [request.instructions for request in backend.requests] == [
        "Critique claims.",
        "Solve precisely.",
    ]
    assert [request.task for request in backend.requests] == ["Review this.", "Answer this."]


@pytest.mark.parametrize(
    ("artifact_type", "input_modality"),
    [
        ("image/png", "image"),
        ("text/plain; charset=utf-8", "text"),
        ("application/json", "text"),
        ("application/x-yaml", "text"),
        ("application/xml", "text"),
        ("application/problem+json", "text"),
    ],
)
async def test_compatible_artifact_mime_reaches_scheduler(
    tmp_path: Path,
    artifact_type: str,
    input_modality: str,
) -> None:
    trace = TraceRecorder(tmp_path / "runs", f"run-{input_modality}")
    scheduler = CountingScheduler()
    clients: dict[str, WorkerClient] = {}
    runtime = AgentRuntime(
        None,
        scheduler,
        ExecutionManager(clients, TransferManager(clients, trace), trace),
        trace,
        model_registry=ModelRegistry(
            [
                ModelSpec(
                    model_id="test-model",
                    description="Modality test model.",
                    input_modalities=[input_modality],
                    output_modalities=["text"],
                    context_window=8192,
                )
            ]
        ),
    )
    invocation = InvocationSpec(
        model_id="test-model",
        role="tester",
        instructions="Test the input.",
        task="Inspect it.",
        input_artifacts=[
            ArtifactRef(
                id="run/input",
                artifact_type=artifact_type,
                size_bytes=1,
                locations=["worker"],
            )
        ],
    )

    with pytest.raises(SchedulingReached):
        await runtime.invoke(invocation)

    assert scheduler.calls == 1


@pytest.mark.parametrize("artifact_type", ["image/jpeg", "audio/mpeg"])
async def test_incompatible_artifact_mime_is_rejected_before_scheduling(
    tmp_path: Path,
    artifact_type: str,
) -> None:
    trace = TraceRecorder(tmp_path / "runs", "run-incompatible")
    scheduler = CountingScheduler()
    clients: dict[str, WorkerClient] = {}
    runtime = AgentRuntime(
        None,
        scheduler,
        ExecutionManager(clients, TransferManager(clients, trace), trace),
        trace,
        model_registry=ModelRegistry(
            [
                ModelSpec(
                    model_id="text-model",
                    description="Text-only model.",
                    input_modalities=["text"],
                    output_modalities=["text"],
                    context_window=8192,
                )
            ]
        ),
    )
    invocation = InvocationSpec(
        model_id="text-model",
        role="tester",
        instructions="Test the input.",
        task="Inspect it.",
        input_artifacts=[
            ArtifactRef(
                id="run/input",
                artifact_type=artifact_type,
                size_bytes=1,
                locations=["worker"],
            )
        ],
    )

    with pytest.raises(
        ValueError,
        match=rf"text-model.*run/input.*{artifact_type}",
    ):
        await runtime.invoke(invocation)

    assert scheduler.calls == 0
