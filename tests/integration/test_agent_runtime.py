"""Integration test for the complete semantic-to-physical runtime boundary."""

import asyncio
from contextlib import AsyncExitStack
from pathlib import Path

import httpx

from infra_mas.core.agent import AgentSpec
from infra_mas.core.executor import ExecutorSpec
from infra_mas.core.model import ModelRequest, ModelResult
from infra_mas.execution.executor_registry import ExecutorRegistry
from infra_mas.execution.manager import ExecutionManager
from infra_mas.execution.transfer import TransferManager
from infra_mas.execution.worker_client import WorkerClient
from infra_mas.runtime.agent_registry import AgentRegistry
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
    worker_b = WorkerService(
        "worker-b",
        store_b,
        [WorkerExecutor("worker-b-reasoner", "reasoning", EvidenceReasonerBackend())],
        artifact_id_factory=lambda request: f"run-001/answer-{request.request_id}",
    )

    async with AsyncExitStack() as stack:
        http_a = await stack.enter_async_context(
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=create_app(worker_a)),
                base_url="http://worker-a.test",
            )
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
                    model="mock-vlm",
                    device="cpu",
                    site="edge",
                ),
                ExecutorSpec(
                    id="worker-b-reasoner",
                    capability="reasoning",
                    worker_id="worker-b",
                    model="mock-llm",
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
        transfer_manager = TransferManager(
            clients,
            tmp_path / "transfers",
            TraceRecorder(tmp_path / "runs", "run-001"),
        )
        manager = ExecutionManager(clients, transfer_manager)
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
            request_id_factory=lambda: next(request_ids),
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
        )

    assert vision_result.executor_id == "worker-a-vision"
    assert vision_result.output_artifacts[0].locations == ["worker-a", "worker-b"]
    assert reasoning_result.executor_id == "worker-b-reasoner"
    assert reasoning_result.transfer_ms > 0
    answer_path = await store_b.get_path(reasoning_result.output_artifacts[0].id)
    assert answer_path.read_text(encoding="utf-8") == "answer from evidence"
