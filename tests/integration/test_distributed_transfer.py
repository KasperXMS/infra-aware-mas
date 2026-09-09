"""Integration test for producing on one worker and consuming on another."""

import asyncio
import json
from contextlib import AsyncExitStack
from pathlib import Path

import httpx

from infra_mas.core.execution import ExecutionRequest
from infra_mas.core.model import ModelRequest, ModelResult
from infra_mas.execution.transfer import TransferManager
from infra_mas.execution.worker_client import WorkerClient
from infra_mas.tracing.recorder import TraceRecorder
from infra_mas.worker.artifact_store import ArtifactStore
from infra_mas.worker.backends.mock import MockBackend
from infra_mas.worker.server import create_app
from infra_mas.worker.service import WorkerExecutor, WorkerService


class EvidenceReasonerBackend:
    """Read transferred evidence to prove the target consumed the physical file."""

    async def infer(self, request: ModelRequest) -> ModelResult:
        evidence_path = Path(request.input_paths[0])
        evidence = await asyncio.to_thread(evidence_path.read_text, encoding="utf-8")
        return ModelResult(output_text=f"reasoned from: {evidence}", latency_ms=5)


class NoControllerRelayClient(WorkerClient):
    """Fail if TransferManager attempts to download artifact bytes into Controller."""

    async def download_artifact(self, artifact_id: str, destination: Path) -> int:
        del artifact_id, destination
        raise AssertionError("Controller must not relay artifact bytes")


async def test_artifact_moves_from_worker_a_to_worker_b(tmp_path: Path) -> None:
    store_a = ArtifactStore(tmp_path / "worker-a", "worker-a")
    service_a = WorkerService(
        worker_id="worker-a",
        artifact_store=store_a,
        executors=[
            WorkerExecutor(
                "worker-a-vision",
                "visual_understanding",
                MockBackend(output="visual evidence", latency_ms=10),
            )
        ],
        artifact_id_factory=lambda request: f"run-001/evidence-{request.request_id}",
    )
    store_b = ArtifactStore(tmp_path / "worker-b", "worker-b")

    async with AsyncExitStack() as stack:
        http_a = await stack.enter_async_context(
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=create_app(service_a)),
                base_url="http://worker-a.test",
            )
        )
        service_b = WorkerService(
            worker_id="worker-b",
            artifact_store=store_b,
            executors=[WorkerExecutor("worker-b-reasoner", "reasoning", EvidenceReasonerBackend())],
            artifact_id_factory=lambda request: f"run-001/answer-{request.request_id}",
            transfer_client=http_a,
        )
        http_b = await stack.enter_async_context(
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=create_app(service_b)),
                base_url="http://worker-b.test",
            )
        )
        clients = {
            "worker-a": NoControllerRelayClient(client=http_a),
            "worker-b": WorkerClient(client=http_b),
        }
        vision_result = await clients["worker-a"].execute(
            ExecutionRequest(
                request_id="request-vision",
                agent="vision_extractor",
                model_id="mock-vlm",
                capability="visual_understanding",
                instructions="Extract visual evidence.",
                task="Extract relevant evidence.",
                inputs=[],
            ),
            executor_id="worker-a-vision",
        )
        evidence = vision_result.output_artifacts[0]
        trace = TraceRecorder(tmp_path / "runs", "run-001")
        transfer = await TransferManager(
            clients,
            trace,
        ).ensure_local(evidence, "worker-b", action_id="action-reasoning")
        reasoning_result = await clients["worker-b"].execute(
            ExecutionRequest(
                request_id="request-reasoning",
                agent="reasoner",
                model_id="mock-llm",
                capability="reasoning",
                instructions="Reason over evidence.",
                task="Answer from the evidence.",
                inputs=[evidence],
            ),
            executor_id="worker-b-reasoner",
        )

    assert transfer.bytes_transferred == len(b"visual evidence")
    assert evidence.locations == ["worker-a", "worker-b"]
    assert await store_b.exists(evidence.id)
    answer_path = await store_b.get_path(reasoning_result.output_artifacts[0].id)
    assert answer_path.read_text(encoding="utf-8") == "reasoned from: visual evidence"

    events = [json.loads(line) for line in trace.path.read_text(encoding="utf-8").splitlines()]
    assert [event["event_type"] for event in events] == [
        "artifact.transfer.start",
        "artifact.transfer.end",
    ]
    assert events[1]["bytes_transferred"] == len(b"visual evidence")
    assert events[1]["path"] == "worker_to_worker"
    assert events[1]["success"] is True
    assert events[1]["action_id"] == "action-reasoning"
