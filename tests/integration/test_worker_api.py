"""Integration tests for the controller-to-worker HTTP path."""

from pathlib import Path
from typing import cast

import httpx
import pytest

from infra_mas.core.artifact import ArtifactRef
from infra_mas.core.errors import (
    ArtifactNotFoundError,
    ExecutionFailedError,
    InvalidModelResponseError,
    WorkerUnavailableError,
)
from infra_mas.core.execution import ExecutionRequest
from infra_mas.core.model import ModelRequest, ModelResult
from infra_mas.execution.worker_client import WorkerClient
from infra_mas.worker.artifact_store import ArtifactStore
from infra_mas.worker.backends.mock import MockBackend
from infra_mas.worker.server import create_app
from infra_mas.worker.service import WorkerExecutor, WorkerService


class InvalidBackend:
    """Deliberately violate the backend protocol at runtime for validation testing."""

    async def infer(self, request: ModelRequest) -> ModelResult:
        del request
        return cast(ModelResult, object())


def build_service(tmp_path: Path) -> WorkerService:
    store = ArtifactStore(tmp_path / "artifacts", "worker-1")
    return WorkerService(
        worker_id="worker-1",
        artifact_store=store,
        executors=[
            WorkerExecutor(
                id="worker-1-reasoner",
                capability="reasoning",
                backend=MockBackend(latency_ms=25, output="mock answer"),
            )
        ],
        artifact_id_factory=lambda request: f"run-001/output-{request.request_id}",
    )


async def test_controller_executes_request_and_downloads_output(tmp_path: Path) -> None:
    service = build_service(tmp_path)
    input_artifact = await service.artifact_store.put_text("run-001/input-001", "input evidence")
    app = create_app(service)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://worker.test"
    ) as http_client:
        client = WorkerClient(client=http_client)

        health = await client.health()
        status = await client.status()
        result = await client.execute(
            ExecutionRequest(
                request_id="request-001",
                agent="reasoner",
                capability="reasoning",
                instructions="Reason over evidence.",
                task="Answer from the evidence.",
                inputs=[input_artifact],
            )
        )
        destination = tmp_path / "downloaded.txt"
        transferred = await client.download_artifact(result.output_artifacts[0].id, destination)

    assert health.status == "ok"
    assert status.worker_id == "worker-1"
    assert status.executors == ["worker-1-reasoner"]
    assert result.executor_id == "worker-1-reasoner"
    assert result.service_ms == 25
    assert result.output_artifacts[0].locations == ["worker-1"]
    assert transferred == len(b"mock answer")
    assert destination.read_text(encoding="utf-8") == "mock answer"


async def test_controller_streams_artifact_upload(tmp_path: Path) -> None:
    service = build_service(tmp_path)
    app = create_app(service)
    source = tmp_path / "source.bin"
    source.write_bytes(b"streamed-data")
    artifact = ArtifactRef(
        id="run-001/upload-001",
        artifact_type="application/octet-stream",
        size_bytes=source.stat().st_size,
        locations=["controller"],
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://worker.test"
    ) as http_client:
        uploaded = await WorkerClient(client=http_client).upload_artifact(artifact, source)

    assert uploaded.locations == ["worker-1"]
    assert uploaded.size_bytes == source.stat().st_size
    assert (await service.artifact_store.get_path(uploaded.id)).read_bytes() == b"streamed-data"


async def test_missing_artifact_download_is_typed(tmp_path: Path) -> None:
    app = create_app(build_service(tmp_path))

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://worker.test"
    ) as http_client:
        client = WorkerClient(client=http_client)
        with pytest.raises(ArtifactNotFoundError, match="run-001/missing"):
            await client.download_artifact("run-001/missing", tmp_path / "missing")


async def test_missing_execution_input_is_typed(tmp_path: Path) -> None:
    app = create_app(build_service(tmp_path))
    missing = ArtifactRef(
        id="run-001/missing",
        artifact_type="text/plain",
        size_bytes=1,
        locations=["worker-1"],
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://worker.test"
    ) as http_client:
        client = WorkerClient(client=http_client)
        with pytest.raises(ArtifactNotFoundError, match="run-001/missing"):
            await client.execute(
                ExecutionRequest(
                    request_id="request-001",
                    agent="reasoner",
                    capability="reasoning",
                    instructions="Reason over evidence.",
                    task="Answer from the evidence.",
                    inputs=[missing],
                )
            )


async def test_worker_connection_failure_is_typed() -> None:
    async def unavailable(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("worker is offline", request=request)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(unavailable), base_url="http://worker.test"
    ) as http_client:
        client = WorkerClient(client=http_client)
        with pytest.raises(WorkerUnavailableError, match="worker is offline"):
            await client.health()


async def test_scheduler_selected_executor_is_honored(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts", "worker-1")
    service = WorkerService(
        worker_id="worker-1",
        artifact_store=store,
        executors=[
            WorkerExecutor("reasoner-a", "reasoning", MockBackend(output="answer-a")),
            WorkerExecutor("reasoner-b", "reasoning", MockBackend(output="answer-b")),
        ],
        artifact_id_factory=lambda request: f"run-001/output-{request.request_id}",
    )
    request = ExecutionRequest(
        request_id="request-001",
        agent="reasoner",
        capability="reasoning",
        instructions="Reason over evidence.",
        task="Answer the question.",
        inputs=[],
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_app(service)),
        base_url="http://worker.test",
    ) as http_client:
        client = WorkerClient(client=http_client)
        with pytest.raises(ExecutionFailedError, match="requires an executor selection"):
            await client.execute(request)
        result = await client.execute(request, executor_id="reasoner-b")

    assert result.executor_id == "reasoner-b"
    output_path = await store.get_path(result.output_artifacts[0].id)
    assert output_path.read_text(encoding="utf-8") == "answer-b"


async def test_invalid_backend_result_is_rejected(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts", "worker-1")
    service = WorkerService(
        worker_id="worker-1",
        artifact_store=store,
        executors=[WorkerExecutor("reasoner", "reasoning", InvalidBackend())],
    )

    with pytest.raises(InvalidModelResponseError, match="invalid result"):
        await service.execute(
            ExecutionRequest(
                request_id="run-001/request-001",
                agent="reasoner",
                capability="reasoning",
                instructions="Reason over evidence.",
                task="Answer the question.",
                inputs=[],
            )
        )
