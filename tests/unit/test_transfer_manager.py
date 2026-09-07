"""Unit tests for explicit artifact transfer decisions."""

import asyncio
from pathlib import Path
from typing import cast

import pytest

from infra_mas.core.artifact import ArtifactRef
from infra_mas.core.errors import ArtifactTransferError
from infra_mas.core.execution import TransferResult
from infra_mas.execution.transfer import TransferManager
from infra_mas.execution.worker_client import WorkerClient
from infra_mas.tracing.recorder import TraceRecorder


class FakeSourceClient:
    @property
    def base_url(self) -> str:
        return "http://worker-a.test"


class CountingTargetClient:
    def __init__(self, size_bytes: int) -> None:
        self.calls = 0
        self.size_bytes = size_bytes

    async def pull_artifact(self, request: object) -> TransferResult:
        del request
        self.calls += 1
        await asyncio.sleep(0.01)
        return TransferResult(bytes_transferred=self.size_bytes, transfer_ms=10)


async def test_already_local_artifact_is_not_transferred(tmp_path: Path) -> None:
    trace = TraceRecorder(tmp_path / "runs", "run-001")
    manager = TransferManager({}, trace)
    artifact = ArtifactRef(
        id="run-001/input-001",
        artifact_type="text/plain",
        size_bytes=4,
        locations=["worker-b"],
    )

    result = await manager.ensure_local(artifact, "worker-b")

    assert result.bytes_transferred == 0
    assert result.transfer_ms == 0
    assert not trace.path.exists()


async def test_unknown_target_is_rejected(tmp_path: Path) -> None:
    trace = TraceRecorder(tmp_path / "runs", "run-001")
    manager = TransferManager({}, trace)
    artifact = ArtifactRef(
        id="run-001/input-001",
        artifact_type="text/plain",
        size_bytes=4,
        locations=["worker-a"],
    )

    with pytest.raises(ArtifactTransferError, match="unknown target worker"):
        await manager.ensure_local(artifact, "worker-b")


async def test_concurrent_transfer_to_same_target_is_deduplicated(tmp_path: Path) -> None:
    target = CountingTargetClient(size_bytes=4)
    clients = {
        "worker-a": cast(WorkerClient, FakeSourceClient()),
        "worker-b": cast(WorkerClient, target),
    }
    trace = TraceRecorder(tmp_path / "runs", "run-001")
    manager = TransferManager(clients, trace)
    artifact = ArtifactRef(
        id="run-001/input-001",
        artifact_type="text/plain",
        size_bytes=4,
        locations=["worker-a"],
    )

    references = [artifact.model_copy(deep=True) for _ in range(10)]
    results = await asyncio.gather(
        *(
            manager.ensure_local(reference, "worker-b", action_id="action")
            for reference in references
        )
    )

    assert target.calls == 1
    assert sum(result.bytes_transferred for result in results) == 4
    assert all(reference.locations == ["worker-a", "worker-b"] for reference in references)
    assert len(trace.path.read_text(encoding="utf-8").splitlines()) == 2
