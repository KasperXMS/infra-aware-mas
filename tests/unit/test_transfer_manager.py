"""Unit tests for explicit artifact transfer decisions."""

from pathlib import Path

import pytest

from infra_mas.core.artifact import ArtifactRef
from infra_mas.core.errors import ArtifactTransferError
from infra_mas.execution.transfer import TransferManager
from infra_mas.tracing.recorder import TraceRecorder


async def test_already_local_artifact_is_not_transferred(tmp_path: Path) -> None:
    trace = TraceRecorder(tmp_path / "runs", "run-001")
    manager = TransferManager({}, tmp_path / "transfers", trace)
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
    manager = TransferManager({}, tmp_path / "transfers", trace)
    artifact = ArtifactRef(
        id="run-001/input-001",
        artifact_type="text/plain",
        size_bytes=4,
        locations=["worker-a"],
    )

    with pytest.raises(ArtifactTransferError, match="unknown target worker"):
        await manager.ensure_local(artifact, "worker-b")
