"""Unit tests for worker-local artifact storage."""

from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from infra_mas.core.errors import ArtifactNotFoundError
from infra_mas.worker.artifact_store import ArtifactStore


async def test_put_and_resolve_text_artifact(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path, "worker-1")

    artifact = await store.put_text("run-001/evidence-001", "你好, artifact")

    path = await store.get_path(artifact.id)
    assert path.read_text(encoding="utf-8") == "你好, artifact"
    assert artifact.size_bytes == len("你好, artifact".encode())
    assert artifact.artifact_type == "text/plain"
    assert artifact.locations == ["worker-1"]
    assert await store.exists(artifact.id)


async def test_streamed_artifact_is_persisted(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path, "worker-1")

    async def chunks() -> AsyncIterator[bytes]:
        yield b"first"
        yield b"-second"

    artifact = await store.put_stream("run-001/data-001", chunks())

    assert artifact.size_bytes == 12
    assert (await store.get_path(artifact.id)).read_bytes() == b"first-second"


async def test_missing_artifact_raises_typed_error(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path, "worker-1")

    with pytest.raises(ArtifactNotFoundError, match="missing"):
        await store.get_path("run-001/missing")


@pytest.mark.parametrize(
    "artifact_id",
    [".", "../secret", "/absolute", "run//artifact", r"run\\artifact", "C:/data"],
)
async def test_unsafe_artifact_id_is_rejected(tmp_path: Path, artifact_id: str) -> None:
    store = ArtifactStore(tmp_path, "worker-1")

    with pytest.raises(ValueError, match="invalid artifact ID"):
        await store.put_bytes(artifact_id, b"data")
