"""Unit tests for planner-side artifact reference and inspection handling."""

from pathlib import Path

import httpx
import pytest

from infra_mas.core.artifact import ArtifactRef
from infra_mas.core.errors import ArtifactNotFoundError
from infra_mas.execution.worker_client import WorkerClient
from infra_mas.planner.context import ArtifactCatalog
from infra_mas.worker.artifact_store import ArtifactStore
from infra_mas.worker.server import create_app
from infra_mas.worker.service import WorkerService


async def test_catalog_resolves_and_inspects_small_text(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "worker", "worker-1")
    artifact = await store.put_text("run-001/evidence-001", "small evidence")
    app = create_app(WorkerService("worker-1", store, []))

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://worker.test",
    ) as http_client:
        catalog = ArtifactCatalog(
            {"worker-1": WorkerClient(client=http_client)},
            tmp_path / "inspection",
        )
        catalog.register(artifact)
        text = await catalog.inspect_text(artifact.id)

    assert text == "small evidence"
    assert catalog.get(artifact.id) is artifact


async def test_catalog_rejects_binary_and_oversized_artifacts(tmp_path: Path) -> None:
    catalog = ArtifactCatalog({}, tmp_path / "inspection", max_inspect_bytes=4)
    binary = ArtifactRef(
        id="run-001/video-001",
        artifact_type="video/mp4",
        size_bytes=4,
        locations=["worker-1"],
    )
    oversized = ArtifactRef(
        id="run-001/text-001",
        artifact_type="text/plain",
        size_bytes=5,
        locations=["worker-1"],
    )
    catalog.register_many([binary, oversized])

    with pytest.raises(ValueError, match="binary or exceeds"):
        await catalog.inspect_text(binary.id)
    with pytest.raises(ValueError, match="binary or exceeds"):
        await catalog.inspect_text(oversized.id)


def test_catalog_merges_locations_and_rejects_conflicts(tmp_path: Path) -> None:
    catalog = ArtifactCatalog({}, tmp_path / "inspection")
    first = ArtifactRef(
        id="run-001/evidence-001",
        artifact_type="text/plain",
        size_bytes=4,
        locations=["worker-a"],
    )
    second = first.model_copy(update={"locations": ["worker-b"]})
    catalog.register(first)

    merged = catalog.register(second)

    assert merged.locations == ["worker-a", "worker-b"]
    with pytest.raises(ValueError, match="conflicting metadata"):
        catalog.register(first.model_copy(update={"size_bytes": 5}))


def test_catalog_unknown_artifact_is_typed(tmp_path: Path) -> None:
    catalog = ArtifactCatalog({}, tmp_path / "inspection")

    with pytest.raises(ArtifactNotFoundError, match="unknown artifact"):
        catalog.get("run-001/missing")
