from contextlib import AsyncExitStack
from pathlib import Path
from typing import Protocol, cast

import httpx
import pytest
from PIL import Image

from infra_mas.core.artifact import ArtifactRef
from infra_mas.core.errors import ExecutionFailedError
from infra_mas.core.execution import (
    BindLocalArtifactRequest,
    MakeContactSheetRequest,
)
from infra_mas.core.model import ModelSpec
from infra_mas.execution.executor_registry import ExecutorRegistry
from infra_mas.execution.manager import ExecutionManager
from infra_mas.execution.transfer import TransferManager
from infra_mas.execution.worker_client import WorkerClient
from infra_mas.planner.context import ArtifactCatalog, PlannerContext
from infra_mas.planner.coordinator import Coordinator
from infra_mas.runtime.model_registry import ModelRegistry
from infra_mas.runtime.runtime import AgentRuntime
from infra_mas.scheduler.round_robin import RoundRobinScheduler
from infra_mas.tracing.recorder import TraceRecorder
from infra_mas.worker.artifact_store import ArtifactStore
from infra_mas.worker.server import create_app
from infra_mas.worker.service import WorkerService


class _FunctionToolLike(Protocol):
    name: str
    params_json_schema: dict[str, object]


def _service(
    root: Path,
    worker_id: str,
    *,
    source_roots: list[Path] | None = None,
) -> WorkerService:
    return WorkerService(
        worker_id,
        ArtifactStore(root / "store", worker_id),
        [],
        local_source_roots=source_roots or [],
    )


async def test_worker_local_binding_is_allowlisted_and_never_uploads_bytes(
    tmp_path: Path,
) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    source = allowed / "chunk.mp4"
    source.write_bytes(b"local-video")
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"outside")
    service = _service(tmp_path / "worker", "a4", source_roots=[allowed])

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_app(service)),
        base_url="http://a4.test",
    ) as http_client:
        client = WorkerClient(client=http_client)
        bound = await client.bind_local_artifact(
            BindLocalArtifactRequest(
                artifact_id="run/input-001.mp4",
                source_path=str(source),
                artifact_type="video/mp4",
            )
        )
        with pytest.raises(ExecutionFailedError, match="outside allowlisted roots"):
            await client.bind_local_artifact(
                BindLocalArtifactRequest(
                    artifact_id="run/input-002.mp4",
                    source_path=str(outside),
                    artifact_type="video/mp4",
                )
            )
        with pytest.raises(ExecutionFailedError, match="expected 999"):
            await client.bind_local_artifact(
                BindLocalArtifactRequest(
                    artifact_id="run/input-wrong-size.mp4",
                    source_path=str(source),
                    artifact_type="video/mp4",
                    expected_size_bytes=999,
                )
            )

    assert bound.locations == ["a4"]
    assert bound.size_bytes == len(b"local-video")


async def test_runtime_aggregates_on_artifact_local_worker_without_transfer(
    tmp_path: Path,
) -> None:
    service_a = _service(tmp_path / "a", "a")
    service_b = _service(tmp_path / "b", "b")
    evidence = await service_b.artifact_store.put_text("run/evidence.txt", "observed event")
    trace = TraceRecorder(tmp_path / "runs", "run")
    await trace.start()

    async with AsyncExitStack() as stack:
        http_a = await stack.enter_async_context(
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=create_app(service_a)),
                base_url="http://a.test",
            )
        )
        http_b = await stack.enter_async_context(
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=create_app(service_b)),
                base_url="http://b.test",
            )
        )
        clients = {"a": WorkerClient(client=http_a), "b": WorkerClient(client=http_b)}
        models = ModelRegistry(
            [
                ModelSpec(
                    model_id="unused",
                    description="Unused test model.",
                    input_modalities=["text"],
                    output_modalities=["text"],
                    context_window=1024,
                )
            ]
        )
        registry = ExecutorRegistry([], {"a": "http://a.test", "b": "http://b.test"})
        runtime = AgentRuntime(
            None,
            RoundRobinScheduler(registry),
            ExecutionManager(clients, TransferManager(clients, trace), trace),
            trace,
            model_registry=models,
        )
        result = await runtime.aggregate_artifacts([evidence])

    assert result.output_artifacts[0].locations == ["b"]
    assert result.transfer_ms == 0
    events = [line for line in trace.path.read_text(encoding="utf-8").splitlines()]
    assert not any('"event_type":"artifact.transfer.start"' in line for line in events)
    assert any('"semantic_operator":"aggregate_artifacts"' in line for line in events)


async def test_contact_sheet_is_a_separate_generic_worker_action(tmp_path: Path) -> None:
    service = _service(tmp_path / "worker", "a4")
    inputs: list[ArtifactRef] = []
    for index, color in enumerate(("red", "blue"), start=1):
        image_path = tmp_path / f"{index}.jpg"
        Image.new("RGB", (16, 12), color=color).save(image_path)
        inputs.append(
            await service.artifact_store.put_bytes(
                f"run/frame-{index}.jpg", image_path.read_bytes(), "image/jpeg"
            )
        )
    result = await service.make_contact_sheet(
        MakeContactSheetRequest(
            request_id="run/sheet-request",
            input_artifacts=inputs,
            output_artifact_id="run/sheet.jpg",
            columns=2,
        )
    )
    assert result.executor_id == "a4:make_contact_sheet"
    assert result.output_artifacts[0].artifact_type == "image/jpeg"


def test_coordinator_exposes_logical_media_tools_without_worker_arguments(
    tmp_path: Path,
) -> None:
    trace = TraceRecorder(tmp_path / "runs", "tools")
    models = ModelRegistry(
        [
            ModelSpec(
                model_id="model",
                description="Test model.",
                input_modalities=["text"],
                output_modalities=["text"],
                context_window=1024,
            )
        ]
    )
    registry = ExecutorRegistry([], {})
    runtime = AgentRuntime(
        None,
        RoundRobinScheduler(registry),
        ExecutionManager({}, TransferManager({}, trace), trace),
        trace,
        model_registry=models,
    )
    context = PlannerContext(
        runtime,
        None,
        ArtifactCatalog({}, tmp_path / "inspection", trace),
        trace,
        model_registry=models,
        planner_mode="dynamic_models",
    )
    coordinator = Coordinator(context, model="unused")
    tools = {tool.name: tool for tool in coordinator.agent.tools}
    assert {
        "sample_frames",
        "make_contact_sheet",
        "extract_clip",
        "process_local_artifact",
        "aggregate_artifacts",
    } <= tools.keys()
    for raw_tool in tools.values():
        tool = cast(_FunctionToolLike, raw_tool)
        properties = cast(dict[str, object], tool.params_json_schema.get("properties", {}))
        assert "worker" not in properties
