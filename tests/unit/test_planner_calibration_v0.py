from pathlib import Path
from typing import Any

import pytest

from infra_mas.execution.executor_registry import ExecutorRegistry
from infra_mas.execution.worker_client import WorkerClient
from infra_mas.experiment import PreflightResult
from infra_mas.planner_calibration_v0 import (
    PlannerTask795,
    preflight_planner_jetsons,
    run_planner_experiment_cell,
    validate_planner_experiment_cells,
)


def _configs() -> list[Path]:
    root = Path(__file__).parents[2] / "configs" / "calibration_v0" / "planner"
    return [
        root / "795-h1-blind.yaml",
        root / "795-h1-aware.yaml",
        root / "795-h2-blind.yaml",
        root / "795-h2-aware.yaml",
    ]


def test_four_planner_cells_are_controlled_and_gold_free() -> None:
    cells = validate_planner_experiment_cells(_configs())

    assert {(cell.world_id, cell.arm) for cell in cells} == {
        ("H1_distributed_constrained", "blind"),
        ("H1_distributed_constrained", "aware"),
        ("H2_distributed_favorable", "blind"),
        ("H2_distributed_favorable", "aware"),
    }
    assert {cell.infrastructure_visibility for cell in cells if cell.arm == "blind"} == {
        "none"
    }
    assert {cell.infrastructure_visibility for cell in cells if cell.arm == "aware"} == {
        "snapshot"
    }


def test_task_795_preserves_three_original_mp4_placements() -> None:
    task_path = _configs()[0].parent / "task-795.yaml"
    task = PlannerTask795.from_yaml(task_path)

    assert [item.site_id for item in task.artifacts] == ["A4", "A5", "A28"]
    assert [item.chunk_index for item in task.artifacts] == [0, 1, 2]
    assert all(item.source_ref.endswith(".mp4") for item in task.artifacts)
    assert all(
        str(item.source_ref).startswith(
            "/home/edge/xiaoming/calibration_v0/artifacts/795/"
        )
        for item in task.artifacts
    )
    assert task.frames_per_chunk == 12
    assert task.frame_width == 320
    assert [item.size_bytes for item in task.artifacts] == [
        93677242,
        99396939,
        89370679,
    ]


async def test_planner_cell_runner_binds_the_three_worker_local_mp4s(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed: dict[str, Any] = {}

    async def fake_run(
        config_path: Path,
        task: str,
        inputs: list[Path],
        **kwargs: object,
    ) -> tuple[str, Path]:
        observed.update(
            config_path=config_path,
            task=task,
            inputs=inputs,
            kwargs=kwargs,
        )
        return "answer", tmp_path / "run"

    monkeypatch.setattr(
        "infra_mas.planner_calibration_v0.run_blind_experiment", fake_run
    )
    answer, _ = await run_planner_experiment_cell(_configs()[0], run_id="planner-test")

    assert answer == "answer"
    assert [path.name for path in observed["inputs"]] == [
        "chunk-0.mp4",
        "chunk-1.mp4",
        "chunk-2.mp4",
    ]
    kwargs = observed["kwargs"]
    assert kwargs["input_workers"] == ["a4", "a5", "a28"]
    assert kwargs["artifact_ids"] == [
        "video_mme:795:chunk:0",
        "video_mme:795:chunk:1",
        "video_mme:795:chunk:2",
    ]
    assert kwargs["input_expected_sizes"] == [93677242, 99396939, 89370679]


async def test_preflight_builds_jetson_only_registry_and_marks_4090_unchecked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_workers: set[str] = set()

    def fake_create_clients(
        registry: ExecutorRegistry, timeout: float
    ) -> dict[str, WorkerClient]:
        del timeout
        endpoints = registry.worker_endpoints()
        observed_workers.update(endpoints)
        return {}

    async def fake_preflight(
        registry: ExecutorRegistry, clients: object
    ) -> PreflightResult:
        del registry, clients
        operators = [
            "sample_frames",
            "make_contact_sheet",
            "aggregate_artifacts",
        ]
        return PreflightResult(
            workers={"a4": ["a4-vlm"], "a5": ["a5-vlm"], "a28": ["a28-vlm"]},
            operators={worker_id: operators for worker_id in ("a4", "a5", "a28")},
        )

    async def fake_close(clients: object) -> None:
        del clients

    monkeypatch.setattr(
        "infra_mas.planner_calibration_v0.create_worker_clients", fake_create_clients
    )
    monkeypatch.setattr(
        "infra_mas.planner_calibration_v0.preflight_workers", fake_preflight
    )
    monkeypatch.setattr(
        "infra_mas.planner_calibration_v0.close_worker_clients", fake_close
    )

    report = await preflight_planner_jetsons(_configs())

    assert observed_workers == {"a4", "a5", "a28"}
    assert report.strong_4090["checked"] is False
    assert report.strong_4090["status"] == "expected_unavailable"
    assert report.capability_validation["status"] == "ready_for_full_preflight"
    assert report.capability_validation["missing_jetson_operators"] == {}
    assert report.capability_validation["missing_optional_jetson_operators"] == {
        "a4": ["extract_clip"],
        "a5": ["extract_clip"],
        "a28": ["extract_clip"],
    }
