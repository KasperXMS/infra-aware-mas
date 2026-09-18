from pathlib import Path

import pytest

from infra_mas.execution.executor_registry import ExecutorRegistry
from infra_mas.execution.worker_client import WorkerClient
from infra_mas.experiment import PreflightResult
from infra_mas.planner_calibration_v0 import (
    PlannerTask795,
    preflight_planner_jetsons,
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
        return PreflightResult(
            workers={"a4": ["a4-vlm"], "a5": ["a5-vlm"], "a28": ["a28-vlm"]}
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
    assert report.capability_validation["status"] == "execution_blocked"
