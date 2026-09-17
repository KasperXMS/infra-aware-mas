"""Aggregate paired V1 semantic-switch Planner runs."""

from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import cast

from infra_mas.bench import MASExecutionResult, RealizedWorkflow


def _load_result(path: Path) -> MASExecutionResult:
    result_path = path / "mas_result.json" if path.is_dir() else path
    return MASExecutionResult.model_validate_json(result_path.read_text(encoding="utf-8"))


def _correct(answer: str, expected_image_id: str) -> bool:
    expected = expected_image_id.lower().replace("-", "_")
    match = re.search(
        r"(?:answer|image\s*id)\s*[:#*` ]+[^\n]*?(img[_-]?[0-9]+)",
        answer,
        flags=re.IGNORECASE,
    )
    return bool(match and match.group(1).lower().replace("-", "_") == expected)


def _world(result: MASExecutionResult) -> str:
    return "H_distributed" if "distributed" in result.case_id else "H_single"


def _workflow_depth(workflow: RealizedWorkflow) -> int:
    parents: dict[str, list[str]] = {}
    for source, target in workflow.dependencies:
        parents.setdefault(target, []).append(source)
    memo: dict[str, int] = {}

    def depth(action_id: str) -> int:
        if action_id not in memo:
            memo[action_id] = 1 + max(
                (depth(parent) for parent in parents.get(action_id, [])), default=0
            )
        return memo[action_id]

    return max((depth(action.action_id) for action in workflow.actions), default=0)


def _max_parallel_actions(workflow: RealizedWorkflow) -> int:
    boundaries: list[tuple[datetime, int]] = []
    for action in workflow.actions:
        if action.started_at is None or action.finished_at is None:
            continue
        boundaries.append((datetime.fromisoformat(action.started_at.replace("Z", "+00:00")), 1))
        boundaries.append((datetime.fromisoformat(action.finished_at.replace("Z", "+00:00")), -1))
    current = 0
    maximum = 0
    for _, delta in sorted(boundaries, key=lambda item: (item[0], item[1])):
        current += delta
        maximum = max(maximum, current)
    return maximum


def _has_synthesis(workflow: RealizedWorkflow) -> bool:
    return bool(workflow.dependencies)


def write_v1_semantic_switch_report(
    run_paths: list[Path],
    output_directory: Path,
    *,
    expected_image_id: str = "img_06",
) -> dict[str, object]:
    """Write per-cell aggregates and per-run phase-aware workflow structure."""
    results = [_load_result(path.resolve()) for path in run_paths]
    if not results:
        raise ValueError("at least one MAS run is required")

    per_run: list[dict[str, object]] = []
    for result in results:
        per_run.append(
            {
                "run_id": result.run_id,
                "case_id": result.case_id,
                "world": _world(result),
                "visibility": result.visibility,
                "correct": _correct(result.final_answer, expected_image_id),
                "final_answer": result.final_answer,
                "e2e_ms": result.e2e_ms,
                "planner_tokens": result.planner_tokens,
                "planner_latency_ms": result.planner_latency_ms,
                "vlm_calls": result.multimodal_invocation_count,
                "service_ms_sum": result.service_ms_sum,
                "transfer_bytes": result.cross_worker_transfer_bytes,
                "transfer_ms_sum": result.transfer_ms_sum,
                "site_alignment_rate": result.site_alignment_rate,
                "artifact_grouping": result.artifact_grouping,
                "initial_artifact_grouping": result.initial_artifact_grouping,
                "later_refinement_calls": result.later_refinement_invocation_count,
                "later_refinement_cross_worker_bytes": (
                    result.later_refinement_cross_worker_bytes
                ),
                "later_refinement_service_ms": result.later_refinement_service_ms,
                "workflow_depth": _workflow_depth(result.realized_workflow),
                "max_parallel_actions": _max_parallel_actions(result.realized_workflow),
                "has_model_synthesis": _has_synthesis(result.realized_workflow),
            }
        )

    by_cell: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in per_run:
        by_cell.setdefault((str(row["world"]), str(row["visibility"])), []).append(row)
    cells: list[dict[str, object]] = []
    for (world, visibility), rows in sorted(by_cell.items()):
        initial_patterns = Counter(
            json.dumps(row["initial_artifact_grouping"], sort_keys=True)
            for row in rows
        )
        cells.append(
            {
                "world": world,
                "visibility": visibility,
                "run_count": len(rows),
                "accuracy": sum(bool(row["correct"]) for row in rows) / len(rows),
                "median_e2e_ms": median(float(cast(float, row["e2e_ms"])) for row in rows),
                "median_planner_tokens": median(
                    int(cast(int, row["planner_tokens"])) for row in rows
                ),
                "median_vlm_calls": median(
                    int(cast(int, row["vlm_calls"])) for row in rows
                ),
                "median_service_ms_sum": median(
                    float(cast(float, row["service_ms_sum"])) for row in rows
                ),
                "median_transfer_bytes": median(
                    int(cast(int, row["transfer_bytes"])) for row in rows
                ),
                "median_site_alignment": median(
                    float(cast(float, row["site_alignment_rate"])) for row in rows
                ),
                "median_later_refinement_calls": median(
                    int(cast(int, row["later_refinement_calls"])) for row in rows
                ),
                "median_later_refinement_cross_worker_bytes": median(
                    int(cast(int, row["later_refinement_cross_worker_bytes"]))
                    for row in rows
                ),
                "median_later_refinement_service_ms": median(
                    float(cast(float, row["later_refinement_service_ms"]))
                    for row in rows
                ),
                "dominant_initial_grouping": (
                    json.loads(initial_patterns.most_common(1)[0][0])
                    if initial_patterns
                    else []
                ),
            }
        )

    payload: dict[str, object] = {
        "task_id": "v1-blue-airplane-and-truck",
        "expected_image_id": expected_image_id,
        "cells": cells,
        "runs": per_run,
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "v1_planner_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# V1 paired-world Planner report",
        "",
        "## Per-cell aggregates",
        "",
        "| World | Visibility | Runs | Accuracy | Median E2E | Planner tokens | VLM calls | "
        "Service | Transfer bytes | Site alignment | Later calls | Later bytes | Later service |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
        "---: | ---: |",
    ]
    for cell in cells:
        lines.append(
            f"| {cell['world']} | {cell['visibility']} | {cell['run_count']} | "
            f"{cell['accuracy']} | {cell['median_e2e_ms']} | "
            f"{cell['median_planner_tokens']} | {cell['median_vlm_calls']} | "
            f"{cell['median_service_ms_sum']} | {cell['median_transfer_bytes']} | "
            f"{cell['median_site_alignment']} | {cell['median_later_refinement_calls']} | "
            f"{cell['median_later_refinement_cross_worker_bytes']} | "
            f"{cell['median_later_refinement_service_ms']} |"
        )
    lines.extend(
        [
            "",
            "## Per-run workflow structure",
            "",
            "The initial phase is the shortest VLM-call prefix that covers every raw image. "
            "Subsequent image calls are classified as refinement/verification.",
            "",
            "| Run | World | Visibility | Correct | Initial grouping | All grouping | "
            "Later calls | "
            "Later bytes | Later service | Depth | Parallelism | Synthesis |",
            "| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in per_run:
        lines.append(
            f"| {row['run_id']} | {row['world']} | {row['visibility']} | "
            f"{row['correct']} | `{json.dumps(row['initial_artifact_grouping'])}` | "
            f"`{json.dumps(row['artifact_grouping'])}` | {row['later_refinement_calls']} | "
            f"{row['later_refinement_cross_worker_bytes']} | "
            f"{row['later_refinement_service_ms']} | {row['workflow_depth']} | "
            f"{row['max_parallel_actions']} | {row['has_model_synthesis']} |"
        )
    (output_directory / "v1_planner_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return payload
