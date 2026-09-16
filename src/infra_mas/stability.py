"""Aggregate repeated V1 MAS runs into a locality-stability report."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any, cast

from infra_mas.bench import MASExecutionResult


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


def _canonical_grouping(groups: list[list[str]]) -> tuple[tuple[str, ...], ...]:
    return tuple(sorted(tuple(sorted(group)) for group in groups))


def _grouping_text(groups: list[list[str]]) -> str:
    return json.dumps([list(group) for group in _canonical_grouping(groups)])


def _is_distributed_locality_pattern(result: MASExecutionResult) -> bool:
    required_pairs = {
        ("img_01", "img_02"),
        ("img_03", "img_04"),
        ("img_05", "img_06"),
    }
    observed = {tuple(sorted(group)) for group in result.artifact_grouping}
    return required_pairs.issubset(observed) and result.site_alignment_rate >= 0.75


def write_v1_repeat_report(
    run_paths: list[Path],
    output_directory: Path,
    *,
    expected_image_id: str = "img_06",
) -> dict[str, object]:
    """Write per-run and grouped stability summaries for the four V1 cells."""
    results = [_load_result(path.resolve()) for path in run_paths]
    if not results:
        raise ValueError("at least one MAS run is required")
    per_run: list[dict[str, object]] = []
    for result in results:
        per_run.append(
            {
                "run_id": result.run_id,
                "case_id": result.case_id,
                "world": (
                    "distributed" if "distributed" in result.case_id else "colocated"
                ),
                "visibility": result.visibility,
                "correct": _correct(result.final_answer, expected_image_id),
                "final_answer": result.final_answer,
                "e2e_ms": result.e2e_ms,
                "planner_tokens": result.planner_tokens,
                "planner_latency_ms": result.planner_latency_ms,
                "vlm_call_count": result.multimodal_invocation_count,
                "service_ms_sum": result.service_ms_sum,
                "cross_worker_transfer_bytes": result.cross_worker_transfer_bytes,
                "cross_worker_transfer_count": result.cross_worker_transfer_count,
                "transfer_ms_sum": result.transfer_ms_sum,
                "site_alignment_rate": result.site_alignment_rate,
                "artifact_grouping": result.artifact_grouping,
            }
        )

    grouped: list[dict[str, object]] = []
    grouped_by_key: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in per_run:
        key = (str(row["world"]), str(row["visibility"]))
        grouped_by_key.setdefault(key, []).append(row)
    for (world, visibility), rows in sorted(grouped_by_key.items()):
        patterns = Counter(
            _grouping_text(cast(list[list[str]], row["artifact_grouping"])) for row in rows
        )
        grouped.append(
            {
                "world": world,
                "visibility": visibility,
                "run_count": len(rows),
                "accuracy": mean(float(bool(row["correct"])) for row in rows),
                "mean_e2e_ms": mean(float(cast(float, row["e2e_ms"])) for row in rows),
                "median_e2e_ms": median(
                    float(cast(float, row["e2e_ms"])) for row in rows
                ),
                "mean_planner_tokens": mean(
                    int(cast(int, row["planner_tokens"])) for row in rows
                ),
                "mean_vlm_calls": mean(
                    int(cast(int, row["vlm_call_count"])) for row in rows
                ),
                "mean_cross_worker_transfer_bytes": mean(
                    int(cast(int, row["cross_worker_transfer_bytes"])) for row in rows
                ),
                "mean_site_alignment_rate": mean(
                    float(cast(float, row["site_alignment_rate"])) for row in rows
                ),
                "dominant_artifact_grouping_patterns": [
                    {"pattern": pattern, "count": count}
                    for pattern, count in patterns.most_common()
                ],
            }
        )

    distributed_static = [
        result
        for result in results
        if "distributed" in result.case_id and result.visibility == "static"
    ]
    distributed_snapshot = [
        result
        for result in results
        if "distributed" in result.case_id and result.visibility == "snapshot"
    ]
    static_rate = (
        mean(_is_distributed_locality_pattern(result) for result in distributed_static)
        if distributed_static
        else 0.0
    )
    snapshot_rate = (
        mean(_is_distributed_locality_pattern(result) for result in distributed_snapshot)
        if distributed_snapshot
        else 0.0
    )
    classification = (
        "validated locality-awareness smoke case"
        if len(distributed_static) >= 3
        and len(distributed_snapshot) >= 3
        and snapshot_rate >= 2 / 3
        and snapshot_rate > static_rate
        else "unstable locality-awareness smoke case"
    )
    payload: dict[str, object] = {
        "task_id": "v1-blue-airplane-and-truck",
        "expected_image_id": expected_image_id,
        "classification": classification,
        "distributed_locality_pattern_rate": {
            "static": static_rate,
            "snapshot": snapshot_rate,
        },
        "groups": grouped,
        "runs": per_run,
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "v1_repeat_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# V1 repeated-run stability report",
        "",
        f"Classification: **{classification}**",
        "",
        "No statistical significance is claimed; this is a small stability pilot.",
        "A distributed run counts as locality-pattern preserving when it contains all three "
        "site-local image pairs and at least 75% of its multimodal calls are site aligned; "
        "later refinement calls may add further groups.",
        "",
        "## Grouped results",
        "",
        "| World | Visibility | Runs | Accuracy | Mean E2E (ms) | Median E2E (ms) | "
        "Mean planner tokens | Mean VLM calls | Mean transfer bytes | "
        "Mean site alignment | Dominant grouping |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in grouped:
        patterns = cast(list[dict[str, Any]], row["dominant_artifact_grouping_patterns"])
        dominant = patterns[0]["pattern"] if patterns else "[]"
        lines.append(
            f"| {row['world']} | {row['visibility']} | {row['run_count']} | "
            f"{row['accuracy']} | {row['mean_e2e_ms']} | {row['median_e2e_ms']} | "
            f"{row['mean_planner_tokens']} | {row['mean_vlm_calls']} | "
            f"{row['mean_cross_worker_transfer_bytes']} | "
            f"{row['mean_site_alignment_rate']} | `{dominant}` |"
        )
    lines.extend(
        [
            "",
            "## Per-run results",
            "",
            "| Run | World | Visibility | Correct | E2E (ms) | Planner tokens | "
            "Planner latency (ms) | VLM calls | Service (ms) | Transfer bytes | "
            "Transfer count | Transfer (ms) | Site alignment | Artifact grouping | "
            "Final answer (abridged) |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | "
            "---: | ---: | ---: | --- | --- |",
        ]
    )
    for row in per_run:
        answer = " ".join(str(row["final_answer"]).split())[:160].replace("|", "\\|")
        lines.append(
            f"| {row['run_id']} | {row['world']} | {row['visibility']} | "
            f"{row['correct']} | {row['e2e_ms']} | {row['planner_tokens']} | "
            f"{row['planner_latency_ms']} | {row['vlm_call_count']} | "
            f"{row['service_ms_sum']} | {row['cross_worker_transfer_bytes']} | "
            f"{row['cross_worker_transfer_count']} | {row['transfer_ms_sum']} | "
            f"{row['site_alignment_rate']} | `{json.dumps(row['artifact_grouping'])}` | "
            f"{answer} |"
        )
    (output_directory / "v1_repeat_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return payload
