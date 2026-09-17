"""Aggregate the paired open-ended SWE-bench semantic-switch experiment."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import cast

import yaml

from infra_mas.code_tasks.experiment import CodeRunResult


def _resolved(run_directory: Path) -> bool | None:
    path = run_directory / "official_evaluation.json"
    if not path.is_file():
        result_path = run_directory / "code_result.json"
        if result_path.is_file():
            result: object = json.loads(result_path.read_text(encoding="utf-8"))
            if (
                isinstance(result, dict)
                and cast(dict[str, object], result).get("submitted_patch") is False
            ):
                return False
        return None
    raw: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return None
    values = cast(dict[str, object], raw)
    resolved = values.get("resolved_ids", [])
    return isinstance(resolved, list) and "astropy__astropy-14309" in resolved


def _signature(row: dict[str, object]) -> str:
    workflow = cast(list[dict[str, object]], row["realized_workflow"])
    return " -> ".join(str(item["tool"]) for item in workflow)


def _formal_result_paths(runs_root: Path) -> list[Path]:
    """Return only the preregistered formal repetitions, never smoke attempts."""

    return sorted(runs_root.glob("formal-*/code_result.json"))


def _as_float(value: object) -> float:
    return float(cast(float | int | str, value))


def _median(rows: list[dict[str, object]], field: str) -> float:
    return float(median(_as_float(item[field]) for item in rows))


def build_semantic_switch_report(runs_root: Path, output_directory: Path) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    formal_result_paths = _formal_result_paths(runs_root)
    configs_by_run: dict[str, dict[str, object]] = {
        path.parent.name: cast(
            dict[str, object],
            yaml.safe_load((path.parent / "config.yaml").read_text(encoding="utf-8")),
        )
        for path in formal_result_paths
    }
    for result_path in formal_result_paths:
        result = CodeRunResult.model_validate_json(result_path.read_text(encoding="utf-8"))
        row = result.model_dump(mode="json")
        row["official_resolved"] = _resolved(result_path.parent)
        row["workflow_signature"] = _signature(row)
        rows.append(row)
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["world_id"]), str(row["visibility"]))].append(row)

    cells: list[dict[str, object]] = []
    for (world, visibility), cell_rows in sorted(grouped.items()):
        evaluated = [row for row in cell_rows if row["official_resolved"] is not None]
        signatures = Counter(str(row["workflow_signature"]) for row in cell_rows)
        cells.append(
            {
                "world_id": world,
                "visibility": visibility,
                "run_count": len(cell_rows),
                "evaluated_count": len(evaluated),
                "resolved_rate": (
                    sum(row["official_resolved"] is True for row in evaluated) / len(evaluated)
                    if evaluated
                    else None
                ),
                "median_e2e_ms": _median(cell_rows, "e2e_ms"),
                "median_planner_tokens": _median(cell_rows, "planner_tokens"),
                "median_planner_latency_ms": _median(cell_rows, "planner_latency_ms"),
                "median_remote_reasoning_calls": _median(cell_rows, "remote_reasoning_calls"),
                "median_targeted_tests": _median(cell_rows, "targeted_test_count"),
                "median_full_tests": _median(cell_rows, "full_test_count"),
                "median_retry_replanning": _median(cell_rows, "retry_replanning_count"),
                "median_cross_site_transfer_bytes": _median(cell_rows, "cross_site_transfer_bytes"),
                "median_cross_site_transfer_ms": _median(cell_rows, "cross_site_transfer_ms"),
                "median_transmitted_code_context_bytes": _median(
                    cell_rows, "transmitted_code_context_bytes"
                ),
                "median_code_tool_service_ms": _median(cell_rows, "code_tool_service_ms"),
                "tool_count_medians": {
                    tool: float(
                        median(
                            int(cast(dict[str, int], row["tool_counts"]).get(tool, 0))
                            for row in cell_rows
                        )
                    )
                    for tool in sorted(
                        {
                            tool
                            for row in cell_rows
                            for tool in cast(dict[str, int], row["tool_counts"])
                        }
                    )
                },
                "workflow_signatures": dict(signatures),
                "modal_workflow_signature": signatures.most_common(1)[0][0],
            }
        )

    cell_index = {(str(item["world_id"]), str(item["visibility"])): item for item in cells}
    snapshot_cells = {
        world: item for (world, visibility), item in cell_index.items() if visibility == "snapshot"
    }
    h1 = snapshot_cells.get("H1_edge")
    h2 = snapshot_cells.get("H2_cloud")
    structural_fields = (
        "median_remote_reasoning_calls",
        "median_targeted_tests",
        "median_full_tests",
        "median_retry_replanning",
    )
    h1_profile = (
        {
            **{field: h1[field] for field in structural_fields},
            **cast(dict[str, float], h1["tool_count_medians"]),
        }
        if h1
        else None
    )
    h2_profile = (
        {
            **{field: h2[field] for field in structural_fields},
            **cast(dict[str, float], h2["tool_count_medians"]),
        }
        if h2
        else None
    )
    state_sensitive = bool(h1_profile and h2_profile and h1_profile != h2_profile)
    h1_tool_counts = cast(dict[str, float], h1["tool_count_medians"]) if h1 else {}
    h2_tool_counts = cast(dict[str, float], h2["tool_count_medians"]) if h2 else {}
    h1_exploration = h1_tool_counts.get("search_code", 0) + h1_tool_counts.get("read_file", 0)
    h2_exploration = h2_tool_counts.get("search_code", 0) + h2_tool_counts.get("read_file", 0)
    directional_analysis: dict[str, object] = {
        "H1_snapshot_median_search_plus_read": h1_exploration,
        "H2_snapshot_median_search_plus_read": h2_exploration,
        "H1_snapshot_median_context_bytes": (
            h1["median_transmitted_code_context_bytes"] if h1 else None
        ),
        "H2_snapshot_median_context_bytes": (
            h2["median_transmitted_code_context_bytes"] if h2 else None
        ),
        "calibrated_direction_supported": bool(
            h1
            and h2
            and h1_exploration > h2_exploration
            and _as_float(h2["median_transmitted_code_context_bytes"])
            < _as_float(h1["median_transmitted_code_context_bytes"])
        ),
        "interpretation": (
            "The snapshot workflows differ structurally, but the expected clean shift from "
            "more edge-local preprocessing in H1 to less preprocessing in H2 is not supported: "
            "median search+read counts are equal. H2 transmits slightly less code context but "
            "uses more reasoning and edit calls."
        ),
    }
    cost_questions: dict[str, object] = {}
    for world in sorted({key[0] for key in cell_index}):
        static = cell_index.get((world, "static"))
        snapshot = cell_index.get((world, "snapshot"))
        if static is None or snapshot is None:
            continue
        correctness_not_lower = (
            static["resolved_rate"] is not None
            and snapshot["resolved_rate"] is not None
            and _as_float(snapshot["resolved_rate"]) >= _as_float(static["resolved_rate"])
        )
        cost_questions[world] = {
            "correctness_not_lower": correctness_not_lower,
            "snapshot_lower_e2e": (
                _as_float(snapshot["median_e2e_ms"]) < _as_float(static["median_e2e_ms"])
            ),
            "claim_supported": correctness_not_lower
            and _as_float(snapshot["median_e2e_ms"]) < _as_float(static["median_e2e_ms"]),
            "static_median_e2e_ms": static["median_e2e_ms"],
            "snapshot_median_e2e_ms": snapshot["median_e2e_ms"],
            "relative_e2e_change": (
                _as_float(snapshot["median_e2e_ms"]) - _as_float(static["median_e2e_ms"])
            )
            / _as_float(static["median_e2e_ms"]),
        }

    leakage_terms = [
        "whole_repo_remote_reasoning",
        "local_search_test_compact_remote_reasoning",
        "calibration margin",
        "oracle winner",
        "fail_to_pass",
        "pass_to_pass",
        "verified patch",
        "ground truth",
        "evaluator configuration",
    ]
    leakage_hits: list[dict[str, str]] = []
    static_contexts: dict[str, str] = {}
    dynamic_snapshots: dict[str, object] = {}
    for run_directory in [path.parent for path in formal_result_paths]:
        trace_events = [
            json.loads(line)
            for line in (run_directory / "trace.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        initial = [
            item
            for item in trace_events
            if item.get("event_type") == "planner.infrastructure_context"
        ]
        if initial:
            context_event = initial[0]
            static_contexts[run_directory.name] = str(context_event.get("static_context", ""))
            dynamic_snapshots[run_directory.name] = context_event.get("dynamic_snapshot")
        checked = json.dumps(initial, ensure_ascii=False).lower()
        for term in leakage_terms:
            if term.lower() in checked:
                leakage_hits.append({"run_id": run_directory.name, "term": term})

    world_facts = {
        world_id: next(
            cast(dict[str, object], config["world"])
            for config in configs_by_run.values()
            if cast(dict[str, object], config["world"])["world_id"] == world_id
        )
        for world_id in sorted({str(row["world_id"]) for row in rows})
    }
    report: dict[str, object] = {
        "schema_version": "semantic-switch-report-v1",
        "task_id": "astropy__astropy-14309",
        "benchmark": "SWE-bench Verified",
        "run_count": len(rows),
        "worlds": world_facts,
        "cells": cells,
        "runs": rows,
        "questions": {
            "G_snapshot_H1_not_equal_G_snapshot_H2": state_sensitive,
            "snapshot_structural_profiles": {
                "H1_edge": h1_profile,
                "H2_cloud": h2_profile,
            },
            "calibrated_direction": directional_analysis,
            "C_snapshot_lower_than_C_static_without_correctness_drop": cost_questions,
        },
        "leakage": {
            "reference_workflow_leakage_count": len(leakage_hits),
            "verified_patch_leakage_count": sum(
                item["term"] == "verified patch" for item in leakage_hits
            ),
            "ground_truth_leakage_count": sum(
                item["term"] == "ground truth" for item in leakage_hits
            ),
            "evaluator_configuration_leakage_count": sum(
                item["term"] == "evaluator configuration" for item in leakage_hits
            ),
            "planner_visible_sources": [
                "frozen oracle-free task instruction",
                "world-invariant static execution context",
                "raw dynamic infrastructure snapshot in snapshot runs only",
                "live generic tool results",
            ],
            "hits": leakage_hits,
        },
        "controls": {
            "formal_run_ids_only": all(str(row["run_id"]).startswith("formal-") for row in rows),
            "static_context_identical_across_worlds": len(set(static_contexts.values())) == 1,
            "snapshot_context_present_only_in_snapshot_runs": all(
                (dynamic_snapshots[str(row["run_id"])] is not None)
                == (str(row["visibility"]) == "snapshot")
                for row in rows
            ),
            "same_scheduler": len({configs_by_run[str(row["run_id"])]["scheduler"] for row in rows})
            == 1,
            "same_planner_and_budgets": len(
                {
                    json.dumps(
                        {
                            key: configs_by_run[str(row["run_id"])][key]
                            for key in (
                                "planner",
                                "max_turns",
                                "planner_max_output_tokens",
                                "max_exploration_calls",
                                "tools",
                            )
                        },
                        sort_keys=True,
                    )
                    for row in rows
                }
            )
            == 1,
        },
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "semantic_switch_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# astropy__astropy-14309 open-ended semantic-switch report",
        "",
        f"Runs discovered: **{len(rows)}**.",
        "",
        "## Materialized worlds",
        "",
        "| World | Repository site | Repository bytes | Bandwidth (Mbps) | RTT (ms) |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for world_id, facts in world_facts.items():
        lines.append(
            f"| {world_id} | {facts['repository_site']} | "
            f"{facts['repository_size_bytes']} | {facts['bandwidth_mbps']} | "
            f"{facts['rtt_ms']} |"
        )
    lines.extend(
        [
            "",
            "## Cell aggregates",
            "",
            "| World | Visibility | Runs | Resolved | Median E2E (ms) | Planner tokens | "
            "Planner latency (ms) | Reasoning calls | Cross-site bytes | Context bytes |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for cell in cells:
        resolved = (
            "n/a" if cell["resolved_rate"] is None else f"{_as_float(cell['resolved_rate']):.1%}"
        )
        lines.append(
            f"| {cell['world_id']} | {cell['visibility']} | {cell['run_count']} | "
            f"{resolved} | {_as_float(cell['median_e2e_ms']):.1f} | "
            f"{_as_float(cell['median_planner_tokens']):.0f} | "
            f"{_as_float(cell['median_planner_latency_ms']):.1f} | "
            f"{_as_float(cell['median_remote_reasoning_calls']):.0f} | "
            f"{_as_float(cell['median_cross_site_transfer_bytes']):.0f} | "
            f"{_as_float(cell['median_transmitted_code_context_bytes']):.0f} |"
        )
    lines.extend(
        [
            "",
            "### Workflow and execution medians",
            "",
            "| World | Visibility | Search | Read | Edit | Apply patch | Targeted tests | "
            "Full tests | Retries | Tool service (ms) | Transfer (ms) |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for cell in cells:
        tool_counts = cast(dict[str, float], cell["tool_count_medians"])
        lines.append(
            f"| {cell['world_id']} | {cell['visibility']} | "
            f"{tool_counts.get('search_code', 0):.0f} | "
            f"{tool_counts.get('read_file', 0):.0f} | "
            f"{tool_counts.get('edit_file', 0):.0f} | "
            f"{tool_counts.get('apply_patch', 0):.0f} | "
            f"{_as_float(cell['median_targeted_tests']):.0f} | "
            f"{_as_float(cell['median_full_tests']):.0f} | "
            f"{_as_float(cell['median_retry_replanning']):.0f} | "
            f"{_as_float(cell['median_code_tool_service_ms']):.1f} | "
            f"{_as_float(cell['median_cross_site_transfer_ms']):.1f} |"
        )
    lines.extend(
        [
            "",
            "## Main questions",
            "",
            f"- `G_snapshot(H1) != G_snapshot(H2)`: **{state_sensitive}**",
            "- Clean shift in the calibrated direction: "
            f"**{directional_analysis['calibrated_direction_supported']}**",
            f"  - H1 snapshot median search+read: **{h1_exploration:.0f}**; "
            f"H2 snapshot: **{h2_exploration:.0f}**.",
            f"  - {directional_analysis['interpretation']}",
        ]
    )
    for world, answer in cast(dict[str, dict[str, object]], cost_questions).items():
        lines.append(
            f"- `{world}: C_snapshot(H) < C_static(H)` without a correctness drop: "
            f"**{answer['claim_supported']}** "
            f"(median E2E change: {_as_float(answer['relative_e2e_change']):+.1%})"
        )
    lines.extend(
        [
            "",
            "## Per-run realized workflows",
            "",
            "| Run | World | Visibility | Resolved | E2E (ms) | Workflow |",
            "| --- | --- | --- | ---: | ---: | --- |",
        ]
    )
    for row in rows:
        workflow_signature = str(row["workflow_signature"]).replace("|", "\\|")
        lines.append(
            f"| {row['run_id']} | {row['world_id']} | {row['visibility']} | "
            f"{row['official_resolved']} | {_as_float(row['e2e_ms']):.1f} | "
            f"{workflow_signature} |"
        )
    lines.extend(
        [
            "",
            "## Leakage audit",
            "",
            f"Reference/evaluator leakage count: **{len(leakage_hits)}**.",
            "",
            "## Experimental controls",
            "",
            *[
                f"- `{name}`: **{value}**"
                for name, value in cast(dict[str, bool], report["controls"]).items()
            ],
            "",
        ]
    )
    (output_directory / "semantic_switch_report.md").write_text("\n".join(lines), encoding="utf-8")
    return report
