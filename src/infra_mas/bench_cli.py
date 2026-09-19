"""CLI for executing one oracle-free infra-bench world."""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import cast

from infra_mas.bench import run_benchmark_case
from infra_mas.calibration import run_calibration_sweep
from infra_mas.calibration_v0 import run_calibration_v0_sweep
from infra_mas.local_profile import run_local_profiles
from infra_mas.longbench_v2 import generate_longbench_v2_sweep, run_longbench_v2_sweep
from infra_mas.network_microbench import run_network_microbench
from infra_mas.planner.context import InfrastructureVisibility
from infra_mas.planner_calibration_v0 import (
    preflight_planner_jetsons,
    run_planner_experiment_cell,
    validate_planner_experiment_cells,
)
from infra_mas.scope_expansion_v0 import (
    generate_scope_expansion_sweep,
    run_scope_expansion_v0_sweep,
)
from infra_mas.semantic_switch import write_v1_semantic_switch_report
from infra_mas.stability import write_v1_repeat_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an exported infra-bench MAS case")
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--visibility", choices=("none", "static", "snapshot"), required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--quiet", action="store_true", help="print only the run directory")
    return parser


def build_calibration_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="infra-mas-bench calibrate",
        description="Run hidden reference workflows across real infrastructure worlds",
    )
    parser.add_argument("--sweep", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def build_calibration_v0_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="infra-mas-bench calibrate-v0",
        description="Run the two fixed long-video calibration_v0 references",
    )
    parser.add_argument("--sweep", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("runs/calibration_v0"))
    return parser


def build_local_profile_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="infra-mas-bench profile-local-v0",
        description="Profile calibration_v0 local reduction on Jetsons only",
    )
    parser.add_argument("--sweep", type=Path, required=True)
    parser.add_argument("--raw-runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def build_scope_expansion_v0_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="infra-mas-bench scope-expansion-v0",
        description="Run the Jetson-only MultiHop-RAG reference workflows",
    )
    parser.add_argument("--sweep", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/scope_expansion_v0/multihop_rag"),
    )
    return parser


def build_prepare_scope_expansion_v0_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="infra-mas-bench prepare-scope-expansion-v0",
        description="Generate a gold-free MAS sweep from a Scope Expansion task bank",
    )
    parser.add_argument("--task-bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--remote-root",
        type=Path,
        default=Path("/home/edge/xiaoming/scope_expansion_v0"),
    )
    return parser


def build_longbench_v2_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="infra-mas-bench longbench-v2",
        description="Run the Jetson-only LongBench-v2 reference workflows",
    )
    parser.add_argument("--sweep", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/scope_expansion_v0"),
    )
    return parser


def build_prepare_longbench_v2_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="infra-mas-bench prepare-longbench-v2",
        description="Generate a gold-free LongBench-v2 sweep from a task bank",
    )
    parser.add_argument("--task-bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--remote-root",
        type=Path,
        default=Path("/home/edge/xiaoming/scope_expansion_v0"),
    )
    return parser


def build_network_microbench_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="infra-mas-bench network-microbench-v0",
        description="Measure all Jetson-to-Jetson directions and concurrent flows",
    )
    parser.add_argument("--executors", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def build_planner_v0_check_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="infra-mas-bench check-planner-v0",
        description="Validate the four Task 795 Planner cells without running a Planner",
    )
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--jetson-preflight", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--output", type=Path)
    return parser


def build_planner_v0_run_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="infra-mas-bench run-planner-v0",
        description="Run one Task 795 Planner cell with Worker-local input binding",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-id")
    return parser


def build_stability_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="infra-mas-bench summarize-v1",
        description="Aggregate repeated V1 benchmark run directories",
    )
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-image-id", default="img_06")
    return parser


def build_semantic_switch_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="infra-mas-bench summarize-semantic-switch",
        description="Aggregate the paired-world V1 Planner experiment",
    )
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-image-id", default="img_06")
    return parser


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "prepare-longbench-v2":
        args = build_prepare_longbench_v2_parser().parse_args(sys.argv[2:])
        config = generate_longbench_v2_sweep(
            args.task_bank,
            args.output,
            remote_root=args.remote_root,
        )
        print(
            json.dumps(
                {"output": str(args.output), "task_count": len(config.tasks)},
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if len(sys.argv) > 1 and sys.argv[1] == "longbench-v2":
        args = build_longbench_v2_parser().parse_args(sys.argv[2:])
        payload = asyncio.run(run_longbench_v2_sweep(args.sweep, args.output))
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    if len(sys.argv) > 1 and sys.argv[1] == "prepare-scope-expansion-v0":
        args = build_prepare_scope_expansion_v0_parser().parse_args(sys.argv[2:])
        config = generate_scope_expansion_sweep(
            args.task_bank,
            args.output,
            remote_root=args.remote_root,
        )
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "task_count": len(config.tasks),
                    "candidate_top_n": config.candidate_top_n,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if len(sys.argv) > 1 and sys.argv[1] == "scope-expansion-v0":
        args = build_scope_expansion_v0_parser().parse_args(sys.argv[2:])
        payload = asyncio.run(run_scope_expansion_v0_sweep(args.sweep, args.output))
        rendered = json.dumps(payload, ensure_ascii=False, indent=2)
        encoding = sys.stdout.encoding or "utf-8"
        print(rendered.encode(encoding, errors="backslashreplace").decode(encoding))
        return
    if len(sys.argv) > 1 and sys.argv[1] == "run-planner-v0":
        args = build_planner_v0_run_parser().parse_args(sys.argv[2:])
        answer, run_directory = asyncio.run(
            run_planner_experiment_cell(args.manifest, run_id=args.run_id)
        )
        print(json.dumps({"answer": answer, "run_directory": str(run_directory)}))
        return
    if len(sys.argv) > 1 and sys.argv[1] == "network-microbench-v0":
        args = build_network_microbench_parser().parse_args(sys.argv[2:])
        payload = asyncio.run(run_network_microbench(args.executors, args.output))
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    if len(sys.argv) > 1 and sys.argv[1] == "check-planner-v0":
        args = build_planner_v0_check_parser().parse_args(sys.argv[2:])
        if args.jetson_preflight:
            report = asyncio.run(
                preflight_planner_jetsons(args.manifest, timeout_seconds=args.timeout_seconds)
            ).model_dump(mode="json")
        else:
            cells = validate_planner_experiment_cells(args.manifest)
            report = {
                "schema_version": "calibration-v0-planner-validation-v1",
                "config_validation": "passed",
                "artifact_placement_validation": "passed",
                "cells": [cell.experiment_id for cell in cells],
                "strong_4090": {
                    "status": "expected_unavailable",
                    "checked": False,
                },
                "capability_validation": {
                    "status": "config_ready",
                    "generic_media_operators_configured": True,
                    "direct_video_configured": True,
                    "worker_local_source_binding_supported": True,
                },
            }
        rendered = json.dumps(report, ensure_ascii=False, indent=2)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n", encoding="utf-8")
        encoding = sys.stdout.encoding or "utf-8"
        print(rendered.encode(encoding, errors="backslashreplace").decode(encoding))
        return
    if len(sys.argv) > 1 and sys.argv[1] == "profile-local-v0":
        args = build_local_profile_parser().parse_args(sys.argv[2:])
        payload = asyncio.run(run_local_profiles(args.sweep, args.raw_runs, args.output))
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    if len(sys.argv) > 1 and sys.argv[1] == "calibrate-v0":
        args = build_calibration_v0_parser().parse_args(sys.argv[2:])
        payload = asyncio.run(run_calibration_v0_sweep(args.sweep, args.output))
        rendered = json.dumps(payload, ensure_ascii=False, indent=2)
        encoding = sys.stdout.encoding or "utf-8"
        print(rendered.encode(encoding, errors="backslashreplace").decode(encoding))
        return
    if len(sys.argv) > 1 and sys.argv[1] == "calibrate":
        args = build_calibration_parser().parse_args(sys.argv[2:])
        payload = asyncio.run(run_calibration_sweep(args.sweep, args.output))
        rendered = json.dumps(payload, ensure_ascii=False, indent=2)
        encoding = sys.stdout.encoding or "utf-8"
        print(rendered.encode(encoding, errors="backslashreplace").decode(encoding))
        return
    if len(sys.argv) > 1 and sys.argv[1] == "summarize-v1":
        args = build_stability_parser().parse_args(sys.argv[2:])
        payload = write_v1_repeat_report(
            args.run, args.output, expected_image_id=args.expected_image_id
        )
        rendered = json.dumps(payload, ensure_ascii=False, indent=2)
        encoding = sys.stdout.encoding or "utf-8"
        print(rendered.encode(encoding, errors="backslashreplace").decode(encoding))
        return
    if len(sys.argv) > 1 and sys.argv[1] == "summarize-semantic-switch":
        args = build_semantic_switch_parser().parse_args(sys.argv[2:])
        payload = write_v1_semantic_switch_report(
            args.run, args.output, expected_image_id=args.expected_image_id
        )
        rendered = json.dumps(payload, ensure_ascii=False, indent=2)
        encoding = sys.stdout.encoding or "utf-8"
        print(rendered.encode(encoding, errors="backslashreplace").decode(encoding))
        return
    args = build_parser().parse_args()
    result, run_directory = asyncio.run(
        run_benchmark_case(
            args.case,
            args.config,
            cast(InfrastructureVisibility, args.visibility),
            run_id=args.run_id,
        )
    )
    if not args.quiet:
        rendered = json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2)
        encoding = sys.stdout.encoding or "utf-8"
        print(rendered.encode(encoding, errors="backslashreplace").decode(encoding))
    print(f"Run artifacts: {run_directory}")
