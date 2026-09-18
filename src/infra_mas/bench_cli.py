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
from infra_mas.network_microbench import run_network_microbench
from infra_mas.planner.context import InfrastructureVisibility
from infra_mas.planner_calibration_v0 import (
    preflight_planner_jetsons,
    validate_planner_experiment_cells,
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
    if len(sys.argv) > 1 and sys.argv[1] == "network-microbench-v0":
        args = build_network_microbench_parser().parse_args(sys.argv[2:])
        payload = asyncio.run(run_network_microbench(args.executors, args.output))
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    if len(sys.argv) > 1 and sys.argv[1] == "check-planner-v0":
        args = build_planner_v0_check_parser().parse_args(sys.argv[2:])
        if args.jetson_preflight:
            report = asyncio.run(
                preflight_planner_jetsons(
                    args.manifest, timeout_seconds=args.timeout_seconds
                )
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
                    "status": "execution_blocked",
                    "missing_planner_operator": "sample_frames",
                    "direct_video_supported": False,
                    "worker_local_source_binding_supported": False,
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
