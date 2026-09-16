"""CLI for executing one oracle-free infra-bench world."""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import cast

from infra_mas.bench import run_benchmark_case
from infra_mas.calibration import run_calibration_sweep
from infra_mas.planner.context import InfrastructureVisibility
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


def build_stability_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="infra-mas-bench summarize-v1",
        description="Aggregate repeated V1 benchmark run directories",
    )
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-image-id", default="img_06")
    return parser


def main() -> None:
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
