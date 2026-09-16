"""Command-line entry point for the real-system pilot."""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from infra_mas.pilot import run_calibration, run_paired_planners


def _print_json(payload: object) -> None:
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    encoding = sys.stdout.encoding or "utf-8"
    print(rendered.encode(encoding, errors="backslashreplace").decode(encoding))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the paired infrastructure-awareness pilot")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("calibration", "planner", "all"),
        default="all",
    )
    return parser


async def _run(config: Path, mode: str) -> None:
    if mode in {"calibration", "all"}:
        calibration = await run_calibration(config)
        _print_json(calibration.model_dump(mode="json"))
    if mode in {"planner", "all"}:
        planner = await run_paired_planners(config)
        _print_json(
            {key: value.model_dump(mode="json") for key, value in planner.items()}
        )


def main() -> None:
    args = build_parser().parse_args()
    asyncio.run(_run(args.config, args.mode))


if __name__ == "__main__":
    main()
