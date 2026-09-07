"""Command-line entry point for resource-blind MAS experiments."""

import argparse
import asyncio
from pathlib import Path

from infra_mas.experiment import check_blind_experiment, run_blind_experiment


def build_parser() -> argparse.ArgumentParser:
    """Create the experiment command-line parser."""
    parser = argparse.ArgumentParser(description="Run a resource-blind MAS experiment")
    parser.add_argument("--config", type=Path, required=True, help="Experiment YAML configuration")
    parser.add_argument("--task", help="User task for the Coordinator")
    parser.add_argument(
        "--input",
        type=Path,
        action="append",
        default=[],
        help="Input artifact path; may be repeated",
    )
    parser.add_argument("--run-id", help="Optional stable run ID")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate configuration and Worker connectivity without invoking models",
    )
    return parser


def main() -> None:
    """Check or run a configured resource-blind experiment."""
    parser = build_parser()
    args = parser.parse_args()
    if args.check:
        result = asyncio.run(check_blind_experiment(args.config))
        for worker_id, executors in result.workers.items():
            print(f"{worker_id}: {', '.join(executors)}")
        return
    if args.task is None:
        parser.error("--task is required unless --check is used")

    answer, run_directory = asyncio.run(
        run_blind_experiment(
            args.config,
            args.task,
            args.input,
            run_id=args.run_id,
        )
    )
    print(answer)
    print(f"\nRun artifacts: {run_directory}")


if __name__ == "__main__":
    main()
