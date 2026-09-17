"""Command line entry points for code Workers and paired benchmark runs."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import uvicorn

from infra_mas.code_tasks.evaluation import evaluate_run
from infra_mas.code_tasks.experiment import run_code_benchmark
from infra_mas.code_tasks.report import build_semantic_switch_report
from infra_mas.code_tasks.server import (
    CodeWorkerConfig,
    CodeWorkspaceService,
    create_code_worker_app,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Infra-aware open-ended code benchmark")
    subparsers = parser.add_subparsers(dest="command", required=True)
    worker = subparsers.add_parser("worker")
    worker.add_argument("config", type=Path)
    run = subparsers.add_parser("run")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--world", required=True)
    run.add_argument("--visibility", choices=("static", "snapshot"), required=True)
    run.add_argument("--run-id", required=True)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--run-directory", type=Path, required=True)
    evaluate.add_argument("--task-id", default="astropy__astropy-14309")
    evaluate.add_argument("--python", type=Path, required=True)
    evaluate.add_argument("--workdir", type=Path, required=True)
    report = subparsers.add_parser("report")
    report.add_argument("--runs-root", type=Path, required=True)
    report.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "worker":
        config = CodeWorkerConfig.from_yaml(args.config.resolve())
        app = create_code_worker_app(CodeWorkspaceService(config))
        uvicorn.run(app, host=config.host, port=config.port)
        return
    if args.command == "run":
        result, directory = asyncio.run(
            run_code_benchmark(
                args.config,
                args.world,
                args.visibility,
                run_id=args.run_id,
            )
        )
        print(json.dumps({"result": result.model_dump(mode="json"), "directory": str(directory)}))
        return
    if args.command == "evaluate":
        payload = evaluate_run(
            args.run_directory.resolve(),
            task_id=args.task_id,
            python_executable=args.python.absolute(),
            evaluator_workdir=args.workdir.resolve(),
        )
        print(json.dumps(payload))
        return
    payload = build_semantic_switch_report(args.runs_root.resolve(), args.output.resolve())
    print(json.dumps(payload["questions"]))


if __name__ == "__main__":
    main()
