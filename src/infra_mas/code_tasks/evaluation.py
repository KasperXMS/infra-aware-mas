"""Official SWE-bench evaluation adapter for submitted Planner patches."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import cast


def evaluate_run(
    run_directory: Path,
    *,
    task_id: str,
    python_executable: Path,
    evaluator_workdir: Path,
) -> dict[str, object]:
    patch_path = run_directory / "model.patch"
    if not patch_path.is_file() or not patch_path.read_text(encoding="utf-8").strip():
        raise ValueError(f"run has no submitted patch: {run_directory}")
    prediction = run_directory / "prediction.jsonl"
    model_name = f"infra-aware-mas__{run_directory.name}"
    prediction.write_text(
        json.dumps(
            {
                "instance_id": task_id,
                "model_name_or_path": model_name,
                "model_patch": patch_path.read_text(encoding="utf-8"),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    run_id = f"semantic-switch-{run_directory.name}"
    command = [
        str(python_executable),
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        "SWE-bench/SWE-bench_Verified",
        "--split",
        "test",
        "--predictions_path",
        str(prediction),
        "--max_workers",
        "1",
        "--run_id",
        run_id,
        "--instance_ids",
        task_id,
    ]
    completed = subprocess.run(  # noqa: S603
        command,
        cwd=evaluator_workdir,
        check=False,
        text=True,
        capture_output=True,
    )
    candidates = sorted(evaluator_workdir.glob(f"{model_name}*.{run_id}.json"))
    if not candidates:
        candidates = sorted(evaluator_workdir.glob(f"*.{run_id}.json"))
    if not candidates:
        raise RuntimeError(
            f"SWE-bench evaluator produced no report (exit {completed.returncode}): "
            f"{completed.stderr[-2000:]}"
        )
    raw: object = json.loads(candidates[-1].read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("SWE-bench evaluator report must be an object")
    report = dict(cast(dict[str, object], raw))
    report["evaluator_exit_code"] = completed.returncode
    report["evaluator_run_id"] = run_id
    destination = run_directory / "official_evaluation.json"
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    log_destination = run_directory / "official_evaluation.log"
    log_destination.write_text(
        f"STDOUT:\n{completed.stdout}\n\nSTDERR:\n{completed.stderr}", encoding="utf-8"
    )
    for candidate in candidates:
        if candidate.is_file() and candidate != destination:
            shutil.copy2(candidate, run_directory / "official_raw_report.json")
            break
    return report
