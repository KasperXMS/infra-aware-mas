"""Jetson-only profiling for the generic calibration_v0 local reduction path."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, cast
from uuid import uuid4

from infra_mas.calibration_v0 import (
    CalibrationV0SweepConfig,
    build_video_task_interaction,
)
from infra_mas.core.artifact import ArtifactRef
from infra_mas.core.execution import InvocationSpec
from infra_mas.execution.executor_registry import ExecutorRegistry
from infra_mas.execution.manager import ExecutionManager
from infra_mas.execution.transfer import TransferManager
from infra_mas.execution.worker_client import WorkerClient
from infra_mas.experiment import (
    BlindExperimentConfig,
    LocalityAwareSchedulerConfig,
    build_scheduler,
    resolve_config_path,
)
from infra_mas.resources.provider import StaticResourceConfig, StaticResourceProvider
from infra_mas.runtime.model_registry import ModelRegistry
from infra_mas.runtime.runtime import AgentRuntime
from infra_mas.tracing.recorder import TraceRecorder


def _read_rows(path: Path) -> list[dict[str, Any]]:
    return [
        cast(dict[str, Any], json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _source_artifacts(
    rows: list[dict[str, Any]], task_id: str, workers: list[str]
) -> list[ArtifactRef]:
    matching = next(
        (
            row
            for row in rows
            if row.get("task_id") == task_id
            and row.get("workflow_id") == "local_reduction"
            and row.get("status") == "completed"
        ),
        None,
    )
    if matching is None:
        raise ValueError(f"no completed local_reduction source run for {task_id}")
    records = cast(dict[str, Any], matching["artifacts"])["records"]
    raw = [record for record in records if record.get("kind") == "raw_video_chunk"]
    if len(raw) != len(workers):
        raise ValueError(f"expected {len(workers)} raw artifacts for {task_id}")
    return [
        ArtifactRef(
            id=str(record["artifact_id"]),
            artifact_type="video/mp4",
            size_bytes=int(record["bytes"]),
            locations=[worker_id],
        )
        for record, worker_id in zip(raw, workers, strict=True)
    ]


def _task_prompt(task: Any) -> str:
    if not task.options:
        return str(task.instruction)
    rendered = "\n".join(
        f"{chr(65 + index)}. {option}" for index, option in enumerate(task.options)
    )
    return f"{task.instruction}\nOptions:\n{rendered}\nFinish with `ANSWER: <option letter>`."


async def run_local_profiles(
    sweep_path: Path,
    raw_runs_path: Path,
    output_directory: Path,
) -> dict[str, object]:
    """Profile one generic sample-then-VLM branch per Orin without a strong node."""
    sweep_path = sweep_path.resolve()
    raw_runs_path = raw_runs_path.resolve()
    output_directory = output_directory.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    sweep = CalibrationV0SweepConfig.from_yaml(sweep_path)
    base = sweep_path.parent
    experiment_path = resolve_config_path(sweep.experiment_config, base)
    experiment = BlindExperimentConfig.from_yaml(experiment_path)
    experiment_base = experiment_path.parent
    all_models = ModelRegistry.from_yaml(
        resolve_config_path(experiment.models_config, experiment_base)
    )
    models = ModelRegistry([all_models.get(sweep.local_model_id)])
    full_registry = ExecutorRegistry.from_yaml(
        resolve_config_path(experiment.executors_config, experiment_base)
    )
    local_executor_ids = [f"{worker_id}-vlm" for worker_id in sweep.input_workers]
    local_registry = full_registry.filtered(local_executor_ids)
    resources = StaticResourceConfig.from_yaml(
        resolve_config_path(cast(Path, experiment.resources_config), experiment_base)
    )
    active_ids = set(local_executor_ids)
    provider = StaticResourceProvider(
        local_registry,
        service_times=[item for item in resources.service_times if item.executor_id in active_ids],
        network_links=resources.network_links,
    )
    scheduler = build_scheduler(
        LocalityAwareSchedulerConfig(type="locality_aware"),
        local_registry,
        models,
        None,
        provider,
    )
    clients = {
        worker_id: WorkerClient(
            full_registry.worker_endpoint(worker_id), timeout=1800.0
        )
        for worker_id in sweep.input_workers
    }
    source_rows = _read_rows(raw_runs_path)
    written: list[str] = []
    try:
        for client in clients.values():
            await client.health()
        for task in sweep.tasks:
            run_id = f"local-profile-{task.task_id.replace(':', '-')}-{uuid4().hex[:8]}"
            trace = TraceRecorder(output_directory / "traces", run_id, exclusive=True)
            await trace.start(
                {
                    "mode": "jetson_only_local_profile",
                    "task_id": task.task_id,
                    "strong_node_used": False,
                    "operators": ["sample_frames", "invoke_model"],
                }
            )
            artifacts = _source_artifacts(source_rows, task.task_id, sweep.input_workers)
            prompt = _task_prompt(task)
            build_video_task_interaction(task.task_id, prompt, artifacts, task.evaluator_id)
            manager = ExecutionManager(clients, TransferManager(clients, trace), trace)
            runtime = AgentRuntime(
                None,
                scheduler,
                manager,
                trace,
                request_id_factory=lambda: f"{run_id}/request-{uuid4().hex}",
                model_registry=models,
            )

            async def profile_branch(index: int) -> dict[str, Any]:
                artifact = artifacts[index]
                worker_id = sweep.input_workers[index]
                branch_start = datetime.now(UTC)
                wall_start = perf_counter()
                sampling_start = datetime.now(UTC)
                sampled = await runtime.sample_frames_on_worker(
                    artifact,
                    worker_id,
                    duration_s=task.inputs[index].duration_s,
                    sample_count=sweep.sample_count_per_chunk,
                    frame_width=sweep.frame_width,
                )
                sampling_end = datetime.now(UTC)
                model_start = datetime.now(UTC)
                reduced = await runtime.invoke(
                    InvocationSpec(
                        model_id=sweep.local_model_id,
                        role=f"calibration-v0-local-profile-{index + 1}",
                        instructions=(
                            "You are an intermediate generic video evidence extractor, not "
                            "the final question-answering stage. Preserve timestamps, entities, "
                            "actions, state changes, ordering, and uncertainty. Never answer "
                            "multiple-choice questions or emit a question-ID-to-option mapping. "
                            "Return compact JSON semantic evidence with keys chunk, observations, "
                            "and uncertainty."
                        ),
                        task=(
                            f"Inspect chronological chunk {index + 1} of 3. The following "
                            f"downstream task is relevance context only; do not answer it:\n"
                            f"{prompt}\nReturn only time-localized observations from this chunk "
                            "for a separate final reasoner."
                        ),
                        input_artifacts=sampled.output_artifacts,
                    )
                )
                model_end = datetime.now(UTC)
                branch_end = datetime.now(UTC)
                evidence = reduced.output_artifacts[0]
                return {
                    "task_id": task.task_id,
                    "worker_id": worker_id,
                    "site_id": full_registry.worker_sites()[worker_id],
                    "artifact_id": artifact.id,
                    "video_duration_s": task.inputs[index].duration_s,
                    "input_bytes": artifact.size_bytes,
                    "local_operator_sequence": ["sample_frames", "invoke_model"],
                    "local_model_calls": 1,
                    "local_tool_calls": 1,
                    "preprocessing_wall_ms": (perf_counter() - wall_start) * 1000,
                    "model_service_ms": reduced.service_ms,
                    "tool_service_ms": sampled.service_ms,
                    "stage_latencies": [
                        {
                            "operator": "sample_frames",
                            "service_ms": sampled.service_ms,
                            "start_timestamp": sampling_start.isoformat(),
                            "end_timestamp": sampling_end.isoformat(),
                            "output_bytes": sampled.output_artifacts[0].size_bytes,
                        },
                        {
                            "operator": "invoke_model",
                            "service_ms": reduced.service_ms,
                            "start_timestamp": model_start.isoformat(),
                            "end_timestamp": model_end.isoformat(),
                            "output_bytes": evidence.size_bytes,
                        },
                    ],
                    "selected_frame_count": sweep.sample_count_per_chunk,
                    "selected_clip_count": 0,
                    "selected_clip_bytes": 0,
                    "contact_sheet_bytes": sampled.output_artifacts[0].size_bytes,
                    "evidence_artifact_id": evidence.id,
                    "evidence_bytes": evidence.size_bytes,
                    "evidence_tokens": int(reduced.metadata.get("output_tokens", 0)),
                    "model_input_tokens": int(reduced.metadata.get("input_tokens", 0)),
                    "start_timestamp": branch_start.isoformat(),
                    "end_timestamp": branch_end.isoformat(),
                }

            profile_start = perf_counter()
            branches = await asyncio.gather(
                *(profile_branch(index) for index in range(len(artifacts)))
            )
            profile_wall_ms = (perf_counter() - profile_start) * 1000
            critical = max(branches, key=lambda item: float(item["preprocessing_wall_ms"]))
            raw_bytes = sum(int(item["input_bytes"]) for item in branches)
            evidence_bytes = sum(int(item["evidence_bytes"]) for item in branches)
            payload: dict[str, object] = {
                "schema_version": "calibration-v0-local-profile-v1",
                "task_id": task.task_id,
                "profile_run_id": run_id,
                "strong_node_used": False,
                "profile_wall_ms": profile_wall_ms,
                "local_preprocessing_sum_ms": sum(
                    float(item["preprocessing_wall_ms"]) for item in branches
                ),
                "local_preprocessing_critical_ms": float(
                    critical["preprocessing_wall_ms"]
                ),
                "critical_worker_id": critical["worker_id"],
                "raw_bytes": raw_bytes,
                "contact_sheet_bytes": sum(
                    int(item["contact_sheet_bytes"]) for item in branches
                ),
                "evidence_bytes": evidence_bytes,
                "evidence_tokens": sum(int(item["evidence_tokens"]) for item in branches),
                "raw_to_evidence_ratio": evidence_bytes / raw_bytes,
                "reduction_factor": raw_bytes / evidence_bytes,
                "branches": branches,
            }
            destination = output_directory / f"video_mme_{task.task_id.split(':')[-1]}.json"
            destination.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            await trace.end({"success": True, "profile": payload})
            written.append(str(destination))
    finally:
        await asyncio.gather(*(client.aclose() for client in clients.values()))

    profiles = [
        cast(dict[str, Any], json.loads(Path(path).read_text(encoding="utf-8")))
        for path in written
    ]
    lines = [
        "# calibration_v0 Jetson local-reduction profiles",
        "",
        "These are new Jetson-only measurements. The 4090 worker was neither contacted nor used.",
        "Each branch runs the generic `sample_frames` operator followed by one local VLM call; "
        "the three branches execute concurrently.",
        "",
        "| Task | Raw bytes | Evidence bytes | Reduction | Sum compute | "
        "Critical branch | Bottleneck |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for profile in profiles:
        branches = cast(list[dict[str, Any]], profile["branches"])
        tool_sum = sum(float(item["tool_service_ms"]) for item in branches)
        model_sum = sum(float(item["model_service_ms"]) for item in branches)
        bottleneck = "local VLM" if model_sum >= tool_sum else "uniform frame sampling"
        lines.append(
            f"| `{profile['task_id']}` | {profile['raw_bytes']:,} | "
            f"{profile['evidence_bytes']:,} | {profile['reduction_factor']:.0f}x | "
            f"{profile['local_preprocessing_sum_ms'] / 1000:.2f}s | "
            f"{profile['critical_worker_id']} "
            f"({profile['local_preprocessing_critical_ms'] / 1000:.2f}s) | {bottleneck} |"
        )
    lines.extend(
        [
            "",
            "`sum` is total resource time over all branches; `critical` is the measured longest "
            "branch from controller timestamps, not `sum / 3`. The reduction path is raw MP4 -> "
            "12 fixed uniform frames -> one chronological JPEG contact sheet -> compact JSON "
            "semantic evidence. No clips were selected in this implementation.",
        ]
    )
    for profile in profiles:
        saved = int(profile["raw_bytes"]) - int(profile["evidence_bytes"])
        lines.append(
            f"For {profile['task_id']}, the measured critical local cost is "
            f"{profile['local_preprocessing_critical_ms'] / 1000:.2f}s "
            f"({profile['local_preprocessing_sum_ms'] / 1000:.2f}s summed resource time) "
            f"and avoids transferring {saved:,} raw bytes."
        )
    lines.append(
        "This is a compute-for-communication trade: preprocessing is not free, and it only "
        "wins when saved transfer time exceeds the measured local critical path plus quality risk."
    )
    (output_directory / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"profiles": written, "summary": str(output_directory / "summary.md")}
