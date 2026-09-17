"""Measured hidden-reference calibration sweep for semantic-switch admission."""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Annotated, Literal, cast
from uuid import uuid4

import yaml
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from infra_mas.core.artifact import ArtifactRef
from infra_mas.core.execution import ExecutionResult, InvocationSpec
from infra_mas.execution.executor_registry import ExecutorRegistry
from infra_mas.execution.manager import ExecutionManager
from infra_mas.execution.transfer import TransferManager
from infra_mas.execution.worker_client import WorkerClient
from infra_mas.experiment import (
    BlindExperimentConfig,
    build_scheduler,
    close_worker_clients,
    create_worker_clients,
    preflight_workers,
    resolve_config_path,
    upload_placed_inputs,
)
from infra_mas.resources.provider import StaticResourceConfig, StaticResourceProvider
from infra_mas.runtime.model_registry import ModelRegistry
from infra_mas.runtime.runtime import AgentRuntime
from infra_mas.tracing.recorder import TraceRecorder

NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
ReferenceWorkflow = Literal["centralized", "distributed_3x2", "distributed_6x1"]
REFERENCE_WORKFLOWS: tuple[ReferenceWorkflow, ...] = (
    "centralized",
    "distributed_3x2",
    "distributed_6x1",
)


class SweepInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artifact_id: NonEmptyString
    path: Path


class SweepWorld(BaseModel):
    model_config = ConfigDict(extra="forbid")

    world_id: NonEmptyString
    input_workers: list[NonEmptyString]
    eligible_executor_ids: list[NonEmptyString]
    workflows: list[ReferenceWorkflow] | None = None


class CalibrationSweepConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment_config: Path
    task: NonEmptyString
    expected_image_id: NonEmptyString
    logical_model_id: NonEmptyString
    synthesis_model_id: NonEmptyString | None = None
    inputs: Annotated[list[SweepInput], Field(min_length=1)]
    worlds: Annotated[list[SweepWorld], Field(min_length=2)]
    repeats: int = Field(default=2, ge=1)
    run_prefix: NonEmptyString = "v1-switch"
    workflows: Annotated[list[ReferenceWorkflow], Field(min_length=2)] = list(
        REFERENCE_WORKFLOWS
    )
    report_name: NonEmptyString = "v1_semantic_switch_search_v2"

    @model_validator(mode="after")
    def validate_dimensions(self) -> CalibrationSweepConfig:
        artifact_ids = [item.artifact_id for item in self.inputs]
        if len(artifact_ids) != len(set(artifact_ids)):
            raise ValueError("sweep input artifact IDs must be unique")
        world_ids = [world.world_id for world in self.worlds]
        if len(world_ids) != len(set(world_ids)):
            raise ValueError("sweep world IDs must be unique")
        for world in self.worlds:
            if len(world.input_workers) != len(self.inputs):
                raise ValueError(
                    f"world {world.world_id!r} must place every input artifact"
                )
            if not world.eligible_executor_ids:
                raise ValueError(f"world {world.world_id!r} has no eligible executor")
            if world.workflows is not None:
                if "centralized" not in world.workflows:
                    raise ValueError(
                        f"world {world.world_id!r} workflows must include centralized"
                    )
                if len(world.workflows) != len(set(world.workflows)):
                    raise ValueError(
                        f"world {world.world_id!r} workflows must be unique"
                    )
        if self.workflows[0] != "centralized" or "centralized" not in self.workflows:
            raise ValueError("calibration workflows must begin with centralized")
        if len(self.workflows) != len(set(self.workflows)):
            raise ValueError("calibration workflows must be unique")
        return self

    @classmethod
    def from_yaml(cls, path: Path) -> CalibrationSweepConfig:
        raw: object = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls.model_validate(raw)


class CalibrationRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    world_id: str
    workflow_id: ReferenceWorkflow
    repeat: int
    run_id: str
    e2e_ms: float
    service_ms_sum: float
    transfer_bytes: int
    transfer_ms_sum: float
    vlm_calls: int
    synthesis_calls: int
    success: bool
    correct: bool
    final_answer: str
    error: str | None = None


def _image_label(artifact: ArtifactRef) -> str:
    match = re.search(r"img[_-]?([0-9]+)", artifact.id, flags=re.IGNORECASE)
    return f"img_{int(match.group(1)):02d}" if match else artifact.id.rsplit("/", 1)[-1]


async def _execute_reference(
    runtime: AgentRuntime,
    artifacts: list[ArtifactRef],
    visual_model_id: str,
    synthesis_model_id: str,
    workflow: ReferenceWorkflow,
) -> tuple[list[ExecutionResult], int, int]:
    labels = [_image_label(artifact) for artifact in artifacts]
    reference_instruction = (
        "Inspect the supplied image artifacts carefully and use the exact img_XX labels from "
        "the task. For this task, `blue airplane` means blue is a dominant visible fuselage or "
        "livery color, not merely a small logo. `truck` includes an airport ground-service "
        "truck with an enclosed cab and service/load body, but excludes baggage carts, tugs, "
        "and jet bridges. Do not renumber images."
    )
    evidence_instruction = (
        "Inspect the supplied image artifacts carefully and use the exact img_XX labels from "
        "the task. Before classifying, report the broad continuous color regions on the main "
        "fuselage, explicitly ignoring airline text, tail markings, logos, and thin accent "
        "stripes. Also report each candidate ground truck, explicitly stating whether both an "
        "enclosed cab and a service, cargo, catering, or load body are visibly present. Do not "
        "require the entire cab to be unobscured: when the visible front/cab structure and the "
        "service or load body clearly form one vehicle, treat a partly occluded cab as present. "
        "Do not count a vehicle when neither a cab structure nor a qualifying body is visible. "
        "Do not infer a truck from a jet bridge, passenger stairs, baggage cart, tug, or vague "
        "distant "
        "vehicle. A blue airplane requires a substantial continuous blue fuselage or livery "
        "region. Judge airplane color and truck presence independently. Do not renumber images."
    )
    if workflow == "centralized":
        result = await runtime.invoke(
            InvocationSpec(
                model_id=visual_model_id,
                role="hidden-calibration-image-analysis",
                instructions=reference_instruction,
                task=(
                    "Images are supplied in this label order: "
                    f"{', '.join(labels)}. Find the unique image containing both a blue "
                    "airplane and a truck. Check both conditions jointly and finish with "
                    "`ANSWER: img_XX` plus a brief justification."
                ),
                input_artifacts=artifacts,
            )
        )
        return [result], 1, 0

    is_three_by_two = workflow == "distributed_3x2"
    group_size = 2 if is_three_by_two else 1
    group_count = len(artifacts) // group_size

    async def inspect_group(index: int) -> ExecutionResult:
        start = index * group_size
        group = artifacts[start : start + group_size]
        group_labels = labels[start : start + group_size]
        return await runtime.invoke(
            InvocationSpec(
                model_id=visual_model_id,
                role=(
                    f"hidden-calibration-pair-{index + 1}"
                    if is_three_by_two
                    else f"hidden-calibration-evidence-{index + 1}"
                ),
                instructions=(
                    reference_instruction if is_three_by_two else evidence_instruction
                ),
                task=(
                    (
                        f"The two inputs are {', '.join(group_labels)} in that order. Report for "
                        "each label using exactly: `img_XX | blue_airplane=yes/no | "
                        "truck=yes/no | both=yes/no`. Inspect airplane color and truck presence "
                        "independently before setting both."
                    )
                    if is_three_by_two
                    else (
                        f"The inputs are {', '.join(group_labels)} in that order. For each image, "
                        "give the requested observations. Then finish that image's evidence with "
                        "exactly: `img_XX | blue_airplane=yes/no | truck=yes/no | both=yes/no`."
                    )
                ),
                input_artifacts=group,
            )
        )

    evidence_results = await asyncio.gather(
        *(inspect_group(index) for index in range(group_count))
    )
    synthesis = await runtime.invoke(
        InvocationSpec(
            model_id=visual_model_id if is_three_by_two else synthesis_model_id,
            role="hidden-calibration-synthesis",
            instructions=(
                (
                    "Select the label whose structured evidence says both=yes. Reply with "
                    "exactly `ANSWER: img_XX` plus one brief justification sentence, or "
                    "`ANSWER: none` only if every row says both=no."
                )
                if is_three_by_two
                else (
                    "Apply the task definitions yourself to the full visual observations; do not "
                    "blindly trust the evidence-row yes/no fields. A broad continuous blue "
                    "fuselage or livery region qualifies, while blue lettering, tail markings, "
                    "logos, or thin accents alone do not. A vehicle explicitly described as "
                    "having both an enclosed cab and a service, cargo, catering, or load body "
                    "qualifies as a truck even if the evidence later says it is not a standard "
                    "road or cargo truck. Jet bridges, passenger stairs, baggage carts, and tugs "
                    "without both features do not qualify. Select the unique image meeting both "
                    "conditions. Reply with exactly `ANSWER: img_XX` plus one brief justification "
                    "sentence, or `ANSWER: none` only if none meets both."
                )
            ),
            task="Find the image containing both a blue airplane and a truck.",
            input_artifacts=[result.output_artifacts[0] for result in evidence_results],
        )
    )
    return [*evidence_results, synthesis], group_count, 1


def _workflow_counts(workflow: ReferenceWorkflow) -> tuple[int, int]:
    if workflow == "centralized":
        return 1, 0
    if workflow == "distributed_3x2":
        return 3, 1
    return 6, 1


async def _download_text(
    artifact: ArtifactRef,
    clients: Mapping[str, WorkerClient],
    temporary_root: Path,
) -> str:
    temporary_root.mkdir(parents=True, exist_ok=True)
    destination = temporary_root / f"calibration-{uuid4().hex}.txt"
    try:
        await clients[artifact.locations[0]].download_artifact(artifact.id, destination)
        return destination.read_text(encoding="utf-8")
    finally:
        if destination.is_file():
            destination.unlink()


def _is_correct(answer: str, expected_image_id: str) -> bool:
    expected = expected_image_id.lower().replace("-", "_")
    match = re.search(
        r"\banswer\s*:\s*`?(img[_-]?[0-9]+)", answer, flags=re.IGNORECASE
    )
    return bool(match and match.group(1).lower().replace("-", "_") == expected)


def _trace_totals(trace_path: Path) -> tuple[float, int, float]:
    events = [
        cast(dict[str, object], json.loads(line))
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    service = sum(
        float(cast(float | int | str, event.get("service_ms", 0)))
        for event in events
        if event.get("event_type") == "worker.execution.end" and event.get("success")
    )
    transfers = [
        event
        for event in events
        if event.get("event_type") == "artifact.transfer.end" and event.get("success")
    ]
    return (
        service,
        sum(int(cast(int | str, event.get("bytes_transferred", 0))) for event in transfers),
        sum(
            float(cast(float | int | str, event.get("transfer_ms", 0)))
            for event in transfers
        ),
    )


def _completed_row(
    run_directory: Path,
    world_id: str,
    workflow_id: ReferenceWorkflow,
    repeat: int,
    expected_image_id: str,
) -> CalibrationRow | None:
    result_path = run_directory / "result.json"
    trace_path = run_directory / "trace.jsonl"
    if not result_path.is_file() or not trace_path.is_file():
        return None
    raw: object = json.loads(result_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return None
    result = cast(dict[str, object], raw)
    if "success" not in result or "e2e_ms" not in result:
        return None
    answer = str(result.get("answer", ""))
    success = bool(result["success"])
    service_ms, transfer_bytes, transfer_ms = _trace_totals(trace_path)
    return CalibrationRow(
        world_id=world_id,
        workflow_id=workflow_id,
        repeat=repeat,
        run_id=run_directory.name,
        e2e_ms=float(cast(float | int | str, result["e2e_ms"])),
        service_ms_sum=service_ms,
        transfer_bytes=transfer_bytes,
        transfer_ms_sum=transfer_ms,
        vlm_calls=_workflow_counts(workflow_id)[0],
        synthesis_calls=_workflow_counts(workflow_id)[1],
        success=success,
        correct=success and _is_correct(answer, expected_image_id),
        final_answer=answer,
        error=str(result["error"]) if result.get("error") else None,
    )
def build_calibration_summary(rows: Sequence[CalibrationRow]) -> dict[str, object]:
    world_ids = sorted({row.world_id for row in rows})
    workflow_ids = [
        workflow
        for workflow in REFERENCE_WORKFLOWS
        if any(row.workflow_id == workflow for row in rows)
    ]
    summaries: list[dict[str, object]] = []
    by_world: dict[str, dict[str, object]] = {}
    for world_id in world_ids:
        workflow_summaries: dict[str, object] = {}
        for workflow in workflow_ids:
            selected = [
                row for row in rows if row.world_id == world_id and row.workflow_id == workflow
            ]
            if not selected:
                continue
            completed = [row for row in selected if row.success]
            values = [row.e2e_ms for row in completed]
            workflow_summaries[workflow] = {
                "run_count": len(selected),
                "success_count": len(completed),
                "all_correct": bool(completed)
                and len(completed) == len(selected)
                and all(row.correct for row in completed),
                "median_e2e_ms": median(values) if values else None,
                "median_service_ms_sum": (
                    median(row.service_ms_sum for row in completed) if completed else None
                ),
                "median_transfer_bytes": (
                    median(row.transfer_bytes for row in completed) if completed else None
                ),
                "median_transfer_ms_sum": (
                    median(row.transfer_ms_sum for row in completed) if completed else None
                ),
                "accuracy": (
                    sum(row.correct for row in selected) / len(selected) if selected else 0.0
                ),
            }
        comparisons: dict[str, object] = {}
        central = cast(dict[str, object], workflow_summaries["centralized"])
        for distributed_id in workflow_ids:
            if distributed_id == "centralized":
                continue
            if distributed_id not in workflow_summaries:
                continue
            distributed = cast(dict[str, object], workflow_summaries[distributed_id])
            central_ms = central["median_e2e_ms"]
            distributed_ms = distributed["median_e2e_ms"]
            correct = bool(central["all_correct"] and distributed["all_correct"])
            winner: str | None = None
            margin: float | None = None
            if central_ms is not None and distributed_ms is not None:
                central_value = float(cast(float, central_ms))
                distributed_value = float(cast(float, distributed_ms))
                winner = "centralized" if central_value <= distributed_value else distributed_id
                margin = abs(central_value - distributed_value) / min(
                    central_value, distributed_value
                )
            comparisons[distributed_id] = {
                "winner": winner,
                "margin": margin,
                "correct": correct,
            }
        summary: dict[str, object] = {
            "world_id": world_id,
            "workflows": workflow_summaries,
            "comparisons": comparisons,
        }
        summaries.append(summary)
        by_world[world_id] = summary

    candidates: list[tuple[float, float, str, str, str]] = []
    near_boundary_pairs: list[dict[str, object]] = []
    for distributed_id in workflow_ids:
        if distributed_id == "centralized":
            continue
        for index, world_a in enumerate(world_ids):
            for world_b in world_ids[index + 1 :]:
                first_comparisons = cast(
                    dict[str, object], by_world[world_a]["comparisons"]
                )
                second_comparisons = cast(
                    dict[str, object], by_world[world_b]["comparisons"]
                )
                if (
                    distributed_id not in first_comparisons
                    or distributed_id not in second_comparisons
                ):
                    continue
                first = cast(
                    dict[str, object],
                    first_comparisons[distributed_id],
                )
                second = cast(
                    dict[str, object],
                    second_comparisons[distributed_id],
                )
                if (
                    first["correct"]
                    and second["correct"]
                    and first["winner"] != second["winner"]
                    and first["winner"] is not None
                    and second["winner"] is not None
                ):
                    margin_a = float(cast(float, first["margin"]))
                    margin_b = float(cast(float, second["margin"]))
                    central_margin = (
                        margin_a if first["winner"] == "centralized" else margin_b
                    )
                    if margin_a >= 0.2 and margin_b >= 0.2:
                        candidates.append(
                            (
                                central_margin,
                                min(margin_a, margin_b),
                                world_a,
                                world_b,
                                distributed_id,
                            )
                        )
                    else:
                        near_boundary_pairs.append(
                            {
                                "world_a": world_a,
                                "world_b": world_b,
                                "compared_workflow": distributed_id,
                                "winner_a": first["winner"],
                                "winner_b": second["winner"],
                                "margin_a": margin_a,
                                "margin_b": margin_b,
                                "classification": "near-boundary calibration case",
                            }
                        )
    if candidates:
        _, _, world_a, world_b, distributed_id = max(candidates)
        first = cast(
            dict[str, object],
            cast(dict[str, object], by_world[world_a]["comparisons"])[distributed_id],
        )
        second = cast(
            dict[str, object],
            cast(dict[str, object], by_world[world_b]["comparisons"])[distributed_id],
        )
        admission: dict[str, object] = {
            "semantic_switch_pair_found": True,
            "world_1": world_a,
            "world_2": world_b,
            "workflow_1": first["winner"],
            "workflow_2": second["winner"],
            "margin_1": first["margin"],
            "margin_2": second["margin"],
            "compared_workflow": distributed_id,
            "world_a": world_a,
            "world_b": world_b,
            "reference_winner_a": first["winner"],
            "reference_winner_b": second["winner"],
            "margin_a": first["margin"],
            "margin_b": second["margin"],
        }
    else:
        admission = {
            "semantic_switch_pair_found": False,
            "reason": "V1 does not provide a sufficiently robust real-system crossover",
        }
    return {
        **admission,
        "near_boundary_pairs": near_boundary_pairs,
        "world_summaries": summaries,
    }


def _write_report(
    output_directory: Path,
    rows: list[CalibrationRow],
    report_name: str,
) -> dict[str, object]:
    summary = build_calibration_summary(rows)
    payload = {
        **summary,
        "runs": [row.model_dump(mode="json") for row in rows],
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    json_path = output_directory / f"{report_name}.json"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# V1 semantic-switch calibration search V2",
        "",
        "## Existing distributed_3x2 implementation audit",
        "",
        "Does distributed_3x2 perform a real model-based semantic synthesis after the three "
        "VLM outputs? **YES**.",
        "",
        "The audited pre-V2 implementation invoked `edge-vlm` for synthesis on Worker A28. The "
        "synthesis invocation was inside the measured E2E interval; its `service_ms` was included "
        "in `service_ms_sum`. Normal transfer semantics moved the A4 and A5 evidence artifacts "
        "to A28 (188 bytes in the inspected distributed trace).",
        "",
        "The V2 rerun leaves `distributed_3x2` unchanged: its synthesis still uses `edge-vlm` "
        "and normal locality-aware placement. Only the new `distributed_6x1` invokes "
        "`remote-llm` on Worker `coordinator-remote`. Its six evidence artifacts are transferred "
        "there through the normal Worker-to-Worker path. Synthesis service time remains part of "
        "`service_ms_sum` and the invocation remains inside measured E2E.",
        "",
        "## Calibration measurements",
        "",
        "| World | Workflow | Median E2E | Service | Transfer bytes | Transfer ms | Accuracy |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for value in cast(list[dict[str, object]], summary["world_summaries"]):
        workflows = cast(dict[str, dict[str, object]], value["workflows"])
        for workflow_id, workflow in workflows.items():
            lines.append(
                (
                    "| {world} | {workflow_id} | {e2e} | {service} | {bytes} | "
                    "{transfer} | {accuracy} |"
                ).format(
                    world=value["world_id"],
                    workflow_id=workflow_id,
                    e2e=workflow["median_e2e_ms"],
                    service=workflow["median_service_ms_sum"],
                    bytes=workflow["median_transfer_bytes"],
                    transfer=workflow["median_transfer_ms_sum"],
                    accuracy=workflow["accuracy"],
                )
            )
    lines.extend(
        [
            "",
            f"Semantic-switch pair found: **{summary['semantic_switch_pair_found']}**.",
            "",
            "## Pairwise winner margins",
            "",
            "| World | Comparison | Winner | Relative margin | Correct |",
            "| --- | --- | --- | ---: | --- |",
        ]
    )
    for value in cast(list[dict[str, object]], summary["world_summaries"]):
        comparisons = cast(dict[str, dict[str, object]], value["comparisons"])
        for workflow_id, comparison in comparisons.items():
            lines.append(
                f"| {value['world_id']} | centralized vs {workflow_id} | "
                f"{comparison['winner']} | {comparison['margin']} | "
                f"{comparison['correct']} |"
            )
    if not summary["semantic_switch_pair_found"]:
        lines.extend(["", f"Reason: {summary['reason']}"])
    near_boundary = cast(list[dict[str, object]], summary["near_boundary_pairs"])
    if near_boundary:
        lines.extend(["", "## Near-boundary reversal pairs", ""])
        for pair in near_boundary:
            lines.append(
                f"- [{pair['compared_workflow']}] {pair['world_a']} "
                f"({pair['winner_a']}, margin={pair['margin_a']}) vs "
                f"{pair['world_b']} ({pair['winner_b']}, margin={pair['margin_b']})"
            )
    (output_directory / f"{report_name}.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return payload


async def run_calibration_sweep(
    sweep_path: Path,
    output_directory: Path,
) -> dict[str, object]:
    """Execute both hidden references in every configured real infrastructure world."""
    sweep_path = sweep_path.resolve()
    sweep = CalibrationSweepConfig.from_yaml(sweep_path)
    base = sweep_path.parent
    experiment_path = resolve_config_path(sweep.experiment_config, base)
    experiment = BlindExperimentConfig.from_yaml(experiment_path)
    experiment_base = experiment_path.parent
    models = ModelRegistry.from_yaml(
        resolve_config_path(experiment.models_config, experiment_base)
    )
    full_registry = ExecutorRegistry.from_yaml(
        resolve_config_path(experiment.executors_config, experiment_base)
    )
    if experiment.resources_config is None:
        raise ValueError("calibration sweep requires resources_config")
    resources_path = resolve_config_path(experiment.resources_config, experiment_base)
    inputs = [resolve_config_path(item.path, base) for item in sweep.inputs]
    artifact_ids = [item.artifact_id for item in sweep.inputs]
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(f"calibration input not found: {path}")
    runs_root = resolve_config_path(experiment.runs_root, experiment_base)
    temporary_root = resolve_config_path(experiment.temporary_root, experiment_base)
    clients = create_worker_clients(full_registry, experiment.worker_timeout_seconds)
    rows: list[CalibrationRow] = []
    try:
        await preflight_workers(full_registry, clients)
        for world in sweep.worlds:
            active_registry = full_registry.filtered(world.eligible_executor_ids)
            resource_config = StaticResourceConfig.from_yaml(resources_path)
            active_ids = {executor.id for executor in active_registry.list()}
            provider = StaticResourceProvider(
                active_registry,
                service_times=[
                    item
                    for item in resource_config.service_times
                    if item.executor_id in active_ids
                ],
                network_links=resource_config.network_links,
            )
            scheduler = build_scheduler(
                experiment.scheduler, active_registry, models, None, provider
            )
            for repeat in range(1, sweep.repeats + 1):
                for workflow_id in world.workflows or sweep.workflows:
                    workflow = workflow_id
                    run_id = f"cal-{sweep.run_prefix}-{world.world_id}-{workflow}-r{repeat}"
                    completed = _completed_row(
                        runs_root / run_id,
                        world.world_id,
                        workflow_id,
                        repeat,
                        sweep.expected_image_id,
                    )
                    if completed is not None:
                        rows.append(completed)
                        continue
                    trace = TraceRecorder(runs_root, run_id, exclusive=True)
                    await trace.start(
                        {
                            "mode": "hidden_reference_calibration",
                            "world_id": world.world_id,
                            "workflow_id": workflow,
                            "planner_constructed": False,
                            "eligible_executor_ids": world.eligible_executor_ids,
                        }
                    )
                    answer = ""
                    started_at = perf_counter()
                    error: str | None = None
                    vlm_calls, synthesis_calls = _workflow_counts(workflow_id)
                    try:
                        artifacts = await upload_placed_inputs(
                            inputs,
                            run_id,
                            world.input_workers,
                            clients,
                            artifact_ids=artifact_ids,
                        )
                        manager = ExecutionManager(
                            clients, TransferManager(clients, trace), trace
                        )
                        runtime = AgentRuntime(
                            None,
                            scheduler,
                            manager,
                            trace,
                            request_id_factory=lambda: f"{run_id}/request-{uuid4().hex}",
                            model_registry=models,
                        )
                        results, vlm_calls, synthesis_calls = await _execute_reference(
                            runtime,
                            artifacts,
                            sweep.logical_model_id,
                            sweep.synthesis_model_id or sweep.logical_model_id,
                            workflow_id,
                        )
                        answer = await _download_text(
                            results[-1].output_artifacts[0], clients, temporary_root
                        )
                        success = True
                    except Exception as exc:  # preserve failed sweep rows for diagnosis
                        success = False
                        error = f"{type(exc).__name__}: {exc}"
                    e2e_ms = (perf_counter() - started_at) * 1000
                    await trace.end(
                        {
                            "success": success,
                            "answer": answer,
                            "e2e_ms": e2e_ms,
                            "error": error,
                        }
                    )
                    service_ms, transfer_bytes, transfer_ms = _trace_totals(trace.path)
                    rows.append(
                        CalibrationRow(
                            world_id=world.world_id,
                            workflow_id=workflow_id,
                            repeat=repeat,
                            run_id=run_id,
                            e2e_ms=e2e_ms,
                            service_ms_sum=service_ms,
                            transfer_bytes=transfer_bytes,
                            transfer_ms_sum=transfer_ms,
                            vlm_calls=vlm_calls,
                            synthesis_calls=synthesis_calls,
                            success=success,
                            correct=success
                            and _is_correct(answer, sweep.expected_image_id),
                            final_answer=answer,
                            error=error,
                        )
                    )
    finally:
        await close_worker_clients(clients)
    return _write_report(output_directory.resolve(), rows, sweep.report_name)
