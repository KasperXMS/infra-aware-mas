"""Planner-free reference workflows for the long-video calibration_v0 experiment.

Both references are expressed solely as the existing generic ``invoke_model`` operator.
The scheduler therefore retains responsibility for physical placement: a replicated local
video model follows each single input chunk to its Orin, while the strong model has one
4090 replica and causes normal Worker-to-Worker transfers.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from pathlib import Path
from time import perf_counter
from typing import Annotated, Any, Literal, Protocol, cast
from uuid import uuid4

import yaml
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from infra_mas.core.artifact import ArtifactRef
from infra_mas.core.execution import ExecutionResult, InvocationSpec
from infra_mas.execution.executor_registry import ExecutorRegistry
from infra_mas.execution.manager import ExecutionManager
from infra_mas.execution.transfer import TransferManager, TransferProfile
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
from infra_mas.operators import (
    GENERAL_OPERATOR_REGISTRY,
    ExternalEvaluatorSpec,
    InitialArtifactSpec,
    ObservationSpec,
    RuntimeVerifierSpec,
    TaskInteractionSpec,
)
from infra_mas.resources.provider import StaticResourceConfig, StaticResourceProvider
from infra_mas.runtime.model_registry import ModelRegistry
from infra_mas.runtime.runtime import AgentRuntime
from infra_mas.tracing.recorder import TraceRecorder

NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]

CalibrationV0Workflow = Literal["centralized_raw", "local_reduction"]
CALIBRATION_V0_WORKFLOWS: tuple[CalibrationV0Workflow, ...] = (
    "centralized_raw",
    "local_reduction",
)


class InvocationRuntime(Protocol):
    async def invoke(
        self,
        invocation: InvocationSpec,
        *,
        parent_action_id: str | None = None,
    ) -> ExecutionResult: ...

    async def sample_frames(
        self,
        artifact: ArtifactRef,
        target_worker_id: str,
        *,
        duration_s: float,
        sample_count: int = 20,
        columns: int = 5,
        frame_width: int = 448,
        parent_action_id: str | None = None,
    ) -> ExecutionResult: ...


class ReferenceWorkflowResult(BaseModel):
    """Measured model-side result of one calibration reference execution."""

    model_config = ConfigDict(extra="forbid")

    workflow_id: CalibrationV0Workflow
    executions: list[ExecutionResult]
    final_artifact: ArtifactRef
    raw_artifact_bytes: int = Field(ge=0)
    reduced_artifact_bytes: int = Field(ge=0)
    local_service_ms: float = Field(ge=0)
    remote_service_ms: float = Field(ge=0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    api_cost_usd: float = Field(ge=0)


class CalibrationV0Input(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artifact_id: NonEmptyString
    path: Path
    duration_s: float = Field(gt=0.0)


class CalibrationV0Task(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: NonEmptyString
    instruction: NonEmptyString
    options: list[NonEmptyString] = Field(default_factory=list)
    evaluator_id: NonEmptyString
    inputs: Annotated[list[CalibrationV0Input], Field(min_length=3, max_length=3)]

    @model_validator(mode="after")
    def unique_artifacts(self) -> CalibrationV0Task:
        ids = [item.artifact_id for item in self.inputs]
        if len(ids) != len(set(ids)):
            raise ValueError("task input artifact IDs must be unique")
        return self


class CalibrationV0World(BaseModel):
    model_config = ConfigDict(extra="forbid")

    world_id: Literal["H1_distributed_constrained", "H2_distributed_favorable"]
    bandwidth_mbps: float = Field(gt=0.0)
    rtt_ms: float = Field(ge=0.0)


class CalibrationV0SweepConfig(BaseModel):
    """Strict small-sweep contract; models/devices/placement are shared by both worlds."""

    model_config = ConfigDict(extra="forbid")

    experiment_config: Path
    local_model_id: NonEmptyString
    strong_model_id: NonEmptyString
    input_workers: Annotated[list[NonEmptyString], Field(min_length=3, max_length=3)]
    eligible_executor_ids: Annotated[list[NonEmptyString], Field(min_length=4)]
    tasks: Annotated[list[CalibrationV0Task], Field(min_length=1, max_length=2)]
    worlds: Annotated[list[CalibrationV0World], Field(min_length=2, max_length=2)]
    workflows: list[CalibrationV0Workflow] = list(CALIBRATION_V0_WORKFLOWS)
    repeats: int = Field(default=3, ge=3)
    sample_count_per_chunk: int = Field(default=12, ge=1, le=64)
    frame_width: int = Field(default=320, ge=64, le=1920)
    run_prefix: NonEmptyString = "calibration-v0"

    @model_validator(mode="after")
    def validate_controlled_pair(self) -> CalibrationV0SweepConfig:
        if len(self.input_workers) != len(set(self.input_workers)):
            raise ValueError("the three input_workers must be distinct")
        if len(self.eligible_executor_ids) != len(set(self.eligible_executor_ids)):
            raise ValueError("eligible_executor_ids must be unique")
        if self.workflows != list(CALIBRATION_V0_WORKFLOWS):
            raise ValueError("workflows must be [centralized_raw, local_reduction]")
        by_id = {world.world_id: world for world in self.worlds}
        if set(by_id) != {
            "H1_distributed_constrained",
            "H2_distributed_favorable",
        }:
            raise ValueError("both controlled calibration_v0 worlds are required")
        h1 = by_id["H1_distributed_constrained"]
        h2 = by_id["H2_distributed_favorable"]
        if h1.bandwidth_mbps >= h2.bandwidth_mbps or h1.rtt_ms <= h2.rtt_ms:
            raise ValueError("H1 must have lower bandwidth and higher RTT than H2")
        task_ids = [task.task_id for task in self.tasks]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("task IDs must be unique")
        return self

    @classmethod
    def from_yaml(cls, path: Path) -> CalibrationV0SweepConfig:
        raw: object = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls.model_validate(raw)


def build_video_task_interaction(
    task_id: str,
    objective: str,
    chunks: list[ArtifactRef],
    evaluator_id: str,
) -> TaskInteractionSpec:
    """Build and validate the generic operator contract used by both references."""
    interaction = TaskInteractionSpec(
        task_id=task_id,
        objective=objective,
        initial_artifacts=[
            InitialArtifactSpec(
                artifact_id=artifact.id,
                kind="fixed_time_video_chunk",
                source_ref=artifact.id,
            )
            for artifact in chunks
        ],
        operators=["sample_frames", "invoke_model"],
        observations=[
            ObservationSpec(
                observation_id="uniform_contact_sheets",
                produced_by=["sample_frames"],
                description="Fixed uniform samples independent of gold evidence.",
            ),
            ObservationSpec(
                observation_id="video_evidence",
                produced_by=["invoke_model"],
                description="Time-localized semantic evidence extracted from video chunks.",
            ),
            ObservationSpec(
                observation_id="final_answer",
                produced_by=["invoke_model"],
                description="Answer produced from raw chunks or reduced evidence.",
            ),
        ],
        runtime_verifier=RuntimeVerifierSpec(level="none"),
        external_evaluator=ExternalEvaluatorSpec(evaluator_id=evaluator_id),
    )
    GENERAL_OPERATOR_REGISTRY.validate_task(interaction)
    return interaction


def _validate_chunks(chunks: list[ArtifactRef]) -> None:
    if len(chunks) != 3:
        raise ValueError("calibration_v0 requires exactly three fixed-time video chunks")
    invalid = [
        artifact.id
        for artifact in chunks
        if not artifact.artifact_type.startswith("video/")
    ]
    if invalid:
        raise ValueError(f"calibration_v0 inputs must be video artifacts: {invalid}")


def _usage(results: list[ExecutionResult]) -> tuple[int, int, float]:
    return (
        sum(int(result.metadata.get("input_tokens", 0)) for result in results),
        sum(int(result.metadata.get("output_tokens", 0)) for result in results),
        sum(float(result.metadata.get("api_cost_usd", 0.0)) for result in results),
    )


async def execute_video_reference_workflow(
    runtime: InvocationRuntime,
    *,
    workflow_id: CalibrationV0Workflow,
    task: str,
    raw_video_chunks: list[ArtifactRef],
    local_model_id: str,
    strong_model_id: str,
    reasoning_worker_id: str,
    chunk_durations_s: list[float],
    sample_count: int = 20,
    frame_width: int = 448,
    direct_video: bool = False,
) -> ReferenceWorkflowResult:
    """Execute one hidden reference without constructing or consulting a Planner."""
    if not task.strip():
        raise ValueError("task must not be empty")
    _validate_chunks(raw_video_chunks)
    if len(chunk_durations_s) != 3 or any(value <= 0 for value in chunk_durations_s):
        raise ValueError("chunk_durations_s must contain three positive durations")
    raw_bytes = sum(artifact.size_bytes for artifact in raw_video_chunks)

    if workflow_id == "centralized_raw":
        if direct_video:
            strong_inputs = raw_video_chunks
            sampling: list[ExecutionResult] = []
        else:
            sampling = await asyncio.gather(
                *(
                    runtime.sample_frames(
                        artifact,
                        reasoning_worker_id,
                        duration_s=chunk_durations_s[index],
                        sample_count=sample_count,
                        frame_width=frame_width,
                    )
                    for index, artifact in enumerate(raw_video_chunks)
                )
            )
            strong_inputs = [
                artifact
                for result in sampling
                for artifact in result.output_artifacts
            ]
        result = await runtime.invoke(
            InvocationSpec(
                model_id=strong_model_id,
                role="calibration-v0-centralized-raw",
                instructions=(
                    "Analyze the supplied fixed-time video chunks in chronological order. "
                    "Use evidence from every relevant time span, distinguish observed facts "
                    "from inference, and return only the exact JSON object requested by the "
                    "task, without explanation."
                ),
                task=task,
                input_artifacts=strong_inputs,
            )
        )
        executions = [*sampling, result]
        input_tokens, output_tokens, api_cost = _usage(executions)
        return ReferenceWorkflowResult(
            workflow_id=workflow_id,
            executions=executions,
            final_artifact=result.output_artifacts[0],
            raw_artifact_bytes=raw_bytes,
            reduced_artifact_bytes=0,
            local_service_ms=0.0,
            remote_service_ms=result.service_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            api_cost_usd=api_cost,
        )

    async def reduce_chunk(
        index: int, artifact: ArtifactRef
    ) -> tuple[ExecutionResult, ExecutionResult]:
        sampled = await runtime.sample_frames(
            artifact,
            artifact.locations[0],
            duration_s=chunk_durations_s[index],
            sample_count=sample_count,
            frame_width=frame_width,
        )
        reduced = await runtime.invoke(
            InvocationSpec(
                model_id=local_model_id,
                role=f"calibration-v0-local-reduction-{index + 1}",
                instructions=(
                    "You are an intermediate generic video evidence extractor, not the final "
                    "question-answering stage. Preserve timestamps, entities, actions, state "
                    "changes, ordering, and uncertainty. Never answer multiple-choice questions "
                    "or emit a question-ID-to-option mapping. Return compact JSON semantic "
                    "evidence with keys chunk, observations, and uncertainty."
                ),
                task=(
                    f"Inspect chronological chunk {index + 1} of 3. The following downstream "
                    f"task is relevance context only; do not answer it:\n{task}\nReturn only "
                    "time-localized observations from this chunk for a separate final reasoner."
                ),
                input_artifacts=sampled.output_artifacts,
            )
        )
        return sampled, reduced

    local_pairs = await asyncio.gather(
        *(reduce_chunk(index, artifact) for index, artifact in enumerate(raw_video_chunks))
    )
    sampling = [pair[0] for pair in local_pairs]
    reductions = [pair[1] for pair in local_pairs]
    reduced_artifacts = [result.output_artifacts[0] for result in reductions]
    synthesis = await runtime.invoke(
        InvocationSpec(
            model_id=strong_model_id,
            role="calibration-v0-evidence-reasoning",
            instructions=(
                "Reason over the three chronological evidence artifacts. Reconcile events across "
                "time spans, respect uncertainty, and return only the exact JSON object requested "
                "by the task, without explanation."
            ),
            task=task,
            input_artifacts=reduced_artifacts,
        )
    )
    executions = [*sampling, *reductions, synthesis]
    input_tokens, output_tokens, api_cost = _usage(executions)
    return ReferenceWorkflowResult(
        workflow_id=workflow_id,
        executions=executions,
        final_artifact=synthesis.output_artifacts[0],
        raw_artifact_bytes=raw_bytes,
        reduced_artifact_bytes=sum(artifact.size_bytes for artifact in reduced_artifacts),
        local_service_ms=sum(result.service_ms for result in [*sampling, *reductions]),
        remote_service_ms=synthesis.service_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        api_cost_usd=api_cost,
    )


async def _download_answer(
    artifact: ArtifactRef,
    clients: Mapping[str, WorkerClient],
    temporary_root: Path,
) -> str:
    temporary_root.mkdir(parents=True, exist_ok=True)
    destination = temporary_root / f"calibration-v0-{uuid4().hex}.txt"
    try:
        await clients[artifact.locations[0]].download_artifact(artifact.id, destination)
        return destination.read_text(encoding="utf-8")
    finally:
        if destination.is_file():
            destination.unlink()


def _task_prompt(task: CalibrationV0Task) -> str:
    if not task.options:
        return task.instruction
    rendered = "\n".join(
        f"{chr(65 + index)}. {option}" for index, option in enumerate(task.options)
    )
    return f"{task.instruction}\nOptions:\n{rendered}\nFinish with `ANSWER: <option letter>`."


def _read_trace(path: Path) -> list[dict[str, Any]]:
    return [
        cast(dict[str, Any], json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _execution_rows(
    events: list[dict[str, Any]],
    registry: ExecutorRegistry,
    initial_sizes: Mapping[str, int],
) -> list[dict[str, object]]:
    sizes = dict(initial_sizes)
    for event in events:
        if event.get("event_type") == "artifact.created":
            sizes[str(event["artifact_id"])] = int(event["size_bytes"])
    requests = {
        str(event["action_id"]): event
        for event in events
        if event.get("event_type") == "execution.request"
    }
    ends = [
        event
        for event in events
        if event.get("event_type") == "worker.execution.end" and event.get("success")
    ]
    rows: list[dict[str, object]] = []
    for event in ends:
        action_id = str(event["action_id"])
        request = requests[action_id]
        executor_id = str(event["executor"])
        worker_id = str(event["worker_id"])
        if executor_id.endswith(":sample_frames"):
            site_id = registry.worker_sites()[worker_id]
            operator_id = "sample_frames"
        else:
            site_id = registry.get(executor_id).site
            operator_id = str(request.get("semantic_operator", "invoke_model"))
        inputs = cast(list[str], request.get("input_artifacts", []))
        outputs = cast(list[str], event.get("output_artifacts", []))
        rows.append(
            {
                "action_id": action_id,
                "operator_id": operator_id,
                "executor_id": executor_id,
                "worker_id": worker_id,
                "site_id": site_id,
                "service_ms": float(event.get("service_ms", 0.0)),
                "input_bytes": sum(sizes[item] for item in inputs),
                "output_bytes": sum(sizes[item] for item in outputs),
            }
        )
    return rows


def _transfer_rows(
    events: list[dict[str, Any]], registry: ExecutorRegistry
) -> list[dict[str, object]]:
    sites = registry.worker_sites()
    transfers = [
        event
        for event in events
        if event.get("event_type") == "artifact.transfer.end"
        and event.get("success")
        and int(event.get("bytes_transferred", 0)) > 0
    ]
    return [
        {
            "transfer_id": f"{event['action_id']}:{index}",
            "artifact_id": str(event["artifact_id"]),
            "src_site": sites[str(event["source_worker_id"])],
            "dst_site": sites[str(event["target_worker_id"])],
            "bytes": int(event["bytes_transferred"]),
            "latency_ms": float(event["transfer_ms"]),
        }
        for index, event in enumerate(transfers, start=1)
    ]


def _append_jsonl(path: Path, row: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="") as output:
        output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _completed_keys(path: Path) -> set[tuple[str, str, str, int]]:
    if not path.is_file():
        return set()
    rows = [
        cast(dict[str, object], json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return {
        (
            str(row["task_id"]),
            str(row["workflow_id"]),
            str(row["world_id"]),
            int(cast(int, row["repeat"])),
        )
        for row in rows
        if row.get("status") == "completed"
    }


def _validate_deployment(
    sweep: CalibrationV0SweepConfig,
    registry: ExecutorRegistry,
) -> str:
    sites = registry.worker_sites()
    actual_input_sites = [sites.get(worker_id) for worker_id in sweep.input_workers]
    if actual_input_sites != ["A4", "A5", "A28"]:
        raise ValueError("input_workers must resolve, in order, to sites A4, A5, A28")
    eligible = registry.filtered(sweep.eligible_executor_ids)
    local_sites = sorted(item.site for item in eligible.model_candidates(sweep.local_model_id))
    if local_sites != ["A28", "A4", "A5"]:
        raise ValueError("local_model_id must have exactly one eligible replica on each Orin")
    strong = eligible.model_candidates(sweep.strong_model_id)
    if len(strong) != 1 or strong[0].site != "4090":
        raise ValueError("strong_model_id must have exactly one eligible replica at site 4090")
    return strong[0].worker_id


async def run_calibration_v0_sweep(
    sweep_path: Path,
    output_directory: Path,
) -> dict[str, object]:
    """Run the fixed 2x2xN calibration and append bench-compatible raw JSONL rows."""
    sweep_path = sweep_path.resolve()
    sweep = CalibrationV0SweepConfig.from_yaml(sweep_path)
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
        raise ValueError("calibration_v0 requires resources_config")
    resources = StaticResourceConfig.from_yaml(
        resolve_config_path(experiment.resources_config, experiment_base)
    )
    active_registry = full_registry.filtered(sweep.eligible_executor_ids)
    active_ids = {item.id for item in active_registry.list()}
    provider = StaticResourceProvider(
        active_registry,
        service_times=[item for item in resources.service_times if item.executor_id in active_ids],
        network_links=resources.network_links,
    )
    scheduler = build_scheduler(experiment.scheduler, active_registry, models, None, provider)
    reasoning_worker_id = _validate_deployment(sweep, full_registry)
    output_directory = output_directory.resolve()
    raw_path = output_directory / "raw_runs.jsonl"
    traces_root = output_directory / "traces"
    temporary_root = output_directory / ".runtime"
    completed = _completed_keys(raw_path)
    clients = create_worker_clients(full_registry, experiment.worker_timeout_seconds)
    written = 0
    try:
        await preflight_workers(full_registry, clients)
        for task in sweep.tasks:
            input_paths = [resolve_config_path(item.path, base) for item in task.inputs]
            for path in input_paths:
                if not path.is_file():
                    raise FileNotFoundError(f"calibration_v0 input not found: {path}")
            for world in sweep.worlds:
                for repeat in range(1, sweep.repeats + 1):
                    for workflow_id in sweep.workflows:
                        key = (task.task_id, workflow_id, world.world_id, repeat)
                        if key in completed:
                            continue
                        safe_task_id = "".join(
                            character
                            if character.isalnum() or character in {"-", "_", "."}
                            else "-"
                            for character in task.task_id
                        )
                        run_id = (
                            f"{sweep.run_prefix}-{safe_task_id}-{world.world_id}-"
                            f"{workflow_id}-r{repeat}-{uuid4().hex[:8]}"
                        )
                        trace = TraceRecorder(traces_root, run_id, exclusive=True)
                        await trace.start(
                            {
                                "mode": "calibration_v0_reference",
                                "planner_constructed": False,
                                "task_id": task.task_id,
                                "workflow_id": workflow_id,
                                "world_id": world.world_id,
                                "repeat": repeat,
                                "bandwidth_mbps": world.bandwidth_mbps,
                                "rtt_ms_added": world.rtt_ms,
                                "network_shaping": "worker_application_layer_wall_clock",
                            }
                        )
                        started_at = perf_counter()
                        try:
                            artifacts = await upload_placed_inputs(
                                input_paths,
                                run_id,
                                sweep.input_workers,
                                clients,
                                artifact_ids=[item.artifact_id for item in task.inputs],
                            )
                            # Controller ingress establishes the requested initial placement;
                            # the measured E2E interval starts from the reference workflow.
                            started_at = perf_counter()
                            initial_sizes = {item.id: item.size_bytes for item in artifacts}
                            build_video_task_interaction(
                                task.task_id,
                                _task_prompt(task),
                                artifacts,
                                task.evaluator_id,
                            )
                            manager = ExecutionManager(
                                clients,
                                TransferManager(
                                    clients,
                                    trace,
                                    TransferProfile(world.bandwidth_mbps, world.rtt_ms),
                                ),
                                trace,
                            )
                            runtime = AgentRuntime(
                                None,
                                scheduler,
                                manager,
                                trace,
                                request_id_factory=lambda: f"{run_id}/request-{uuid4().hex}",
                                model_registry=models,
                            )
                            result = await execute_video_reference_workflow(
                                runtime,
                                workflow_id=workflow_id,
                                task=_task_prompt(task),
                                raw_video_chunks=artifacts,
                                local_model_id=sweep.local_model_id,
                                strong_model_id=sweep.strong_model_id,
                                reasoning_worker_id=reasoning_worker_id,
                                chunk_durations_s=[item.duration_s for item in task.inputs],
                                sample_count=sweep.sample_count_per_chunk,
                                frame_width=sweep.frame_width,
                            )
                            answer = await _download_answer(
                                result.final_artifact, clients, temporary_root
                            )
                            e2e_ms = (perf_counter() - started_at) * 1000
                            await trace.end(
                                {
                                    "success": True,
                                    "answer": answer,
                                    "e2e_latency_ms": e2e_ms,
                                }
                            )
                            events = _read_trace(trace.path)
                            transfers = _transfer_rows(events, full_registry)
                            executions = _execution_rows(
                                events, full_registry, initial_sizes
                            )
                            raw_records = [
                                {
                                    "artifact_id": artifact.id,
                                    "kind": "raw_video_chunk",
                                    "site_id": full_registry.worker_sites()[worker_id],
                                    "bytes": artifact.size_bytes,
                                }
                                for artifact, worker_id in zip(
                                    artifacts, sweep.input_workers, strict=True
                                )
                            ]
                            reduced = [
                                item.output_artifacts[0]
                                for item in result.executions
                                if item.metadata.get("semantic_operator") != "sample_frames"
                                and item is not result.executions[-1]
                            ]
                            reduced_records = [
                                {
                                    "artifact_id": artifact.id,
                                    "kind": "semantic_evidence",
                                    "site_id": full_registry.worker_sites()[artifact.locations[0]],
                                    "bytes": artifact.size_bytes,
                                }
                                for artifact in reduced
                            ]
                            row: dict[str, object] = {
                                "schema_version": "calibration-v0-run-v1",
                                "run_id": run_id,
                                "task_id": task.task_id,
                                "workflow_id": workflow_id,
                                "world_id": world.world_id,
                                "repeat": repeat,
                                "status": "completed",
                                # Gold stays on the infra-bench evaluator side. Its reporter
                                # fills this field before quality-gated comparison.
                                "quality": None,
                                "artifacts": {
                                    "raw_bytes": result.raw_artifact_bytes,
                                    "reduced_bytes": result.reduced_artifact_bytes,
                                    "reduction_ratio": (
                                        result.reduced_artifact_bytes
                                        / result.raw_artifact_bytes
                                    ),
                                    "records": [*raw_records, *reduced_records],
                                },
                                "transfers": {
                                    "count": len(transfers),
                                    "bytes": sum(
                                        int(cast(int | str, item["bytes"]))
                                        for item in transfers
                                    ),
                                    "latency_ms": sum(
                                        float(cast(float | int | str, item["latency_ms"]))
                                        for item in transfers
                                    ),
                                    "records": transfers,
                                },
                                "service": {
                                    "local_preprocessing_ms": result.local_service_ms,
                                    "remote_model_ms": result.remote_service_ms,
                                    "total_ms": sum(
                                        item.service_ms for item in result.executions
                                    ),
                                },
                                "usage": {
                                    "input_tokens": result.input_tokens,
                                    "output_tokens": result.output_tokens,
                                    "api_cost_usd": result.api_cost_usd,
                                },
                                "e2e_latency_ms": e2e_ms,
                                "executions": executions,
                                "final_answer": answer,
                                "metadata": {
                                    "network_shaping": "worker_application_layer_wall_clock",
                                    "bandwidth_mbps": world.bandwidth_mbps,
                                    "rtt_ms_added": world.rtt_ms,
                                },
                            }
                        except Exception as error:
                            e2e_ms = (perf_counter() - started_at) * 1000
                            await trace.end(
                                {
                                    "success": False,
                                    "error": f"{type(error).__name__}: {error}",
                                    "e2e_latency_ms": e2e_ms,
                                }
                            )
                            row = {
                                "schema_version": "calibration-v0-run-v1",
                                "run_id": run_id,
                                "task_id": task.task_id,
                                "workflow_id": workflow_id,
                                "world_id": world.world_id,
                                "repeat": repeat,
                                "status": "failed",
                                "error": f"{type(error).__name__}: {error}",
                                "metadata": {
                                    "network_shaping": "worker_application_layer_wall_clock",
                                    "bandwidth_mbps": world.bandwidth_mbps,
                                    "rtt_ms_added": world.rtt_ms,
                                },
                            }
                        _append_jsonl(raw_path, row)
                        if row["status"] == "completed":
                            completed.add(key)
                        written += 1
    finally:
        await close_worker_clients(clients)
    return {
        "raw_runs": str(raw_path),
        "new_rows": written,
        "total_rows": len(completed),
    }
