"""Paired real-system infrastructure-awareness pilot and hidden calibration."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from pathlib import Path
from time import perf_counter
from typing import Annotated, Literal
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
    LocalityAwareSchedulerConfig,
    build_scheduler,
    close_worker_clients,
    create_worker_clients,
    preflight_workers,
    resolve_config_path,
    run_blind_experiment,
    upload_placed_inputs,
)
from infra_mas.resources.provider import StaticResourceProvider
from infra_mas.runtime.model_registry import ModelRegistry
from infra_mas.runtime.runtime import AgentRuntime
from infra_mas.tracing.recorder import TraceRecorder

NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class PilotWorld(BaseModel):
    """Name one initial placement for the same six input artifacts."""

    model_config = ConfigDict(extra="forbid")

    input_workers: Annotated[list[NonEmptyString], Field(min_length=6, max_length=6)]


class PilotConfig(BaseModel):
    """Configure paired Planner runs and the isolated calibration experiment."""

    model_config = ConfigDict(extra="forbid")

    blind_config: Path
    static_config: Path | None = None
    snapshot_config: Path
    task: NonEmptyString
    inputs: Annotated[list[Path], Field(min_length=6, max_length=6)]
    logical_model_id: NonEmptyString
    worlds: dict[Literal["colocated", "distributed"], PilotWorld]
    output_root: Path = Path("../../runs/pilot")

    @model_validator(mode="after")
    def require_paired_worlds(self) -> PilotConfig:
        if set(self.worlds) != {"colocated", "distributed"}:
            raise ValueError("worlds must contain exactly colocated and distributed")
        return self

    @classmethod
    def from_yaml(cls, path: Path) -> PilotConfig:
        raw: object = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls.model_validate(raw)


class RunMetrics(BaseModel):
    """Compact measurements derived from one execution trace."""

    model_config = ConfigDict(extra="forbid")

    workflow_structure: list[dict[str, object]]
    transfer_bytes: int
    transfer_ms: float
    service_ms: float
    e2e_ms: float
    planner_tokens: dict[str, int]
    final_answer: str


class CalibrationOutcome(BaseModel):
    """Report whether measured reference-workflow preference changes by world."""

    model_config = ConfigDict(extra="forbid")

    runs: dict[str, RunMetrics]
    preferred_by_world: dict[str, str]
    preference_reversed: bool


def _resolve_pilot_paths(
    config_path: Path,
) -> tuple[PilotConfig, Path, Path, list[Path], Path]:
    config_path = config_path.resolve()
    config = PilotConfig.from_yaml(config_path)
    directory = config_path.parent
    blind_path = resolve_config_path(config.blind_config, directory)
    snapshot_path = resolve_config_path(config.snapshot_config, directory)
    inputs = [resolve_config_path(path, directory) for path in config.inputs]
    output_root = resolve_config_path(config.output_root, directory)
    return config, blind_path, snapshot_path, inputs, output_root


def validate_paired_planner_configs(
    blind: BlindExperimentConfig,
    snapshot: BlindExperimentConfig,
) -> None:
    """Ensure visibility is the sole Planner-run configuration difference."""
    if blind.infrastructure_visibility != "none":
        raise ValueError("blind pilot config must use infrastructure_visibility: none")
    if snapshot.infrastructure_visibility != "snapshot":
        raise ValueError("snapshot pilot config must use infrastructure_visibility: snapshot")
    if blind.planner_mode != "dynamic_models" or snapshot.planner_mode != "dynamic_models":
        raise ValueError("pilot requires planner_mode: dynamic_models")
    if blind.planner_harness != "efficient" or snapshot.planner_harness != "efficient":
        raise ValueError("pilot requires planner_harness: efficient")
    if not isinstance(blind.scheduler, LocalityAwareSchedulerConfig) or not isinstance(
        snapshot.scheduler, LocalityAwareSchedulerConfig
    ):
        raise ValueError("both pilot modes must use the locality_aware scheduler")
    ignored = {"infrastructure_visibility"}
    blind_data = blind.model_dump(exclude=ignored)
    snapshot_data = snapshot.model_dump(exclude=ignored)
    if blind_data != snapshot_data:
        raise ValueError("paired Planner configs may differ only in infrastructure_visibility")


def collect_run_metrics(run_directory: Path) -> RunMetrics:
    """Derive workflow, transfer, service, token, E2E, and answer fields."""
    events = [
        json.loads(line)
        for line in (run_directory / "trace.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected = {
        event["request_id"]: event
        for event in events
        if event["event_type"] == "executor.selected"
    }
    workflow: list[dict[str, object]] = []
    for event in events:
        if event["event_type"] != "execution.request":
            continue
        binding = selected.get(event["request_id"], {})
        workflow.append(
            {
                "request_id": event["request_id"],
                "role": event["agent"],
                "model_id": event["model_id"],
                "input_artifacts": event["input_artifacts"],
                "executor_id": binding.get("executor"),
                "worker_id": binding.get("worker_id"),
            }
        )
    transfers = [
        event
        for event in events
        if event["event_type"] == "artifact.transfer.end" and event.get("success")
    ]
    executions = [
        event
        for event in events
        if event["event_type"] == "worker.execution.end" and event.get("success")
    ]
    usage = {"requests": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for event in events:
        if event["event_type"] != "planner.llm.end" or not event.get("token_usage"):
            continue
        for key in usage:
            usage[key] += int(event["token_usage"].get(key, 0))
    result = json.loads((run_directory / "result.json").read_text(encoding="utf-8"))
    return RunMetrics(
        workflow_structure=workflow,
        transfer_bytes=sum(int(event.get("bytes_transferred", 0)) for event in transfers),
        transfer_ms=sum(float(event.get("transfer_ms", 0)) for event in transfers),
        service_ms=sum(float(event.get("service_ms", 0)) for event in executions),
        e2e_ms=float(result["e2e_ms"]),
        planner_tokens=usage,
        final_answer=str(result.get("answer", "")),
    )


async def run_paired_planners(config_path: Path) -> dict[str, RunMetrics]:
    """Run blind and snapshot-visible Planners in both paired worlds."""
    config, blind_path, snapshot_path, inputs, output_root = _resolve_pilot_paths(config_path)
    blind = BlindExperimentConfig.from_yaml(blind_path)
    snapshot = BlindExperimentConfig.from_yaml(snapshot_path)
    validate_paired_planner_configs(blind, snapshot)
    output_root.mkdir(parents=True, exist_ok=True)

    results: dict[str, RunMetrics] = {}
    for world_name, world in config.worlds.items():
        for visibility, experiment_path in (("none", blind_path), ("snapshot", snapshot_path)):
            run_id = f"planner-{world_name}-{visibility}-{uuid4().hex[:8]}"
            _, run_directory = await run_blind_experiment(
                experiment_path,
                config.task,
                inputs,
                run_id=run_id,
                input_workers=world.input_workers,
            )
            results[f"{world_name}/{visibility}"] = collect_run_metrics(run_directory)
    summary_path = output_root / "planner_summary.json"
    summary_path.write_text(
        json.dumps(
            {key: value.model_dump(mode="json") for key, value in results.items()},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return results


async def _run_reference_workflow(
    runtime: AgentRuntime,
    artifacts: list[ArtifactRef],
    model_id: str,
    kind: Literal["single", "paired_synthesis"],
) -> tuple[list[ExecutionResult], str]:
    instructions = "Analyze only the supplied artifacts and answer accurately and concisely."
    task = (
        "Identify which supplied image contains both a blue airplane and a truck; include the "
        "input filename in the answer."
    )
    if kind == "single":
        result = await runtime.invoke(
            InvocationSpec(
                model_id=model_id,
                role="reference-six-image-vlm",
                instructions=instructions,
                task=task,
                input_artifacts=artifacts,
            )
        )
        return [result], result.output_artifacts[0].id

    pair_results = await asyncio.gather(
        *(
            runtime.invoke(
                InvocationSpec(
                    model_id=model_id,
                    role=f"reference-pair-{index + 1}",
                    instructions=instructions,
                    task=task,
                    input_artifacts=artifacts[index * 2 : index * 2 + 2],
                )
            )
            for index in range(3)
        )
    )
    synthesis_inputs = [result.output_artifacts[0] for result in pair_results]
    synthesis = await runtime.invoke(
        InvocationSpec(
            model_id=model_id,
            role="reference-synthesis",
            instructions="Synthesize the supplied candidate analyses into one final answer.",
            task="Select the candidate that satisfies both required visual attributes.",
            input_artifacts=synthesis_inputs,
        )
    )
    return [*pair_results, synthesis], synthesis.output_artifacts[0].id


async def run_calibration(config_path: Path) -> CalibrationOutcome:
    """Execute hidden reference workflows without creating any Planner context."""
    config, blind_path, _, inputs, output_root = _resolve_pilot_paths(config_path)
    experiment = BlindExperimentConfig.from_yaml(blind_path)
    directory = blind_path.parent
    models = ModelRegistry.from_yaml(resolve_config_path(experiment.models_config, directory))
    registry = ExecutorRegistry.from_yaml(
        resolve_config_path(experiment.executors_config, directory)
    )
    if experiment.resources_config is None:
        raise ValueError("calibration requires resources_config")
    provider = StaticResourceProvider.from_yaml(
        registry,
        resolve_config_path(experiment.resources_config, directory),
    )
    scheduler = build_scheduler(experiment.scheduler, registry, models, None, provider)
    clients = create_worker_clients(registry, experiment.worker_timeout_seconds)
    runs_root = resolve_config_path(experiment.runs_root, directory)
    output_root.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, RunMetrics] = {}
    try:
        await preflight_workers(registry, clients)
        for world_name, world in config.worlds.items():
            for kind in ("single", "paired_synthesis"):
                run_id = f"calibration-{world_name}-{kind}-{uuid4().hex[:8]}"
                trace = TraceRecorder(runs_root, run_id, exclusive=True)
                await trace.start(
                    {
                        "mode": "calibration",
                        "world": world_name,
                        "reference_workflow": kind,
                        "planner_exposed": False,
                    }
                )
                artifacts = await upload_placed_inputs(
                    inputs, run_id, world.input_workers, clients
                )
                manager = ExecutionManager(clients, TransferManager(clients, trace), trace)
                runtime = AgentRuntime(
                    None,
                    scheduler,
                    manager,
                    trace,
                    request_id_factory=lambda: f"{run_id}/request-{uuid4().hex}",
                    model_registry=models,
                )
                started_at = perf_counter()
                execution_results, final_artifact_id = await _run_reference_workflow(
                    runtime, artifacts, config.logical_model_id, kind
                )
                e2e_ms = (perf_counter() - started_at) * 1000
                final_answer = await _download_text(
                    execution_results[-1].output_artifacts[0], clients
                )
                await trace.end(
                    {
                        "success": True,
                        "answer": final_answer,
                        "final_artifact_id": final_artifact_id,
                        "e2e_ms": e2e_ms,
                    }
                )
                run_directory = trace.result_path.parent
                metrics[f"{world_name}/{kind}"] = collect_run_metrics(run_directory)
    finally:
        await close_worker_clients(clients)

    preferred: dict[str, str] = {}
    for world_name in config.worlds:
        single = metrics[f"{world_name}/single"].e2e_ms
        paired = metrics[f"{world_name}/paired_synthesis"].e2e_ms
        preferred[world_name] = "single" if single <= paired else "paired_synthesis"
    outcome = CalibrationOutcome(
        runs=metrics,
        preferred_by_world=preferred,
        preference_reversed=preferred["colocated"] != preferred["distributed"],
    )
    (output_root / "calibration_summary.json").write_text(
        json.dumps(outcome.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return outcome


async def _download_text(
    artifact: ArtifactRef,
    clients: Mapping[str, WorkerClient],
) -> str:
    source_id = artifact.locations[0]
    client = clients[source_id]
    temporary = Path.cwd() / ".runtime" / f"pilot-output-{uuid4().hex}.txt"
    temporary.parent.mkdir(parents=True, exist_ok=True)
    try:
        await client.download_artifact(artifact.id, temporary)
        return temporary.read_text(encoding="utf-8")
    finally:
        if temporary.is_file():
            temporary.unlink()
