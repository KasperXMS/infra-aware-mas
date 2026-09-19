"""Resource-blind real-machine experiment assembly."""

from __future__ import annotations

import asyncio
import mimetypes
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from time import perf_counter
from typing import Annotated, Literal
from uuid import uuid4

import yaml
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from infra_mas.core.artifact import ArtifactRef
from infra_mas.core.execution import BindLocalArtifactRequest
from infra_mas.core.resource import NetworkLink
from infra_mas.core.trace import TraceSink
from infra_mas.execution.executor_registry import ExecutorRegistry
from infra_mas.execution.manager import ExecutionManager
from infra_mas.execution.transfer import TransferManager
from infra_mas.execution.worker_client import WorkerClient
from infra_mas.planner.context import (
    ArtifactCatalog,
    InfrastructureVisibility,
    PlannerContext,
)
from infra_mas.planner.coordinator import Coordinator
from infra_mas.planner.model_factory import PlannerModelConfig, create_planner_model
from infra_mas.resources.provider import StaticResourceConfig, StaticResourceProvider
from infra_mas.runtime.agent_registry import AgentRegistry
from infra_mas.runtime.model_registry import ModelRegistry
from infra_mas.runtime.runtime import AgentRuntime
from infra_mas.scheduler.base import Scheduler
from infra_mas.scheduler.fixed import FixedScheduler
from infra_mas.scheduler.resource_aware import ResourceAwareScheduler
from infra_mas.scheduler.round_robin import RoundRobinScheduler
from infra_mas.tracing.recorder import TraceRecorder

NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class FixedSchedulerConfig(BaseModel):
    """Configure deterministic resource-blind executor assignments."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["fixed"]
    assignments: dict[NonEmptyString, NonEmptyString]


class RoundRobinSchedulerConfig(BaseModel):
    """Configure resource-blind rotation through compatible executors."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["round_robin"]


class LocalityAwareSchedulerConfig(BaseModel):
    """Configure deterministic input-locality scheduling."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["locality_aware"]


SchedulerConfig = Annotated[
    FixedSchedulerConfig | RoundRobinSchedulerConfig | LocalityAwareSchedulerConfig,
    Field(discriminator="type"),
]


class BlindExperimentConfig(BaseModel):
    """Describe one deployable resource-blind MAS experiment."""

    model_config = ConfigDict(extra="forbid")

    planner_mode: Literal["static_agents", "dynamic_models", "hybrid"] = "static_agents"
    planner_harness: Literal["minimal", "stateful", "efficient"] = "minimal"
    agents_config: Path | None = Path("agents.yaml")
    models_config: Path = Path("models.yaml")
    executors_config: Path = Path("executors.yaml")
    runs_root: Path = Path("../runs")
    temporary_root: Path = Path("../.runtime")
    resources_config: Path | None = None
    infrastructure_visibility: InfrastructureVisibility = "none"
    input_worker: NonEmptyString | None = None
    input_workers: list[NonEmptyString] | None = None
    input_source: Literal["controller_upload", "worker_local"] = "controller_upload"
    worker_timeout_seconds: Annotated[float, Field(gt=0)] = 300.0
    max_turns: Annotated[int, Field(gt=0)] = 10
    scheduler: SchedulerConfig
    planner: PlannerModelConfig

    @model_validator(mode="after")
    def validate_agent_config_for_mode(self) -> BlindExperimentConfig:
        """Require presets only in modes that expose them to the Planner."""
        if self.planner_mode in {"static_agents", "hybrid"} and self.agents_config is None:
            raise ValueError(f"planner_mode {self.planner_mode!r} requires agents_config")
        if self.input_worker is not None and self.input_workers is not None:
            raise ValueError("configure input_worker or input_workers, not both")
        if isinstance(self.scheduler, LocalityAwareSchedulerConfig):
            if self.resources_config is None:
                raise ValueError("locality_aware scheduler requires resources_config")
        if self.infrastructure_visibility != "none" and self.resources_config is None:
            raise ValueError(
                f"{self.infrastructure_visibility} visibility requires resources_config"
            )
        return self

    @classmethod
    def from_yaml(cls, path: Path) -> BlindExperimentConfig:
        """Load and validate one experiment YAML file."""
        raw: object = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls.model_validate(raw)


class PreflightResult(BaseModel):
    """Report Worker identities and executors observed over HTTP."""

    model_config = ConfigDict(extra="forbid")

    workers: dict[str, list[str]]
    operators: dict[str, list[str]] = Field(default_factory=dict)


def resolve_config_path(path: Path, config_directory: Path) -> Path:
    """Resolve a path relative to its owning YAML file."""
    return path if path.is_absolute() else (config_directory / path).resolve()


def create_worker_clients(
    registry: ExecutorRegistry,
    timeout_seconds: float,
) -> dict[str, WorkerClient]:
    """Create one reusable HTTP client per configured Worker."""
    return {
        worker_id: WorkerClient(endpoint, timeout=timeout_seconds)
        for worker_id, endpoint in registry.worker_endpoints().items()
    }


async def preflight_workers(
    registry: ExecutorRegistry,
    clients: Mapping[str, WorkerClient],
) -> PreflightResult:
    """Verify that configured endpoints expose the expected Worker identities."""
    expected: dict[str, set[str]] = {worker_id: set() for worker_id in clients}
    for executor in registry.list():
        expected[executor.worker_id].add(executor.id)

    async def check(
        worker_id: str, client: WorkerClient
    ) -> tuple[str, list[str], list[str]]:
        await client.health()
        await client.ready()
        status = await client.status()
        if status.worker_id != worker_id:
            raise ValueError(
                f"endpoint for {worker_id!r} identifies itself as {status.worker_id!r}"
            )
        actual = set(status.executors)
        if actual != expected[worker_id]:
            raise ValueError(
                f"worker {worker_id!r} exposes executors {sorted(actual)}, "
                f"expected {sorted(expected[worker_id])}"
            )
        return worker_id, status.executors, status.operators

    checked = await asyncio.gather(
        *(check(worker_id, client) for worker_id, client in clients.items())
    )
    return PreflightResult(
        workers={worker_id: executors for worker_id, executors, _ in checked},
        operators={worker_id: operators for worker_id, _, operators in checked},
    )


def build_scheduler(
    config: SchedulerConfig,
    registry: ExecutorRegistry,
    models: ModelRegistry,
    agents: AgentRegistry | None = None,
    resource_provider: StaticResourceProvider | None = None,
) -> Scheduler:
    """Build and validate logical-model-to-replica scheduling."""
    model_ids = {model.model_id for model in models.list()}
    executor_model_ids = {executor.model_id for executor in registry.list()}
    unknown = sorted(executor_model_ids - model_ids)
    if unknown:
        raise ValueError(f"executors reference unknown logical models: {unknown}")
    unavailable = sorted(
        model_id for model_id in model_ids if not registry.model_candidates(model_id)
    )
    if unavailable:
        raise ValueError(f"no executor replicas configured for logical models: {unavailable}")

    if agents is not None:
        unknown_presets = sorted(
            {
                agent.model_id
                for agent in agents.list()
                if agent.model_id is not None and agent.model_id not in model_ids
            }
        )
        if unknown_presets:
            raise ValueError(f"agent presets reference unknown models: {unknown_presets}")

    if isinstance(config, FixedSchedulerConfig):
        assigned_models = {
            registry.get(executor_id).model_id for executor_id in config.assignments.values()
        }
        missing = sorted(model_ids - assigned_models)
        if missing:
            raise ValueError(f"fixed scheduler has no assignments for models: {missing}")
        return FixedScheduler(registry, config.assignments)
    if isinstance(config, LocalityAwareSchedulerConfig):
        if resource_provider is None:
            raise ValueError("locality_aware scheduler requires a ResourceProvider")
        return ResourceAwareScheduler(registry, resource_provider)
    return RoundRobinScheduler(registry)


def build_blind_scheduler(
    config: SchedulerConfig,
    registry: ExecutorRegistry,
    models: ModelRegistry,
    agents: AgentRegistry | None = None,
) -> Scheduler:
    """Backward-compatible builder for historical blind configurations."""
    return build_scheduler(config, registry, models, agents)


async def upload_inputs(
    paths: Sequence[Path],
    run_id: str,
    worker_id: str,
    client: WorkerClient,
) -> list[ArtifactRef]:
    """Upload initial user artifacts to one explicitly configured ingress Worker."""
    uploaded: list[ArtifactRef] = []
    for index, raw_path in enumerate(paths, start=1):
        path = raw_path.resolve()
        if not await asyncio.to_thread(path.is_file):
            raise FileNotFoundError(f"input artifact not found: {path}")
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "-", path.name).strip(".-") or "artifact"
        artifact_id = f"{run_id}/input-{index:03d}-{safe_name}"
        artifact_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        reference = ArtifactRef(
            id=artifact_id,
            artifact_type=artifact_type,
            size_bytes=path.stat().st_size,
            locations=["controller"],
        )
        result = await client.upload_artifact(reference, path)
        if result.locations != [worker_id]:
            raise ValueError(
                f"ingress Worker returned unexpected artifact location: {result.locations}"
            )
        uploaded.append(result)
    return uploaded


async def upload_placed_inputs(
    paths: Sequence[Path],
    run_id: str,
    worker_ids: Sequence[str],
    clients: Mapping[str, WorkerClient],
    *,
    artifact_ids: Sequence[str] | None = None,
) -> list[ArtifactRef]:
    """Upload each input to its independently selected initial Worker."""
    if len(paths) != len(worker_ids):
        raise ValueError("input placement count must equal input artifact count")
    if artifact_ids is not None and len(paths) != len(artifact_ids):
        raise ValueError("artifact ID count must equal input artifact count")
    if artifact_ids is not None and len(artifact_ids) != len(set(artifact_ids)):
        raise ValueError("input artifact IDs must be unique")
    unknown = sorted(set(worker_ids) - clients.keys())
    if unknown:
        raise ValueError(f"unknown input Workers: {unknown}")

    uploaded: list[ArtifactRef] = []
    for index, (raw_path, worker_id) in enumerate(zip(paths, worker_ids, strict=True), start=1):
        path = raw_path.resolve()
        if not await asyncio.to_thread(path.is_file):
            raise FileNotFoundError(f"input artifact not found: {path}")
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "-", path.name).strip(".-") or "artifact"
        binding_id = artifact_ids[index - 1] if artifact_ids is not None else safe_name
        safe_binding_id = re.sub(r"[^A-Za-z0-9._-]+", "-", binding_id).strip(".-")
        if not safe_binding_id:
            raise ValueError(f"invalid artifact ID: {binding_id!r}")
        if artifact_ids is not None and path.suffix and not safe_binding_id.lower().endswith(
            path.suffix.lower()
        ):
            safe_binding_id += path.suffix
        artifact_id = f"{run_id}/input-{index:03d}-{safe_binding_id}"
        artifact_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        reference = ArtifactRef(
            id=artifact_id,
            artifact_type=artifact_type,
            size_bytes=path.stat().st_size,
            locations=["controller"],
        )
        result = await clients[worker_id].upload_artifact(reference, path)
        if result.locations != [worker_id]:
            raise ValueError(
                f"input Worker returned unexpected artifact location: {result.locations}"
            )
        uploaded.append(result)
    return uploaded


async def bind_placed_inputs(
    paths: Sequence[Path],
    run_id: str,
    worker_ids: Sequence[str],
    clients: Mapping[str, WorkerClient],
    *,
    artifact_ids: Sequence[str] | None = None,
    expected_size_bytes: Sequence[int | None] | None = None,
    trace: TraceSink | None = None,
) -> list[ArtifactRef]:
    """Bind files already on their Workers without routing bytes through the controller."""
    if len(paths) != len(worker_ids):
        raise ValueError("input placement count must equal input artifact count")
    if artifact_ids is not None and len(paths) != len(artifact_ids):
        raise ValueError("artifact ID count must equal input artifact count")
    if expected_size_bytes is not None and len(paths) != len(expected_size_bytes):
        raise ValueError("expected size count must equal input artifact count")
    unknown = sorted(set(worker_ids) - clients.keys())
    if unknown:
        raise ValueError(f"unknown input Workers: {unknown}")

    bound: list[ArtifactRef] = []
    for index, (path, worker_id) in enumerate(zip(paths, worker_ids, strict=True), start=1):
        worker_source_path = path.as_posix()
        binding = artifact_ids[index - 1] if artifact_ids is not None else path.name
        safe_binding = re.sub(r"[^A-Za-z0-9._-]+", "-", binding).strip(".-")
        if not safe_binding:
            raise ValueError(f"invalid artifact ID: {binding!r}")
        if path.suffix and not safe_binding.lower().endswith(path.suffix.lower()):
            safe_binding += path.suffix
        artifact_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        action_id = f"{run_id}/bind-{index:03d}"
        if trace is not None:
            await trace.record(
                "artifact.bind.start",
                action_id=action_id,
                artifact_id=f"{run_id}/input-{index:03d}-{safe_binding}",
                worker_id=worker_id,
                source_path=worker_source_path,
            )
        try:
            artifact = await clients[worker_id].bind_local_artifact(
                BindLocalArtifactRequest(
                    artifact_id=f"{run_id}/input-{index:03d}-{safe_binding}",
                    source_path=worker_source_path,
                    artifact_type=artifact_type,
                    expected_size_bytes=(
                        expected_size_bytes[index - 1]
                        if expected_size_bytes is not None
                        else None
                    ),
                )
            )
        except Exception as error:
            if trace is not None:
                await trace.record(
                    "artifact.bind.end",
                    action_id=action_id,
                    artifact_id=f"{run_id}/input-{index:03d}-{safe_binding}",
                    worker_id=worker_id,
                    success=False,
                    error_type=type(error).__name__,
                    error=str(error),
                )
            raise
        bound.append(artifact)
        if trace is not None:
            await trace.record(
                "artifact.bind.end",
                action_id=action_id,
                artifact_id=artifact.id,
                worker_id=worker_id,
                size_bytes=artifact.size_bytes,
                success=True,
            )
    return bound


async def close_worker_clients(clients: Mapping[str, WorkerClient]) -> None:
    """Close all controller-side HTTP connection pools."""
    await asyncio.gather(*(client.aclose() for client in clients.values()))


async def check_blind_experiment(config_path: Path) -> PreflightResult:
    """Validate configuration and check every Worker without invoking any model."""
    config_path = config_path.resolve()
    config = BlindExperimentConfig.from_yaml(config_path)
    directory = config_path.parent
    agents = (
        AgentRegistry.from_yaml(resolve_config_path(config.agents_config, directory))
        if config.agents_config is not None
        else None
    )
    models = ModelRegistry.from_yaml(resolve_config_path(config.models_config, directory))
    registry = ExecutorRegistry.from_yaml(resolve_config_path(config.executors_config, directory))
    resource_provider = (
        StaticResourceProvider.from_yaml(
            registry,
            resolve_config_path(config.resources_config, directory),
        )
        if config.resources_config is not None
        else None
    )
    build_scheduler(config.scheduler, registry, models, agents, resource_provider)
    configured_workers = (
        config.input_workers
        if config.input_workers is not None
        else ([config.input_worker] if config.input_worker is not None else [])
    )
    unknown_workers = sorted(set(configured_workers) - registry.worker_endpoints().keys())
    if unknown_workers:
        raise ValueError(f"unknown input Workers: {unknown_workers}")
    clients = create_worker_clients(registry, config.worker_timeout_seconds)
    try:
        return await preflight_workers(registry, clients)
    finally:
        await close_worker_clients(clients)


async def run_blind_experiment(
    config_path: Path,
    task: str,
    inputs: Sequence[Path],
    *,
    run_id: str | None = None,
    input_workers: Sequence[str] | None = None,
    artifact_ids: Sequence[str] | None = None,
    input_expected_sizes: Sequence[int | None] | None = None,
    infrastructure_visibility: InfrastructureVisibility | None = None,
    network_links: Sequence[NetworkLink] | None = None,
    eligible_executor_ids: Sequence[str] | None = None,
    run_metadata: Mapping[str, object] | None = None,
) -> tuple[str, Path]:
    """Run one complete resource-blind MAS experiment against physical Workers."""
    if not task.strip():
        raise ValueError("task must not be empty")

    config_path = config_path.resolve()
    config = BlindExperimentConfig.from_yaml(config_path)
    directory = config_path.parent
    agents_path = (
        resolve_config_path(config.agents_config, directory)
        if config.agents_config is not None
        else None
    )
    models_path = resolve_config_path(config.models_config, directory)
    executors_path = resolve_config_path(config.executors_config, directory)
    resources_path = (
        resolve_config_path(config.resources_config, directory)
        if config.resources_config is not None
        else None
    )
    runs_root = resolve_config_path(config.runs_root, directory)
    temporary_root = resolve_config_path(config.temporary_root, directory)
    agents = AgentRegistry.from_yaml(agents_path) if agents_path is not None else None
    models = ModelRegistry.from_yaml(models_path)
    full_registry = ExecutorRegistry.from_yaml(executors_path)
    registry = (
        full_registry.filtered(eligible_executor_ids)
        if eligible_executor_ids is not None
        else full_registry
    )
    effective_visibility = infrastructure_visibility or config.infrastructure_visibility
    resource_provider = None
    if resources_path is not None:
        resource_config = StaticResourceConfig.from_yaml(resources_path)
        active_executor_ids = {executor.id for executor in registry.list()}
        resource_provider = StaticResourceProvider(
            registry,
            service_times=[
                item
                for item in resource_config.service_times
                if item.executor_id in active_executor_ids
            ],
            network_links=(
                network_links
                if network_links is not None
                else resource_config.network_links
            ),
        )
    if effective_visibility != "none" and resource_provider is None:
        raise ValueError(f"{effective_visibility} visibility requires resources_config")
    scheduler = build_scheduler(config.scheduler, registry, models, agents, resource_provider)
    placements = list(input_workers) if input_workers is not None else config.input_workers
    if placements is None and config.input_worker is not None:
        placements = [config.input_worker] * len(inputs)
    if placements is None:
        if inputs:
            raise ValueError("input placement is required for every input artifact")
        placements = []
    if len(placements) != len(inputs):
        raise ValueError("input placement count must equal input artifact count")
    unknown_workers = sorted(set(placements) - full_registry.worker_endpoints().keys())
    if unknown_workers:
        raise ValueError(f"unknown input Workers: {unknown_workers}")

    effective_run_id = run_id or f"blind-{uuid4().hex[:12]}"
    opaque_namespace = f"opaque-{uuid4().hex}"
    trace = TraceRecorder(runs_root, effective_run_id, exclusive=True)
    clients = create_worker_clients(full_registry, config.worker_timeout_seconds)
    planner_client = None
    context: PlannerContext | None = None
    effective_config: dict[str, object] = {
        "run_id": effective_run_id,
        "planner_opaque_namespace": opaque_namespace,
        "mode": effective_visibility,
        "planner_mode": config.planner_mode,
        "planner_harness": config.planner_harness,
        "resource_aware": effective_visibility != "none",
        "infrastructure_visibility": effective_visibility,
        "task": task,
        "experiment_config": str(config_path),
        "agents_config": str(agents_path) if agents_path is not None else None,
        "models_config": str(models_path),
        "executors_config": str(executors_path),
        "resources_config": str(resources_path) if resources_path is not None else None,
        "runs_root": str(runs_root),
        "temporary_root": str(temporary_root),
        "input_worker": config.input_worker,
        "input_workers": placements,
        "input_source": config.input_source,
        "worker_timeout_seconds": config.worker_timeout_seconds,
        "max_turns": config.max_turns,
        "input_paths": [
            str(path.resolve()) if config.input_source == "controller_upload" else str(path)
            for path in inputs
        ],
        "input_artifact_ids": list(artifact_ids) if artifact_ids is not None else None,
        "input_expected_sizes": (
            list(input_expected_sizes) if input_expected_sizes is not None else None
        ),
        "run_metadata": dict(run_metadata or {}),
        "eligible_executor_ids": (
            sorted(eligible_executor_ids) if eligible_executor_ids is not None else None
        ),
        "planning_ledger_initial_state": None,
        "agents": (
            [agent.model_dump(mode="json") for agent in agents.list()]
            if agents is not None
            else []
        ),
        "models": [model.model_dump(mode="json") for model in models.list()],
        "executors": [executor.model_dump(mode="json") for executor in registry.list()],
        "worker_endpoints": registry.worker_endpoints(),
        "scheduler": config.scheduler.model_dump(mode="json"),
        "planner": config.planner.model_dump(mode="json"),
    }
    await trace.start(effective_config)
    try:
        await preflight_workers(full_registry, clients)
        if config.input_source == "worker_local":
            initial_artifacts = await bind_placed_inputs(
                inputs,
                opaque_namespace,
                placements,
                clients,
                artifact_ids=artifact_ids,
                expected_size_bytes=input_expected_sizes,
                trace=trace,
            )
        else:
            initial_artifacts = await upload_placed_inputs(
                inputs,
                opaque_namespace,
                placements,
                clients,
                artifact_ids=artifact_ids,
            )
        transfer = TransferManager(clients, trace)
        manager = ExecutionManager(clients, transfer, trace)
        runtime = AgentRuntime(
            agents,
            scheduler,
            manager,
            trace,
            request_id_factory=lambda: f"{opaque_namespace}/request-{uuid4().hex}",
            model_registry=models,
        )
        catalog = ArtifactCatalog(clients, temporary_root / "inspection", trace)
        catalog.register_many(initial_artifacts)
        planner_model, planner_client = create_planner_model(config.planner)
        context = PlannerContext(
            runtime,
            agents,
            catalog,
            trace,
            model_registry=models,
            planner_mode=config.planner_mode,
            planner_harness=config.planner_harness,
            resource_provider=resource_provider,
            infrastructure_visibility=effective_visibility,
            action_namespace=opaque_namespace,
        )
        assert context.planning_ledger is not None
        initial_planning_state = await context.planning_ledger.snapshot()
        effective_config["planning_ledger_initial_state"] = (
            initial_planning_state.model_dump(mode="json")
        )
        await trace.save_config(effective_config)
        workflow_started_at = perf_counter()
        answer = await Coordinator(
            context,
            model=planner_model,
            max_turns=config.max_turns,
        ).run(task)
        e2e_ms = (perf_counter() - workflow_started_at) * 1000
        planning_state = await context.planning_ledger.snapshot()
        run_result: dict[str, object] = {
            "success": True,
            "answer": answer,
            "e2e_ms": e2e_ms,
        }
        if config.planner_harness != "minimal":
            run_result["planner_harness"] = config.planner_harness
            run_result["planning_ledger"] = planning_state.model_dump(mode="json")
        await trace.end(run_result)
        return answer, trace.result_path.parent
    except Exception as error:
        planning_state = None
        if context is not None and context.planning_ledger is not None:
            planning_state = (await context.planning_ledger.snapshot()).model_dump(
                mode="json"
            )
        run_result = {
            "success": False,
            "error_type": type(error).__name__,
            "error": str(error),
        }
        if config.planner_harness != "minimal":
            run_result["planner_harness"] = config.planner_harness
            run_result["planning_ledger"] = planning_state
        await trace.end(run_result)
        raise
    finally:
        if planner_client is not None:
            await planner_client.close()
        await close_worker_clients(clients)
