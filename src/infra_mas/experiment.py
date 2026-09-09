"""Resource-blind real-machine experiment assembly."""

from __future__ import annotations

import asyncio
import mimetypes
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Annotated, Literal
from uuid import uuid4

import yaml
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from infra_mas.core.artifact import ArtifactRef
from infra_mas.execution.executor_registry import ExecutorRegistry
from infra_mas.execution.manager import ExecutionManager
from infra_mas.execution.transfer import TransferManager
from infra_mas.execution.worker_client import WorkerClient
from infra_mas.planner.context import ArtifactCatalog, PlannerContext
from infra_mas.planner.coordinator import Coordinator
from infra_mas.planner.model_factory import PlannerModelConfig, create_planner_model
from infra_mas.runtime.agent_registry import AgentRegistry
from infra_mas.runtime.model_registry import ModelRegistry
from infra_mas.runtime.runtime import AgentRuntime
from infra_mas.scheduler.base import Scheduler
from infra_mas.scheduler.fixed import FixedScheduler
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


SchedulerConfig = Annotated[
    FixedSchedulerConfig | RoundRobinSchedulerConfig,
    Field(discriminator="type"),
]


class BlindExperimentConfig(BaseModel):
    """Describe one deployable resource-blind MAS experiment."""

    model_config = ConfigDict(extra="forbid")

    planner_mode: Literal["static_agents", "dynamic_models", "hybrid"] = "static_agents"
    agents_config: Path | None = Path("agents.yaml")
    models_config: Path = Path("models.yaml")
    executors_config: Path = Path("executors.yaml")
    runs_root: Path = Path("../runs")
    temporary_root: Path = Path("../.runtime")
    input_worker: NonEmptyString
    worker_timeout_seconds: Annotated[float, Field(gt=0)] = 300.0
    max_turns: Annotated[int, Field(gt=0)] = 10
    scheduler: SchedulerConfig
    planner: PlannerModelConfig

    @model_validator(mode="after")
    def validate_agent_config_for_mode(self) -> BlindExperimentConfig:
        """Require presets only in modes that expose them to the Planner."""
        if self.planner_mode in {"static_agents", "hybrid"} and self.agents_config is None:
            raise ValueError(f"planner_mode {self.planner_mode!r} requires agents_config")
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

    async def check(worker_id: str, client: WorkerClient) -> tuple[str, list[str]]:
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
        return worker_id, status.executors

    checked = await asyncio.gather(
        *(check(worker_id, client) for worker_id, client in clients.items())
    )
    return PreflightResult(workers=dict(checked))


def build_blind_scheduler(
    config: SchedulerConfig,
    registry: ExecutorRegistry,
    models: ModelRegistry,
    agents: AgentRegistry | None = None,
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
    return RoundRobinScheduler(registry)


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
    build_blind_scheduler(config.scheduler, registry, models, agents)
    if config.input_worker not in registry.worker_endpoints():
        raise ValueError(f"unknown input_worker: {config.input_worker!r}")
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
    runs_root = resolve_config_path(config.runs_root, directory)
    temporary_root = resolve_config_path(config.temporary_root, directory)
    agents = AgentRegistry.from_yaml(agents_path) if agents_path is not None else None
    models = ModelRegistry.from_yaml(models_path)
    registry = ExecutorRegistry.from_yaml(executors_path)
    scheduler = build_blind_scheduler(config.scheduler, registry, models, agents)
    if config.input_worker not in registry.worker_endpoints():
        raise ValueError(f"unknown input_worker: {config.input_worker!r}")

    effective_run_id = run_id or f"blind-{uuid4().hex[:12]}"
    trace = TraceRecorder(runs_root, effective_run_id, exclusive=True)
    clients = create_worker_clients(registry, config.worker_timeout_seconds)
    planner_client = None
    effective_config: dict[str, object] = {
        "run_id": effective_run_id,
        "mode": "resource_blind",
        "planner_mode": config.planner_mode,
        "resource_aware": False,
        "task": task,
        "experiment_config": str(config_path),
        "agents_config": str(agents_path) if agents_path is not None else None,
        "models_config": str(models_path),
        "executors_config": str(executors_path),
        "runs_root": str(runs_root),
        "temporary_root": str(temporary_root),
        "input_worker": config.input_worker,
        "worker_timeout_seconds": config.worker_timeout_seconds,
        "max_turns": config.max_turns,
        "input_paths": [str(path.resolve()) for path in inputs],
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
        await preflight_workers(registry, clients)
        initial_artifacts = await upload_inputs(
            inputs,
            effective_run_id,
            config.input_worker,
            clients[config.input_worker],
        )
        transfer = TransferManager(clients, trace)
        manager = ExecutionManager(clients, transfer, trace)
        runtime = AgentRuntime(
            agents,
            scheduler,
            manager,
            trace,
            request_id_factory=lambda: f"{effective_run_id}/request-{uuid4().hex}",
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
        )
        answer = await Coordinator(
            context,
            model=planner_model,
            max_turns=config.max_turns,
        ).run(task)
        await trace.end({"success": True, "answer": answer})
        return answer, trace.result_path.parent
    except Exception as error:
        await trace.end(
            {
                "success": False,
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )
        raise
    finally:
        if planner_client is not None:
            await planner_client.close()
        await close_worker_clients(clients)
