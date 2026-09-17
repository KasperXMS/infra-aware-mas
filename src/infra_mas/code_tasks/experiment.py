"""Run one open-ended SWE-bench code task against a materialized world."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from time import perf_counter
from typing import Annotated, Literal, cast

import yaml
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from infra_mas.code_tasks.client import CodeWorkerClient
from infra_mas.code_tasks.context import CodePlannerContext
from infra_mas.code_tasks.coordinator import CodeTaskCoordinator
from infra_mas.code_tasks.models import CodeExecutor, RepositoryWorld
from infra_mas.code_tasks.scheduler import RepositoryLocalityScheduler
from infra_mas.planner.context import InfrastructureVisibility
from infra_mas.planner.model_factory import PlannerModelConfig, create_planner_model
from infra_mas.tracing.recorder import TraceRecorder

NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class CodeBenchmarkConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["code-benchmark-v1"]
    task_id: NonEmptyString
    benchmark: Literal["SWE-bench Verified"]
    base_commit: NonEmptyString
    admission_exports: dict[NonEmptyString, Path]
    repository_artifact_id: NonEmptyString
    repository_size_bytes: Annotated[int, Field(ge=0)]
    planner_site: NonEmptyString = "cloud"
    worlds: dict[NonEmptyString, RepositoryWorld]
    executors: list[CodeExecutor]
    planner: PlannerModelConfig
    runs_root: Path = Path("../../runs/semantic_switch/astropy__astropy-14309")
    max_turns: Annotated[int, Field(gt=0)] = 20
    planner_max_output_tokens: Annotated[int, Field(gt=0)] = 4096
    max_exploration_calls: Annotated[int, Field(gt=0)] = 16
    worker_timeout_seconds: Annotated[float, Field(gt=0)] = 900

    @classmethod
    def from_yaml(cls, path: Path) -> CodeBenchmarkConfig:
        raw: object = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls.model_validate(raw)


class CodeRunResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    run_id: str
    world_id: str
    visibility: InfrastructureVisibility
    success: bool
    submitted_patch: bool
    final_answer: str
    e2e_ms: float
    planner_latency_ms: float
    planner_tokens: int
    remote_reasoning_calls: int
    tool_counts: dict[str, int]
    targeted_test_count: int
    full_test_count: int
    retry_replanning_count: int
    cross_site_transfer_bytes: int
    cross_site_transfer_ms: float
    transmitted_code_context_bytes: int
    code_tool_service_ms: float
    realized_workflow: list[dict[str, object]]
    patch_path: str | None
    official_resolved: bool | None = None
    error_type: str | None = None
    error: str | None = None


def _read_events(path: Path) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value: object = json.loads(line)
        if isinstance(value, dict):
            events.append(cast(dict[str, object], value))
    return events


def _usage_tokens(event: dict[str, object]) -> int:
    usage = event.get("token_usage")
    if not isinstance(usage, dict):
        return 0
    value = cast(dict[str, object], usage).get("total_tokens", 0)
    return int(cast(int | str, value))


def _as_float(value: object) -> float:
    return float(cast(float | int | str, value))


def _as_int(value: object) -> int:
    return int(cast(int | str, value))


def load_admission_export(
    config: CodeBenchmarkConfig, config_path: Path, world_id: str
) -> tuple[str, str]:
    try:
        raw_path = config.admission_exports[world_id]
    except KeyError as error:
        raise ValueError(f"no frozen admission export configured for world {world_id!r}") from error
    export_path = raw_path if raw_path.is_absolute() else (config_path.parent / raw_path).resolve()
    raw: object = json.loads(export_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("admission export must contain one JSON object")
    export = cast(dict[str, object], raw)
    task = export.get("task")
    infrastructure = export.get("infrastructure")
    contract = export.get("planner_contract")
    if not isinstance(task, dict) or not isinstance(infrastructure, dict):
        raise ValueError("admission export is missing task or infrastructure facts")
    if (
        not isinstance(contract, dict)
        or cast(dict[str, object], contract).get("oracle_free") is not True
    ):
        raise ValueError("admission export is not marked oracle-free")
    task_data = cast(dict[str, object], task)
    infra_data = cast(dict[str, object], infrastructure)
    if task_data.get("task_id") != config.task_id:
        raise ValueError("admission export task ID does not match benchmark config")
    if task_data.get("benchmark") != config.benchmark:
        raise ValueError("admission export benchmark does not match benchmark config")
    artifact_refs = task_data.get("artifact_refs")
    expected_ref = f"repo://astropy/astropy@{config.base_commit}"
    if artifact_refs != [expected_ref]:
        raise ValueError("admission export repository/base commit does not match config")
    world = config.worlds[world_id]
    expected_locality = "edge" if world.repository_site == "edge" else "cloud"
    if infra_data.get("artifact_locality") != expected_locality:
        raise ValueError("materialized world does not match frozen artifact locality")
    network = infra_data.get("network")
    if not isinstance(network, dict):
        raise ValueError("admission export has no network facts")
    network_data = cast(dict[str, object], network)
    exported_bandwidth = _as_float(network_data.get("bandwidth_mbps", 0))
    if exported_bandwidth != world.bandwidth_mbps:
        raise ValueError("materialized bandwidth does not match frozen world")
    if float(cast(float | int | str, network_data.get("rtt_ms", -1))) != world.rtt_ms:
        raise ValueError("materialized RTT does not match frozen world")
    instruction = task_data.get("instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("admission export instruction is empty")
    return instruction, str(export_path)


def summarize_code_run(
    run_directory: Path,
    *,
    task_id: str,
    world_id: str,
    visibility: InfrastructureVisibility,
    final_answer: str,
    e2e_ms: float,
    patch_path: Path | None,
    error: Exception | None = None,
) -> CodeRunResult:
    events = _read_events(run_directory / "trace.jsonl")
    llm_events = [
        item
        for item in events
        if item.get("event_type") == "planner.llm.end" and item.get("success") is True
    ]
    tool_events = [item for item in events if item.get("event_type") == "code_tool.end"]
    counts = Counter(str(item.get("tool")) for item in tool_events)
    workflow = [
        {
            "step": index,
            "tool": item.get("tool"),
            "site": item.get("site"),
            "success": item.get("success"),
            "service_ms": item.get("service_ms", 0),
            "cross_site_transfer_bytes": item.get("cross_site_transfer_bytes", 0),
        }
        for index, item in enumerate(tool_events, start=1)
    ]
    result = CodeRunResult(
        task_id=task_id,
        run_id=run_directory.name,
        world_id=world_id,
        visibility=visibility,
        success=patch_path is not None and error is None,
        submitted_patch=patch_path is not None,
        final_answer=final_answer,
        e2e_ms=e2e_ms,
        planner_latency_ms=sum(_as_float(item.get("latency_ms", 0)) for item in llm_events),
        planner_tokens=sum(_usage_tokens(item) for item in llm_events),
        remote_reasoning_calls=len(llm_events),
        tool_counts=dict(sorted(counts.items())),
        targeted_test_count=counts["run_targeted_test"],
        full_test_count=counts["run_full_test"],
        retry_replanning_count=sum(item.get("success") is False for item in tool_events),
        cross_site_transfer_bytes=sum(
            _as_int(item.get("cross_site_transfer_bytes", 0)) for item in tool_events
        ),
        cross_site_transfer_ms=sum(
            _as_float(item.get("cross_site_transfer_ms", 0)) for item in tool_events
        ),
        transmitted_code_context_bytes=sum(
            _as_int(item.get("planner_context_bytes", 0)) for item in tool_events
        ),
        code_tool_service_ms=sum(_as_float(item.get("service_ms", 0)) for item in tool_events),
        realized_workflow=workflow,
        patch_path=str(patch_path) if patch_path is not None else None,
        error_type=type(error).__name__ if error is not None else None,
        error=str(error) if error is not None else None,
    )
    (run_directory / "code_result.json").write_text(
        json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


async def run_code_benchmark(
    config_path: Path,
    world_id: str,
    visibility: InfrastructureVisibility,
    *,
    run_id: str,
) -> tuple[CodeRunResult, Path]:
    if visibility not in {"static", "snapshot"}:
        raise ValueError("code benchmark visibility must be static or snapshot")
    config_path = config_path.resolve()
    config = CodeBenchmarkConfig.from_yaml(config_path)
    try:
        world = config.worlds[world_id]
    except KeyError as error:
        raise ValueError(f"unknown world: {world_id}") from error
    if world.repository_artifact_id != config.repository_artifact_id:
        raise ValueError("world repository artifact does not match benchmark config")
    if world.repository_size_bytes != config.repository_size_bytes:
        raise ValueError("world repository size does not match benchmark config")
    instruction, admission_export_path = load_admission_export(config, config_path, world_id)

    scheduler = RepositoryLocalityScheduler(config.executors)
    executor = scheduler.select(world)
    worker = CodeWorkerClient(executor.endpoint, timeout_seconds=config.worker_timeout_seconds)
    health = await worker.health()
    if health.get("site") != world.repository_site:
        raise ValueError("selected code Worker site does not match repository world")
    if health.get("base_commit") != config.base_commit:
        raise ValueError("selected code Worker base commit does not match benchmark case")
    reset = await worker.reset(run_id)
    await worker.aclose()
    if reset.head_commit != config.base_commit or not reset.clean:
        raise ValueError("code Worker failed to reset the repository to a clean base commit")

    runs_root = config.runs_root
    if not runs_root.is_absolute():
        runs_root = (config_path.parent / runs_root).resolve()
    trace = TraceRecorder(runs_root, run_id, exclusive=True)
    effective_config: dict[str, object] = {
        "schema_version": config.schema_version,
        "task_id": config.task_id,
        "benchmark": config.benchmark,
        "base_commit": config.base_commit,
        "world_id": world_id,
        "visibility": visibility,
        "planner_site": config.planner_site,
        "planner": config.planner.model_dump(mode="json"),
        "max_turns": config.max_turns,
        "planner_max_output_tokens": config.planner_max_output_tokens,
        "max_exploration_calls": config.max_exploration_calls,
        "tools": [
            "search_code",
            "read_file",
            "edit_file",
            "apply_patch",
            "run_targeted_test",
            "run_full_test",
            "submit_patch",
        ],
        "scheduler": "repository_locality",
        "selected_executor": executor.model_dump(mode="json"),
        "world": world.model_dump(mode="json"),
        "admission_export": admission_export_path,
    }
    await trace.start(effective_config)
    planner_model, planner_client = create_planner_model(config.planner)
    context = CodePlannerContext(
        trace=trace,
        scheduler=scheduler,
        world=world,
        visibility=visibility,
        planner_site=config.planner_site,
        timeout_seconds=config.worker_timeout_seconds,
        max_exploration_calls=config.max_exploration_calls,
    )
    final_answer = ""
    started = perf_counter()
    try:
        final_answer = await CodeTaskCoordinator(
            context,
            model=planner_model,
            max_turns=config.max_turns,
            max_output_tokens=config.planner_max_output_tokens,
        ).run(instruction)
        e2e_ms = (perf_counter() - started) * 1000
        patch_path = None
        if context.submitted_patch:
            patch_path = trace.result_path.parent / "model.patch"
            patch_path.write_text(context.submitted_patch, encoding="utf-8")
        run_result: dict[str, object] = {
            "success": patch_path is not None,
            "answer": final_answer,
            "e2e_ms": e2e_ms,
            "submitted_patch": patch_path is not None,
        }
        await trace.end(run_result)
        result = summarize_code_run(
            trace.result_path.parent,
            task_id=config.task_id,
            world_id=world_id,
            visibility=visibility,
            final_answer=final_answer,
            e2e_ms=e2e_ms,
            patch_path=patch_path,
        )
        return result, trace.result_path.parent
    except Exception as error:
        e2e_ms = (perf_counter() - started) * 1000
        patch_path = None
        if context.submitted_patch:
            patch_path = trace.result_path.parent / "model.patch"
            patch_path.write_text(context.submitted_patch, encoding="utf-8")
        await trace.end(
            {
                "success": False,
                "error_type": type(error).__name__,
                "error": str(error),
                "answer": final_answer,
                "e2e_ms": e2e_ms,
                "submitted_patch": patch_path is not None,
            }
        )
        result = summarize_code_run(
            trace.result_path.parent,
            task_id=config.task_id,
            world_id=world_id,
            visibility=visibility,
            final_answer=final_answer,
            e2e_ms=e2e_ms,
            patch_path=patch_path,
            error=error,
        )
        return result, trace.result_path.parent
    finally:
        await planner_client.close()
