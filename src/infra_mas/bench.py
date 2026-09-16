"""Oracle-free benchmark bridge for the open-ended real-system MAS."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, cast
from urllib.parse import unquote, urlparse

import yaml
from pydantic import BaseModel, ConfigDict, Field

from infra_mas.core.resource import NetworkLink
from infra_mas.execution.executor_registry import ExecutorRegistry
from infra_mas.experiment import BlindExperimentConfig, resolve_config_path, run_blind_experiment
from infra_mas.planner.context import InfrastructureVisibility


class ArtifactBinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artifact_id: str
    source_ref: str
    site_id: str
    size_bytes: int | None = Field(default=None, ge=0)


class MASRunSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["mas-run-spec-v1"]
    case_id: str
    group_id: str
    task_id: str
    instruction: str
    artifacts: list[ArtifactBinding]
    infra_world: dict[str, Any]
    metadata: dict[str, Any] = Field(default_factory=dict)


class RealizedAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_id: str
    parent_action_id: str | None = None
    model_id: str | None = None
    role: str | None = None
    input_artifacts: list[str] = Field(default_factory=list)
    output_artifacts: list[str] = Field(default_factory=list)
    executor_id: str | None = None
    site_id: str | None = None
    started_at: str | None = None
    finished_at: str | None = None


class RealizedWorkflow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    actions: list[RealizedAction] = []
    dependencies: list[tuple[str, str]] = []


class MASExecutionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["mas-execution-result-v1"] = "mas-execution-result-v1"
    case_id: str
    group_id: str
    task_id: str
    visibility: InfrastructureVisibility
    run_id: str
    final_answer: str
    e2e_ms: float = Field(ge=0)
    planner_latency_ms: float = Field(ge=0)
    planner_tokens: int = Field(ge=0)
    invocation_count: int = Field(ge=0)
    model_counts: dict[str, int]
    service_ms_sum: float = Field(ge=0)
    transfer_bytes: int = Field(ge=0)
    transfer_ms_sum: float = Field(ge=0)
    raw_transfer_bytes: int = Field(default=0, ge=0)
    derived_transfer_bytes: int = Field(default=0, ge=0)
    cross_worker_transfer_bytes: int = Field(default=0, ge=0)
    cross_worker_transfer_count: int = Field(default=0, ge=0)
    multimodal_invocation_count: int = Field(default=0, ge=0)
    site_aligned_multimodal_invocations: int = Field(default=0, ge=0)
    site_alignment_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    artifact_grouping: list[list[str]] = []
    realized_workflow: RealizedWorkflow


def load_mas_case(path: Path) -> MASRunSpec:
    return MASRunSpec.model_validate_json(path.read_text(encoding="utf-8"))


def _experiment_config_path(
    config_path: Path, visibility: InfrastructureVisibility
) -> Path:
    """Accept either an experiment YAML or the pilot's config-of-configs."""
    config_path = config_path.resolve()
    raw: object = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("benchmark config must contain a YAML object")
    raw_mapping = cast(dict[object, object], raw)
    if "planner" in raw_mapping and "scheduler" in raw_mapping:
        return config_path
    key_by_visibility = {
        "none": "blind_config",
        "static": "static_config",
        "snapshot": "snapshot_config",
    }
    key = key_by_visibility[visibility]
    value = raw_mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"pilot config does not define {key}")
    return resolve_config_path(Path(value), config_path.parent)


def resolve_source_path(source_ref: str, case_directory: Path) -> Path:
    native_path = Path(source_ref)
    if native_path.is_absolute():
        return native_path.resolve()
    parsed = urlparse(source_ref)
    if parsed.scheme == "file":
        decoded = unquote(parsed.path)
        if len(decoded) >= 3 and decoded[0] == "/" and decoded[2] == ":":
            decoded = decoded[1:]
        raw = Path(decoded)
    elif parsed.scheme:
        raise ValueError(
            f"source_ref {source_ref!r} is not a materialized local file; "
            "materialize it before real-system execution"
        )
    else:
        raw = Path(source_ref)
    return (raw if raw.is_absolute() else case_directory / raw).resolve()


def _workers_for_sites(
    registry: ExecutorRegistry, bindings: Sequence[ArtifactBinding]
) -> list[str]:
    workers_by_site: dict[str, list[str]] = {}
    for worker_id, site_id in registry.worker_sites().items():
        workers_by_site.setdefault(site_id, []).append(worker_id)
    selected: list[str] = []
    for binding in bindings:
        candidates = sorted(workers_by_site.get(binding.site_id, []))
        if len(candidates) != 1:
            raise ValueError(
                f"site {binding.site_id!r} must map to exactly one configured Worker; "
                f"found {candidates}"
            )
        selected.append(candidates[0])
    return selected


def _network_links(spec: MASRunSpec) -> list[NetworkLink] | None:
    raw_links = spec.infra_world.get("links")
    if raw_links is None:
        return None
    if not isinstance(raw_links, list):
        raise ValueError("infra_world.links must be a list")
    links: list[NetworkLink] = []
    for raw_item in cast(list[object], raw_links):
        if not isinstance(raw_item, dict):
            raise ValueError("every infra_world link must be an object")
        item = cast(dict[str, object], raw_item)
        links.append(
            NetworkLink(
                source_site=str(item["src_site"]),
                target_site=str(item["dst_site"]),
                bandwidth_mbps=float(cast(float | int | str, item["bandwidth_mbps"])),
                rtt_ms=float(cast(float | int | str, item["rtt_ms"])),
                bidirectional=bool(item.get("bidirectional", True)),
            )
        )
    return links


def _eligible_executors(spec: MASRunSpec) -> list[str] | None:
    value = spec.infra_world.get("eligible_executor_ids")
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError("infra_world.eligible_executor_ids must be a list of IDs")
    result: list[str] = []
    for raw_item in cast(list[object], value):
        if not isinstance(raw_item, str) or not raw_item.strip():
            raise ValueError("infra_world.eligible_executor_ids must be a list of IDs")
        result.append(raw_item)
    return result


def _json_object(line: str) -> dict[str, Any]:
    value: object = json.loads(line)
    if not isinstance(value, dict):
        raise ValueError("expected one JSON object")
    return cast(dict[str, Any], value)


def export_realized_workflow(trace_path: Path) -> RealizedWorkflow:
    """Reconstruct action dependencies from produced/consumed artifacts."""
    events: list[dict[str, Any]] = [
        _json_object(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    requests = [event for event in events if event["event_type"] == "execution.request"]
    selected = {
        event["action_id"]: event
        for event in events
        if event["event_type"] == "executor.selected"
    }
    started = {
        event["action_id"]: event
        for event in events
        if event["event_type"] == "worker.execution.start"
    }
    finished = {
        event["action_id"]: event
        for event in events
        if event["event_type"] == "worker.execution.end" and event.get("success")
    }
    actions: list[RealizedAction] = []
    producer_by_artifact: dict[str, str] = {}
    for request in requests:
        action_id = str(request["action_id"])
        binding = selected.get(action_id, {})
        end = finished.get(action_id, {})
        outputs = [str(value) for value in end.get("output_artifacts", [])]
        action = RealizedAction(
            action_id=action_id,
            parent_action_id=request.get("parent_action_id"),
            model_id=request.get("model_id"),
            role=request.get("agent"),
            input_artifacts=[str(value) for value in request.get("input_artifacts", [])],
            output_artifacts=outputs,
            executor_id=binding.get("executor"),
            site_id=binding.get("site"),
            started_at=started.get(action_id, request).get("timestamp"),
            finished_at=end.get("timestamp"),
        )
        actions.append(action)
        for artifact_id in outputs:
            producer_by_artifact[artifact_id] = action_id

    dependencies = sorted(
        {
            (producer_by_artifact[artifact_id], action.action_id)
            for action in actions
            for artifact_id in action.input_artifacts
            if artifact_id in producer_by_artifact
            and producer_by_artifact[artifact_id] != action.action_id
        }
    )
    return RealizedWorkflow(actions=actions, dependencies=dependencies)


def _event_total_tokens(event: dict[str, Any]) -> int:
    usage = event.get("token_usage")
    if not isinstance(usage, dict):
        return 0
    value = cast(dict[str, object], usage).get("total_tokens", 0)
    return int(cast(int | str, value))


def summarize_mas_run(
    run_directory: Path,
    spec: MASRunSpec,
    visibility: InfrastructureVisibility,
) -> MASExecutionResult:
    events: list[dict[str, Any]] = [
        _json_object(line)
        for line in (run_directory / "trace.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    result = _json_object((run_directory / "result.json").read_text(encoding="utf-8"))
    workflow = export_realized_workflow(run_directory / "trace.jsonl")
    llm_ends = [
        event
        for event in events
        if event["event_type"] == "planner.llm.end" and event.get("success")
    ]
    executions = [
        event
        for event in events
        if event["event_type"] == "worker.execution.end" and event.get("success")
    ]
    transfers = [
        event
        for event in events
        if event["event_type"] == "artifact.transfer.end" and event.get("success")
    ]
    raw_artifact_ids: set[str] = set()
    ordered_raw_artifact_ids: list[str] = []
    for event in events:
        if event["event_type"] != "planner.ledger.initialized":
            continue
        planning_state = cast(dict[str, Any], event.get("planning_state", {}))
        for value in cast(list[object], planning_state.get("initial_inputs", [])):
            artifact_id = (
                str(cast(dict[str, object], value).get("artifact_id"))
                if isinstance(value, dict)
                else str(value)
            )
            raw_artifact_ids.add(artifact_id)
            ordered_raw_artifact_ids.append(artifact_id)
    benchmark_id_by_runtime_id = {
        runtime_id: binding.artifact_id
        for runtime_id, binding in zip(
            ordered_raw_artifact_ids, spec.artifacts, strict=True
        )
    }
    model_counts: dict[str, int] = {}
    for action in workflow.actions:
        if action.model_id is not None:
            model_counts[action.model_id] = model_counts.get(action.model_id, 0) + 1
    raw_transfer_bytes = sum(
        int(event.get("bytes_transferred", 0))
        for event in transfers
        if event.get("artifact_id") in raw_artifact_ids
        or "/input-" in str(event.get("artifact_id", ""))
    )
    transfer_bytes = sum(int(event.get("bytes_transferred", 0)) for event in transfers)
    cross_worker_transfers = [
        event
        for event in transfers
        if event.get("source_worker_id") != event.get("target_worker_id")
    ]
    transferred_raw_by_action = {
        (str(event.get("action_id")), str(event.get("artifact_id")))
        for event in cross_worker_transfers
        if event.get("artifact_id") in raw_artifact_ids
    }
    multimodal_actions = [
        action
        for action in workflow.actions
        if any(artifact_id in raw_artifact_ids for artifact_id in action.input_artifacts)
    ]
    aligned_actions = [
        action
        for action in multimodal_actions
        if all(
            (action.action_id, artifact_id) not in transferred_raw_by_action
            for artifact_id in action.input_artifacts
            if artifact_id in raw_artifact_ids
        )
    ]
    artifact_grouping = [
        [
            benchmark_id_by_runtime_id[artifact_id]
            for artifact_id in action.input_artifacts
            if artifact_id in benchmark_id_by_runtime_id
        ]
        for action in multimodal_actions
    ]
    summary = MASExecutionResult(
        case_id=spec.case_id,
        group_id=spec.group_id,
        task_id=spec.task_id,
        visibility=visibility,
        run_id=str(events[0]["run_id"]),
        final_answer=str(result["answer"]),
        e2e_ms=float(result["e2e_ms"]),
        planner_latency_ms=sum(float(event.get("latency_ms", 0)) for event in llm_ends),
        planner_tokens=sum(_event_total_tokens(event) for event in llm_ends),
        invocation_count=len(workflow.actions),
        model_counts=model_counts,
        service_ms_sum=sum(float(event.get("service_ms", 0)) for event in executions),
        transfer_bytes=transfer_bytes,
        transfer_ms_sum=sum(float(event.get("transfer_ms", 0)) for event in transfers),
        raw_transfer_bytes=raw_transfer_bytes,
        derived_transfer_bytes=transfer_bytes - raw_transfer_bytes,
        cross_worker_transfer_bytes=sum(
            int(event.get("bytes_transferred", 0)) for event in cross_worker_transfers
        ),
        cross_worker_transfer_count=len(cross_worker_transfers),
        multimodal_invocation_count=len(multimodal_actions),
        site_aligned_multimodal_invocations=len(aligned_actions),
        site_alignment_rate=(
            len(aligned_actions) / len(multimodal_actions) if multimodal_actions else 0.0
        ),
        artifact_grouping=artifact_grouping,
        realized_workflow=workflow,
    )
    (run_directory / "mas_result.json").write_text(
        json.dumps(summary.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


async def run_benchmark_case(
    case_path: Path,
    config_path: Path,
    visibility: InfrastructureVisibility,
    *,
    run_id: str | None = None,
) -> tuple[MASExecutionResult, Path]:
    """Materialize one exported world and execute the normal open-ended Planner."""
    case_path = case_path.resolve()
    spec = load_mas_case(case_path)
    experiment_path = _experiment_config_path(config_path, visibility)
    experiment = BlindExperimentConfig.from_yaml(experiment_path)
    registry = ExecutorRegistry.from_yaml(
        resolve_config_path(experiment.executors_config, experiment_path.parent)
    )
    paths = [resolve_source_path(item.source_ref, case_path.parent) for item in spec.artifacts]
    for path, binding in zip(paths, spec.artifacts, strict=True):
        if not path.is_file():
            raise FileNotFoundError(f"benchmark artifact not found: {path}")
        if binding.size_bytes is not None and path.stat().st_size != binding.size_bytes:
            raise ValueError(
                f"artifact {binding.artifact_id!r} size is {path.stat().st_size}, "
                f"expected {binding.size_bytes}"
            )
    workers = _workers_for_sites(registry, spec.artifacts)
    effective_run_id = run_id or f"mas-{spec.case_id.replace(':', '-')}-{visibility}"
    _, run_directory = await run_blind_experiment(
        experiment_path,
        spec.instruction,
        paths,
        run_id=effective_run_id,
        input_workers=workers,
        artifact_ids=[item.artifact_id for item in spec.artifacts],
        infrastructure_visibility=visibility,
        network_links=_network_links(spec),
        eligible_executor_ids=_eligible_executors(spec),
        run_metadata={
            "benchmark_case_id": spec.case_id,
            "benchmark_group_id": spec.group_id,
            "benchmark_task_id": spec.task_id,
            "mas_schema_version": spec.schema_version,
        },
    )
    return summarize_mas_run(run_directory, spec, visibility), run_directory
