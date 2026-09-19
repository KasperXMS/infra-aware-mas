"""Planner-free MultiHop-RAG references for Scope Expansion v0.

The workflow surface is dataset-independent. Candidate construction and gold-evidence
coverage checks happen offline; this module sees only a query and a fixed candidate bundle.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path, PurePosixPath
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
from infra_mas.experiment import bind_placed_inputs, close_worker_clients, create_worker_clients
from infra_mas.operators import (
    GENERAL_OPERATOR_REGISTRY,
    ExternalEvaluatorSpec,
    InitialArtifactSpec,
    ObservationSpec,
    RuntimeVerifierSpec,
    TaskInteractionSpec,
)
from infra_mas.runtime.model_registry import ModelRegistry
from infra_mas.runtime.runtime import AgentRuntime
from infra_mas.scheduler.fixed import FixedScheduler
from infra_mas.tracing.recorder import TraceRecorder

NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
ScopeWorkflow = Literal["centralized_raw", "distributed_retrieval"]
ScopeWorld = Literal["H1_distributed_constrained", "H2_distributed_favorable"]

SCOPE_WORKFLOWS: tuple[ScopeWorkflow, ...] = (
    "centralized_raw",
    "distributed_retrieval",
)
MEASUREMENT_PROTOCOL = "steady_state_1_warmup_3_measured_v1"
SHARD_WORKERS = ("a4", "a5", "a28")
SHARD_SITES = ("A4", "A5", "A28")


class ScopeRuntime(Protocol):
    async def invoke(
        self,
        invocation: InvocationSpec,
        *,
        parent_action_id: str | None = None,
    ) -> ExecutionResult: ...

    async def bm25_retrieve_on_worker(
        self,
        query: str,
        artifacts: list[ArtifactRef],
        target_worker_id: str,
        *,
        top_k: int = 3,
        parent_action_id: str | None = None,
    ) -> ExecutionResult: ...


class ScopeInput(BaseModel):
    """One independently placed candidate document."""

    model_config = ConfigDict(extra="forbid")

    document_id: NonEmptyString
    artifact_id: NonEmptyString
    path: Path
    worker_id: Literal["a4", "a5", "a28"]
    expected_size_bytes: int = Field(gt=0)
    expected_tokens: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_deterministic_shard(self) -> ScopeInput:
        expected = stable_document_worker(self.document_id)
        if self.worker_id != expected:
            raise ValueError(
                f"document {self.document_id!r} must be placed on {expected!r}, "
                f"not {self.worker_id!r}"
            )
        return self


class ScopeTask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: NonEmptyString
    query: NonEmptyString
    evaluator_id: NonEmptyString
    inputs: Annotated[list[ScopeInput], Field(min_length=1, max_length=50)]

    @model_validator(mode="after")
    def validate_documents(self) -> ScopeTask:
        document_ids = [item.document_id for item in self.inputs]
        artifact_ids = [item.artifact_id for item in self.inputs]
        if len(document_ids) != len(set(document_ids)):
            raise ValueError("document IDs must be unique within a task")
        if len(artifact_ids) != len(set(artifact_ids)):
            raise ValueError("artifact IDs must be unique within a task")
        return self


class ScopeNetworkWorld(BaseModel):
    model_config = ConfigDict(extra="forbid")

    world_id: ScopeWorld
    bandwidth_mbps: float = Field(gt=0, allow_inf_nan=False)
    rtt_ms: float = Field(ge=0, allow_inf_nan=False)


class ScopeExpansionSweepConfig(BaseModel):
    """Controlled Jetson-only MultiHop-RAG reference sweep."""

    model_config = ConfigDict(extra="forbid")

    models_config: Path
    executors_config: Path
    final_model_id: NonEmptyString
    final_executor_id: NonEmptyString
    input_source: Literal["worker_local"] = "worker_local"
    candidate_top_n: Literal[30, 50] = 30
    local_top_k_per_shard: Literal[3] = 3
    tasks: Annotated[list[ScopeTask], Field(min_length=1, max_length=3)]
    worlds: Annotated[list[ScopeNetworkWorld], Field(min_length=2, max_length=2)]
    workflows: Annotated[list[ScopeWorkflow], Field(min_length=2, max_length=2)] = list(
        SCOPE_WORKFLOWS
    )
    warmup_runs: Literal[1] = 1
    repeats: Literal[3] = 3
    worker_timeout_seconds: float = Field(default=1800, gt=0)
    run_prefix: NonEmptyString = "scope-expansion-v0-multihop-rag"

    @model_validator(mode="after")
    def validate_controlled_sweep(self) -> ScopeExpansionSweepConfig:
        if set(self.workflows) != set(SCOPE_WORKFLOWS):
            raise ValueError("both controlled MultiHop-RAG workflows are required")
        by_id = {world.world_id: world for world in self.worlds}
        if set(by_id) != {
            "H1_distributed_constrained",
            "H2_distributed_favorable",
        }:
            raise ValueError("both H1 and H2 network worlds are required")
        h1 = by_id["H1_distributed_constrained"]
        h2 = by_id["H2_distributed_favorable"]
        if h1.bandwidth_mbps >= h2.bandwidth_mbps or h1.rtt_ms <= h2.rtt_ms:
            raise ValueError("H1 must have lower bandwidth and higher RTT than H2")
        if len(self.workflows) != len(set(self.workflows)):
            raise ValueError("workflow IDs must be unique")
        task_ids = [task.task_id for task in self.tasks]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("task IDs must be unique")
        for task in self.tasks:
            if len(task.inputs) != self.candidate_top_n:
                raise ValueError(
                    f"task {task.task_id!r} has {len(task.inputs)} documents; "
                    f"candidate_top_n is {self.candidate_top_n}"
                )
            if set(item.worker_id for item in task.inputs) != set(SHARD_WORKERS):
                raise ValueError(f"task {task.task_id!r} must place documents on all 3 Jetsons")
        return self

    @classmethod
    def from_yaml(cls, path: Path) -> ScopeExpansionSweepConfig:
        raw: object = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls.model_validate(raw)


class ReferenceResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    workflow_id: ScopeWorkflow
    executions: list[ExecutionResult]
    final_artifact: ArtifactRef
    raw_bytes: int = Field(ge=0)
    raw_tokens: int = Field(ge=0)
    reduced_bytes: int = Field(ge=0)
    reduced_tokens: int = Field(ge=0)
    retrieval_calls: int = Field(ge=0)
    retrieved_document_count: int = Field(ge=0)
    retrieval_sum_ms: float = Field(ge=0)
    retrieval_critical_ms: float = Field(ge=0)
    final_model_ms: float = Field(ge=0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    api_cost_usd: float = Field(ge=0)


def stable_document_worker(document_id: str) -> Literal["a4", "a5", "a28"]:
    """Map a document ID to a stable shard independently of task gold or world."""
    digest = hashlib.sha256(document_id.encode("utf-8")).digest()
    index = int.from_bytes(digest[:8], "big") % 3
    return SHARD_WORKERS[index]


def generate_scope_expansion_sweep(
    task_bank_path: Path,
    output_path: Path,
    *,
    remote_root: Path = Path("/home/edge/xiaoming/scope_expansion_v0"),
) -> ScopeExpansionSweepConfig:
    """Generate an execution manifest from planner-visible task-bank fields only."""
    records = [
        cast(dict[str, Any], json.loads(line))
        for line in task_bank_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    remote_root_posix = PurePosixPath(remote_root.as_posix())
    tasks: list[dict[str, object]] = []
    for task_index, record in enumerate(records, start=2):
        visible = cast(dict[str, Any], record["planner_visible"])
        artifact_ids = cast(list[str], visible["artifact_ids"])
        document_ids = cast(dict[str, str], visible["artifact_document_ids"])
        sizes = cast(dict[str, int], visible["artifact_sizes"])
        tokens = cast(dict[str, int], visible["artifact_tokens"])
        placements = cast(dict[str, str], visible["artifact_placement"])
        site_to_worker = {"A4": "a4", "A5": "a5", "A28": "a28"}
        inputs: list[dict[str, object]] = []
        for artifact_id in artifact_ids:
            site_id = placements[artifact_id]
            worker_id = site_to_worker[site_id]
            document_id = document_ids[artifact_id]
            inputs.append(
                {
                    "document_id": document_id,
                    "artifact_id": artifact_id,
                    "path": str(
                        remote_root_posix
                        / "multihop_rag"
                        / f"task_m{task_index}"
                        / "artifacts"
                        / site_id
                        / f"{artifact_id}.json"
                    ),
                    "worker_id": worker_id,
                    "expected_size_bytes": sizes[artifact_id],
                    "expected_tokens": tokens[artifact_id],
                }
            )
        tasks.append(
            {
                "task_id": str(visible["task_id"]),
                "query": str(visible["query"]),
                "evaluator_id": str(visible["evaluator_type"]),
                "inputs": inputs,
            }
        )
    payload: dict[str, object] = {
            "models_config": "models.yaml",
            "executors_config": "executors.yaml",
            "final_model_id": "edge-text-reasoner",
            "final_executor_id": "a28-vlm",
            "input_source": "worker_local",
            "candidate_top_n": 30,
            "local_top_k_per_shard": 3,
            "tasks": tasks,
            "worlds": [
                {
                    "world_id": "H1_distributed_constrained",
                    "bandwidth_mbps": 3.0,
                    "rtt_ms": 83.0,
                },
                {
                    "world_id": "H2_distributed_favorable",
                    "bandwidth_mbps": 100.0,
                    "rtt_ms": 33.0,
                },
            ],
            "workflows": list(SCOPE_WORKFLOWS),
            "warmup_runs": 1,
            "repeats": 3,
            "worker_timeout_seconds": 1800,
            "run_prefix": "scope-expansion-v0-multihop-rag",
        }
    config = ScopeExpansionSweepConfig.model_validate(payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.safe_dump(
            payload,
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    return config


def build_multidoc_task_interaction(
    task: ScopeTask,
    artifacts: list[ArtifactRef],
) -> TaskInteractionSpec:
    """Validate the dataset-neutral operator contract for a fixed candidate bundle."""
    interaction = TaskInteractionSpec(
        task_id=task.task_id,
        objective=task.query,
        initial_artifacts=[
            InitialArtifactSpec(
                artifact_id=artifact.id,
                kind="candidate_document",
                source_ref=item.document_id,
            )
            for item, artifact in zip(task.inputs, artifacts, strict=True)
        ],
        operators=["bm25_retrieve", "invoke_model"],
        observations=[
            ObservationSpec(
                observation_id="retrieved_evidence",
                produced_by=["bm25_retrieve"],
                description="Deterministic shard-local ranked evidence without gold metadata.",
            ),
            ObservationSpec(
                observation_id="final_answer",
                produced_by=["invoke_model"],
                description="Answer from either full candidate context or retrieved evidence.",
            ),
        ],
        runtime_verifier=RuntimeVerifierSpec(level="none"),
        external_evaluator=ExternalEvaluatorSpec(evaluator_id=task.evaluator_id),
    )
    GENERAL_OPERATOR_REGISTRY.validate_task(interaction)
    return interaction


def _final_task(query: str) -> str:
    return (
        f"Question: {query}\n"
        "Use only the supplied candidate context. Give a concise answer and finish with "
        "`ANSWER: <answer>`."
    )


async def execute_multidoc_reference(
    runtime: ScopeRuntime,
    *,
    workflow_id: ScopeWorkflow,
    query: str,
    documents: list[ArtifactRef],
    document_workers: list[str],
    raw_tokens: int,
    final_model_id: str,
    local_top_k_per_shard: Literal[3] = 3,
) -> ReferenceResult:
    """Execute one controlled reference; both branches use the same final model."""
    raw_bytes = sum(item.size_bytes for item in documents)
    final_instructions = (
        "You are the common final reasoning stage for a multi-document QA experiment. "
        "Reason across documents, do not use external knowledge, and follow the output format."
    )
    retrievals: list[ExecutionResult] = []
    if workflow_id == "centralized_raw":
        final_inputs = documents
        reduced_bytes = raw_bytes
        reduced_tokens = raw_tokens
    elif workflow_id == "distributed_retrieval":
        shards: dict[str, list[ArtifactRef]] = defaultdict(list)
        for artifact, worker_id in zip(documents, document_workers, strict=True):
            shards[worker_id].append(artifact)
        retrievals = list(
            await asyncio.gather(
                *(
                    runtime.bm25_retrieve_on_worker(
                        query,
                        shards[worker_id],
                        worker_id,
                        top_k=local_top_k_per_shard,
                    )
                    for worker_id in SHARD_WORKERS
                )
            )
        )
        final_inputs = [result.output_artifacts[0] for result in retrievals]
        reduced_bytes = sum(item.size_bytes for item in final_inputs)
        reduced_tokens = sum(
            int(result.metadata.get("retrieved_tokens", 0)) for result in retrievals
        )
    else:
        raise ValueError(f"unsupported Scope Expansion workflow: {workflow_id!r}")

    final = await runtime.invoke(
        InvocationSpec(
            model_id=final_model_id,
            role="scope-expansion-v0-final-reasoner",
            instructions=final_instructions,
            task=_final_task(query),
            input_artifacts=final_inputs,
        )
    )
    usage_results = [*retrievals, final]
    retrieval_sum_ms = sum(item.service_ms for item in retrievals)
    return ReferenceResult(
        workflow_id=workflow_id,
        executions=usage_results,
        final_artifact=final.output_artifacts[0],
        raw_bytes=raw_bytes,
        raw_tokens=raw_tokens,
        reduced_bytes=reduced_bytes,
        reduced_tokens=reduced_tokens,
        retrieval_calls=len(retrievals),
        retrieved_document_count=sum(
            int(item.metadata.get("retrieved_document_count", 0)) for item in retrievals
        ),
        retrieval_sum_ms=retrieval_sum_ms,
        retrieval_critical_ms=max((item.service_ms for item in retrievals), default=0.0),
        final_model_ms=final.service_ms,
        input_tokens=sum(int(item.metadata.get("input_tokens", 0)) for item in usage_results),
        output_tokens=sum(int(item.metadata.get("output_tokens", 0)) for item in usage_results),
        api_cost_usd=sum(
            float(item.metadata.get("api_cost_usd", 0.0)) for item in usage_results
        ),
    )


def _validate_deployment(
    config: ScopeExpansionSweepConfig,
    registry: ExecutorRegistry,
    models: ModelRegistry,
) -> None:
    sites = registry.worker_sites()
    if {worker: sites.get(worker) for worker in SHARD_WORKERS} != {
        "a4": "A4",
        "a5": "A5",
        "a28": "A28",
    }:
        raise ValueError("scope expansion workers must resolve to A4, A5, and A28")
    if any(site == "4090" for site in sites.values()):
        raise ValueError("Scope Expansion v0 MultiHop-RAG is Jetson-only")
    model_ids = {item.model_id for item in models.list()}
    if config.final_model_id not in model_ids:
        raise ValueError(f"unknown final model: {config.final_model_id!r}")
    executor = registry.get(config.final_executor_id)
    if executor.worker_id != "a28" or executor.site != "A28":
        raise ValueError("final_executor_id must be hosted by A28")
    if executor.model_id != config.final_model_id:
        raise ValueError("final executor must serve final_model_id")


async def _preflight_jetsons(
    registry: ExecutorRegistry,
    clients: Mapping[str, Any],
    final_executor_id: str,
) -> None:
    expected_workers = set(SHARD_WORKERS)
    if set(registry.worker_endpoints()) != expected_workers:
        raise ValueError("executors_config must contain exactly the three Jetson workers")

    async def check(worker_id: str) -> None:
        client = clients[worker_id]
        await client.health()
        await client.ready()
        status = await client.status()
        if status.worker_id != worker_id:
            raise ValueError(f"endpoint {worker_id!r} identifies as {status.worker_id!r}")
        if "bm25_retrieve" not in status.operators:
            raise ValueError(f"worker {worker_id!r} does not expose bm25_retrieve")
        if worker_id == "a28" and final_executor_id not in status.executors:
            raise ValueError(f"A28 does not expose final executor {final_executor_id!r}")

    await asyncio.gather(*(check(worker_id) for worker_id in SHARD_WORKERS))


def _attempt_schedule(repeats: int, warmup_runs: int) -> tuple[tuple[int, bool], ...]:
    if warmup_runs != 1:
        raise ValueError("Scope Expansion v0 requires exactly one warm-up per cell")
    return ((0, True), *((repeat, False) for repeat in range(1, repeats + 1)))


def _physical_run_id(
    task_id: str,
    world_id: ScopeWorld,
    workflow_id: ScopeWorkflow,
    repeat: int,
    warmup: bool,
    *,
    nonce: str | None = None,
) -> str:
    """Keep trace paths short while full semantic IDs remain in run metadata."""
    attempt_id = "warmup" if warmup else f"r{repeat}"
    world_code = "h1" if world_id == "H1_distributed_constrained" else "h2"
    workflow_code = "cr" if workflow_id == "centralized_raw" else "dr"
    selected_nonce = nonce or uuid4().hex[:8]
    return (
        f"se0-{task_id[-8:]}-{world_code}-{workflow_code}-"
        f"{attempt_id}-{selected_nonce}"
    )


def _completed_keys(path: Path, series_id: str) -> set[tuple[str, str, str, int, bool]]:
    if not path.is_file():
        return set()
    completed: set[tuple[str, str, str, int, bool]] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = cast(dict[str, object], json.loads(line))
        if row.get("status") != "completed":
            continue
        if row.get("measurement_series_id") != series_id:
            continue
        completed.add(
            (
                str(row["task_id"]),
                str(row["workflow_id"]),
                str(row["world_id"]),
                int(cast(int, row["repeat"])),
                bool(row["warmup"]),
            )
        )
    return completed


async def _download_answer(
    artifact: ArtifactRef,
    clients: Mapping[str, Any],
    temporary_root: Path,
) -> str:
    worker_id = next((item for item in artifact.locations if item in clients), None)
    if worker_id is None:
        raise ValueError(f"final artifact {artifact.id!r} has no reachable Worker")
    temporary_root.mkdir(parents=True, exist_ok=True)
    destination = temporary_root / f"scope-expansion-{uuid4().hex}.txt"
    try:
        await clients[worker_id].download_artifact(artifact.id, destination)
        return destination.read_text(encoding="utf-8")
    finally:
        if destination.is_file():
            destination.unlink()


def _read_trace(path: Path) -> list[dict[str, Any]]:
    return [
        cast(dict[str, Any], json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _transfer_rows(
    events: list[dict[str, Any]], registry: ExecutorRegistry
) -> list[dict[str, object]]:
    sites = registry.worker_sites()
    producers = {
        str(event["artifact_id"]): str(event["action_id"])
        for event in events
        if event.get("event_type") == "artifact.created"
    }
    pending: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    rows: list[dict[str, object]] = []
    for event in events:
        event_type = event.get("event_type")
        if event_type not in {"artifact.transfer.start", "artifact.transfer.end"}:
            continue
        key = (
            str(event.get("action_id")),
            str(event.get("artifact_id")),
            str(event.get("source_worker_id")),
            str(event.get("target_worker_id")),
        )
        if event_type == "artifact.transfer.start":
            pending.setdefault(key, []).append(event)
            continue
        starts = pending.get(key, [])
        start = starts.pop(0) if starts else None
        if not event.get("success") or int(event.get("bytes_transferred", 0)) <= 0:
            continue
        source_worker = str(event["source_worker_id"])
        target_worker = str(event["target_worker_id"])
        artifact_id = str(event["artifact_id"])
        consumer = str(event["action_id"])
        rows.append(
            {
                "transfer_id": f"{consumer}:transfer-{len(rows) + 1}",
                "artifact_id": artifact_id,
                "src_agent": source_worker,
                "dst_agent": target_worker,
                "src_site": sites[source_worker],
                "dst_site": sites[target_worker],
                "bytes": int(event["bytes_transferred"]),
                "latency_ms": float(event["transfer_ms"]),
                "started_at": start["timestamp"] if start is not None else None,
                "finished_at": event["timestamp"],
                "producer_action_id": producers.get(artifact_id),
                "consumer_action_id": consumer,
                "depends_on": (
                    [producers[artifact_id]] if artifact_id in producers else []
                ),
            }
        )
    return rows


def _execution_rows(
    events: list[dict[str, Any]],
    registry: ExecutorRegistry,
    initial_sizes: Mapping[str, int],
    transfers: list[dict[str, object]],
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
    starts = {
        str(event["action_id"]): event
        for event in events
        if event.get("event_type") == "worker.execution.start"
    }
    producers = {
        str(event["artifact_id"]): str(event["action_id"])
        for event in events
        if event.get("event_type") == "artifact.created"
    }
    transfer_for_input = {
        (str(item["consumer_action_id"]), str(item["artifact_id"])): str(
            item["transfer_id"]
        )
        for item in transfers
        if item.get("consumer_action_id") is not None
    }
    rows: list[dict[str, object]] = []
    for event in events:
        if event.get("event_type") != "worker.execution.end" or not event.get("success"):
            continue
        action_id = str(event["action_id"])
        request = requests[action_id]
        worker_id = str(event["worker_id"])
        inputs = cast(list[str], request.get("input_artifacts", []))
        outputs = cast(list[str], event.get("output_artifacts", []))
        dependencies: list[str] = []
        for artifact_id in inputs:
            dependency = transfer_for_input.get((action_id, artifact_id)) or producers.get(
                artifact_id
            )
            if dependency is not None and dependency not in dependencies:
                dependencies.append(dependency)
        rows.append(
            {
                "action_id": action_id,
                "operator_id": str(request.get("semantic_operator", "invoke_model")),
                "executor_id": str(event["executor"]),
                "worker_id": worker_id,
                "site_id": registry.worker_sites()[worker_id],
                "service_ms": float(event.get("service_ms", 0.0)),
                "input_bytes": sum(sizes.get(item, 0) for item in inputs),
                "output_bytes": sum(sizes.get(item, 0) for item in outputs),
                "input_tokens": int(event.get("input_tokens", 0)),
                "output_tokens": int(event.get("output_tokens", 0)),
                "input_artifacts": inputs,
                "output_artifacts": outputs,
                "started_at": starts[action_id]["timestamp"],
                "finished_at": event["timestamp"],
                "depends_on": dependencies,
                "model_id": request.get("model_id"),
            }
        )
    return rows


def _timestamp_ms(value: object) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp() * 1000


def _interval_union_ms(rows: list[dict[str, object]]) -> float:
    intervals = sorted(
        (
            _timestamp_ms(item["started_at"]),
            _timestamp_ms(item["finished_at"]),
        )
        for item in rows
        if item.get("started_at") is not None and item.get("finished_at") is not None
    )
    if not intervals:
        return 0.0
    total = 0.0
    start, end = intervals[0]
    for next_start, next_end in intervals[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            total += end - start
            start, end = next_start, next_end
    return total + end - start


def _lineage_rows(
    executions: list[dict[str, object]],
    initial_workers: Mapping[str, str],
    initial_sizes: Mapping[str, int],
) -> list[dict[str, object]]:
    producer_worker = dict(initial_workers)
    artifact_sizes = dict(initial_sizes)
    for execution in executions:
        worker_id = str(execution["worker_id"])
        outputs = cast(list[str], execution["output_artifacts"])
        for artifact_id in outputs:
            producer_worker[artifact_id] = worker_id
            if len(outputs) == 1:
                artifact_sizes[artifact_id] = int(
                    cast(int | str, execution["output_bytes"])
                )
    lineage: list[dict[str, object]] = []
    for execution in executions:
        outputs = cast(list[str], execution["output_artifacts"])
        derived = outputs[0] if outputs else ""
        for input_artifact in cast(list[str], execution["input_artifacts"]):
            lineage.append(
                {
                    "producer_agent": producer_worker.get(input_artifact, "unknown"),
                    "consumer_agent": execution["worker_id"],
                    "input_artifact": input_artifact,
                    "derived_artifact": derived,
                    "artifact_bytes": artifact_sizes[input_artifact],
                    "operator": execution["operator_id"],
                }
            )
    return lineage


def _realized_trace(
    *,
    run_id: str,
    task_id: str,
    workflow_id: ScopeWorkflow,
    warmup: bool,
    executions: list[dict[str, object]],
    transfers: list[dict[str, object]],
    lineage: list[dict[str, object]],
    initial_workers: Mapping[str, str],
    e2e_latency_ms: float,
) -> dict[str, object]:
    retrieval_rows = [
        item for item in executions if item["operator_id"] == "bm25_retrieve"
    ]
    transfer_sum_ms = sum(
        float(cast(float | int | str, item["latency_ms"])) for item in transfers
    )
    retrieval_sum_ms = sum(
        float(cast(float | int | str, item["service_ms"]))
        for item in retrieval_rows
    )
    transfer_critical_ms = min(_interval_union_ms(transfers), transfer_sum_ms)
    retrieval_critical_ms = min(
        _interval_union_ms(retrieval_rows), retrieval_sum_ms
    )
    final_model_ms = sum(
        float(cast(float | int, item["service_ms"]))
        for item in executions
        if item["operator_id"] == "invoke_model"
    )
    action_spans = [
        {
            "span_id": str(item["action_id"]),
            "span_kind": "action",
            "name": str(item["operator_id"]),
            "started_at": item["started_at"],
            "finished_at": item["finished_at"],
            "duration_ms": item["service_ms"],
            "depends_on": item["depends_on"],
            "service_scope": "local",
            "operator_id": item["operator_id"],
            "model_id": item["model_id"],
            "executor_id": item["executor_id"],
            "site_id": item["site_id"],
            "input_artifacts": item["input_artifacts"],
            "output_artifacts": item["output_artifacts"],
            "input_bytes": item["input_bytes"],
            "output_bytes": item["output_bytes"],
            "input_tokens": item["input_tokens"],
            "output_tokens": item["output_tokens"],
        }
        for item in executions
    ]
    transfer_spans = [
        {
            "span_id": str(item["transfer_id"]),
            "span_kind": "transfer",
            "name": "artifact_transfer",
            "started_at": item["started_at"],
            "finished_at": item["finished_at"],
            "duration_ms": item["latency_ms"],
            "depends_on": item["depends_on"],
            "service_scope": "other",
            "artifact_id": item["artifact_id"],
            "producer_span_id": item["producer_action_id"],
            "consumer_span_id": item["consumer_action_id"],
            "src_site": item["src_site"],
            "dst_site": item["dst_site"],
            "bytes": item["bytes"],
        }
        for item in transfers
    ]
    return {
        "schema_version": "realized-workflow-trace-v1",
        "run_id": run_id,
        "task_id": task_id,
        "workflow_id": workflow_id,
        "warmup": warmup,
        "dependency_evidence": "complete",
        "timestamp_evidence": "complete",
        "trace_coverage": "complete",
        "spans": [*action_spans, *transfer_spans],
        "e2e_latency_ms": e2e_latency_ms,
        "quality": None,
        "metadata": {
            "source": "infra-aware-mas-runtime-trace",
            "initial_artifact_workers": dict(initial_workers),
            "lineage_record_count": len(lineage),
            "critical_path_summary": {
                "retrieval_critical_ms": retrieval_critical_ms,
                "transfer_critical_ms": transfer_critical_ms,
                "final_model_ms": final_model_ms,
                "accounted_phase_ms": (
                    retrieval_critical_ms + transfer_critical_ms + final_model_ms
                ),
                "method": "measured_phase_interval_union_capped_by_work_sum",
            },
        },
    }


def _append_jsonl(path: Path, row: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="") as output:
        output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


async def run_scope_expansion_v0_sweep(
    sweep_path: Path,
    output_directory: Path,
) -> dict[str, object]:
    """Run 1 warm-up + 3 measured trials for every MultiHop-RAG reference cell."""
    sweep_path = sweep_path.resolve()
    config = ScopeExpansionSweepConfig.from_yaml(sweep_path)
    config_base = sweep_path.parent
    models_path = (
        config.models_config
        if config.models_config.is_absolute()
        else (config_base / config.models_config).resolve()
    )
    executors_path = (
        config.executors_config
        if config.executors_config.is_absolute()
        else (config_base / config.executors_config).resolve()
    )
    models = ModelRegistry.from_yaml(models_path)
    registry = ExecutorRegistry.from_yaml(executors_path)
    _validate_deployment(config, registry, models)
    final_executor = registry.get(config.final_executor_id)
    scheduler = FixedScheduler(registry, {config.final_model_id: final_executor.id})
    output_directory = output_directory.resolve()
    raw_path = output_directory / "raw_runs.jsonl"
    traces_root = output_directory / "traces"
    temporary_root = output_directory / ".runtime"
    completed = _completed_keys(raw_path, config.run_prefix)
    clients = create_worker_clients(registry, config.worker_timeout_seconds)
    written = 0
    try:
        await _preflight_jetsons(registry, clients, config.final_executor_id)
        for task in config.tasks:
            paths = [item.path for item in task.inputs]
            worker_ids = [item.worker_id for item in task.inputs]
            raw_tokens = sum(item.expected_tokens for item in task.inputs)
            documents_per_agent = {
                site_id: sum(item.worker_id == worker_id for item in task.inputs)
                for worker_id, site_id in zip(SHARD_WORKERS, SHARD_SITES, strict=True)
            }
            for world in config.worlds:
                for repeat, warmup in _attempt_schedule(
                    config.repeats, config.warmup_runs
                ):
                    for workflow_id in config.workflows:
                        key = (task.task_id, workflow_id, world.world_id, repeat, warmup)
                        if key in completed:
                            continue
                        run_id = _physical_run_id(
                            task.task_id,
                            world.world_id,
                            workflow_id,
                            repeat,
                            warmup,
                        )
                        trace = TraceRecorder(traces_root, run_id, exclusive=True)
                        await trace.start(
                            {
                                "mode": "scope_expansion_v0_multihop_reference",
                                "planner_constructed": False,
                                "measurement_series_id": config.run_prefix,
                                "measurement_protocol": MEASUREMENT_PROTOCOL,
                                "task_id": task.task_id,
                                "workflow_id": workflow_id,
                                "world_id": world.world_id,
                                "repeat": repeat,
                                "warmup": warmup,
                                "candidate_top_n": config.candidate_top_n,
                                "local_top_k_per_shard": config.local_top_k_per_shard,
                                "retrieval_algorithm": "bm25",
                                "configured_bandwidth_mbps": world.bandwidth_mbps,
                                "configured_added_rtt_ms": world.rtt_ms,
                                "input_source": config.input_source,
                            }
                        )
                        started_at = perf_counter()
                        try:
                            artifacts = await bind_placed_inputs(
                                paths,
                                run_id,
                                worker_ids,
                                clients,
                                artifact_ids=[item.artifact_id for item in task.inputs],
                                expected_size_bytes=[
                                    item.expected_size_bytes for item in task.inputs
                                ],
                                trace=trace,
                            )
                            if any(
                                not (
                                    artifact.artifact_type.startswith("text/")
                                    or artifact.artifact_type == "application/json"
                                )
                                for artifact in artifacts
                            ):
                                raise ValueError("MultiHop-RAG inputs must be UTF-8 text/JSON")
                            build_multidoc_task_interaction(task, artifacts)
                            started_at = perf_counter()
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
                                request_id_factory=(
                                    lambda: f"{run_id}/request-{uuid4().hex}"
                                ),
                                model_registry=models,
                            )
                            result = await execute_multidoc_reference(
                                runtime,
                                workflow_id=workflow_id,
                                query=task.query,
                                documents=artifacts,
                                document_workers=worker_ids,
                                raw_tokens=raw_tokens,
                                final_model_id=config.final_model_id,
                                local_top_k_per_shard=config.local_top_k_per_shard,
                            )
                            e2e_ms = (perf_counter() - started_at) * 1000
                            answer = await _download_answer(
                                result.final_artifact, clients, temporary_root
                            )
                            await trace.end(
                                {
                                    "success": True,
                                    "answer": answer,
                                    "e2e_latency_ms": e2e_ms,
                                }
                            )
                            events = _read_trace(trace.path)
                            transfers = _transfer_rows(events, registry)
                            initial_sizes = {
                                artifact.id: artifact.size_bytes for artifact in artifacts
                            }
                            executions = _execution_rows(
                                events, registry, initial_sizes, transfers
                            )
                            initial_workers = {
                                artifact.id: worker_id
                                for artifact, worker_id in zip(
                                    artifacts, worker_ids, strict=True
                                )
                            }
                            lineage = _lineage_rows(
                                executions, initial_workers, initial_sizes
                            )
                            realized_trace = _realized_trace(
                                run_id=run_id,
                                task_id=task.task_id,
                                workflow_id=workflow_id,
                                warmup=warmup,
                                executions=executions,
                                transfers=transfers,
                                lineage=lineage,
                                initial_workers=initial_workers,
                                e2e_latency_ms=e2e_ms,
                            )
                            trace_metadata = cast(
                                dict[str, object], realized_trace["metadata"]
                            )
                            critical_path = cast(
                                dict[str, object],
                                trace_metadata["critical_path_summary"],
                            )
                            transfer_critical_ms = float(
                                cast(float | int | str, critical_path["transfer_critical_ms"])
                            )
                            retrieval_critical_ms = float(
                                cast(float | int | str, critical_path["retrieval_critical_ms"])
                            )
                            artifact_records = [
                                {
                                    "artifact_id": artifact.id,
                                    "document_id": input_item.document_id,
                                    "kind": "raw_document",
                                    "site_id": registry.worker_sites()[input_item.worker_id],
                                    "bytes": artifact.size_bytes,
                                    "tokens": input_item.expected_tokens,
                                }
                                for input_item, artifact in zip(
                                    task.inputs, artifacts, strict=True
                                )
                            ]
                            reduced_records: list[dict[str, object]] = []
                            for execution in result.executions:
                                if (
                                    execution.metadata.get("semantic_operator")
                                    != "bm25_retrieve"
                                ):
                                    continue
                                for artifact in execution.output_artifacts:
                                    reduced_records.append(
                                        {
                                            "artifact_id": artifact.id,
                                            "kind": "retrieved_evidence",
                                            "site_id": registry.worker_sites()[
                                                artifact.locations[0]
                                            ],
                                            "bytes": artifact.size_bytes,
                                            "tokens": int(
                                                execution.metadata.get(
                                                    "retrieved_tokens", 0
                                                )
                                            ),
                                        }
                                    )
                            row: dict[str, object] = {
                                "schema_version": "scope-expansion-v0-run-v1",
                                "run_id": run_id,
                                "measurement_series_id": config.run_prefix,
                                "protocol_id": MEASUREMENT_PROTOCOL,
                                "task_id": task.task_id,
                                "task_family": "multi_document_qa",
                                "workflow_id": workflow_id,
                                "world_id": world.world_id,
                                "repeat": repeat,
                                "warmup": warmup,
                                "status": "completed",
                                "quality": None,
                                "demand": {
                                    "artifact_count": len(artifacts),
                                    "raw_bytes": result.raw_bytes,
                                    "raw_tokens": result.raw_tokens,
                                    "documents_per_agent": documents_per_agent,
                                },
                                "retrieval": {
                                    "algorithm": "bm25",
                                    "top_k_per_shard": config.local_top_k_per_shard,
                                    "calls": result.retrieval_calls,
                                    "sum_ms": result.retrieval_sum_ms,
                                    "critical_ms": retrieval_critical_ms,
                                    "retrieved_document_count": (
                                        result.retrieved_document_count
                                    ),
                                },
                                "artifacts": {
                                    "raw_bytes": result.raw_bytes,
                                    "reduced_bytes": result.reduced_bytes,
                                    "reduced_tokens": result.reduced_tokens,
                                    "absolute_reducible_bytes": (
                                        result.raw_bytes - result.reduced_bytes
                                    ),
                                    "records": [*artifact_records, *reduced_records],
                                },
                                "transfers": {
                                    "count": len(transfers),
                                    "bytes": sum(
                                        int(cast(int | str, item["bytes"]))
                                        for item in transfers
                                    ),
                                    "latency_ms": sum(
                                        float(cast(float | int, item["latency_ms"]))
                                        for item in transfers
                                    ),
                                    "critical_ms": transfer_critical_ms,
                                    "records": transfers,
                                },
                                "service": {
                                    "local_preprocessing_ms": result.retrieval_sum_ms,
                                    "final_model_ms": result.final_model_ms,
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
                                "lineage": lineage,
                                "final_answer": answer,
                                "trace": realized_trace,
                                "metadata": {
                                    "planner_constructed": False,
                                    "candidate_top_n": config.candidate_top_n,
                                    "local_top_k_per_shard": (
                                        config.local_top_k_per_shard
                                    ),
                                    "retrieval_algorithm": "bm25",
                                    "configured_bandwidth_mbps": world.bandwidth_mbps,
                                    "configured_added_rtt_ms": world.rtt_ms,
                                    "network_shaping": (
                                        "worker_application_layer_wall_clock"
                                    ),
                                    "final_model_id": config.final_model_id,
                                    "final_executor_id": config.final_executor_id,
                                    "e2e_boundary": (
                                        "workflow_start_to_final_artifact_created"
                                    ),
                                    "answer_download_in_e2e": False,
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
                                "schema_version": "scope-expansion-v0-run-v1",
                                "run_id": run_id,
                                "measurement_series_id": config.run_prefix,
                                "protocol_id": MEASUREMENT_PROTOCOL,
                                "task_id": task.task_id,
                                "task_family": "multi_document_qa",
                                "workflow_id": workflow_id,
                                "world_id": world.world_id,
                                "repeat": repeat,
                                "warmup": warmup,
                                "status": "failed",
                                "quality": None,
                                "error": f"{type(error).__name__}: {error}",
                                "metadata": {
                                    "planner_constructed": False,
                                    "candidate_top_n": config.candidate_top_n,
                                    "local_top_k_per_shard": (
                                        config.local_top_k_per_shard
                                    ),
                                    "retrieval_algorithm": "bm25",
                                    "configured_bandwidth_mbps": world.bandwidth_mbps,
                                    "configured_added_rtt_ms": world.rtt_ms,
                                    "network_shaping": (
                                        "worker_application_layer_wall_clock"
                                    ),
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
        "completed_cells": len(completed),
    }
