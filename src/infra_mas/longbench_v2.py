"""Planner-free LongBench-v2 references for Scope Expansion v0.

The execution surface is intentionally dataset-neutral.  Dataset adapters materialize
natural document/record boundaries and an answer-independent structured plan; this
module only executes generic retrieval, record processing, and final reasoning.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from time import perf_counter
from typing import Annotated, Any, Literal, Protocol, cast
from uuid import uuid4

import yaml
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from infra_mas.core.artifact import ArtifactRef
from infra_mas.core.execution import (
    ExecutionResult,
    InvocationSpec,
    RecordAggregation,
    RecordDerivation,
    RecordPredicate,
    RecordSort,
)
from infra_mas.execution.executor_registry import ExecutorRegistry
from infra_mas.execution.manager import ExecutionManager
from infra_mas.execution.transfer import TransferManager, TransferProfile
from infra_mas.experiment import bind_placed_inputs, close_worker_clients, create_worker_clients
from infra_mas.operators import (
    GENERAL_OPERATOR_REGISTRY,
    ExternalEvaluatorSpec,
    InitialArtifactSpec,
    ObservationSpec,
    OperatorId,
    RuntimeVerifierSpec,
    TaskInteractionSpec,
)
from infra_mas.runtime.model_registry import ModelRegistry
from infra_mas.runtime.runtime import AgentRuntime
from infra_mas.scheduler.fixed import FixedScheduler
from infra_mas.scope_expansion_v0 import (
    MEASUREMENT_PROTOCOL,
    SHARD_SITES,
    SHARD_WORKERS,
    ScopeNetworkWorld,
    _append_jsonl,  # pyright: ignore[reportPrivateUsage]
    _attempt_schedule,  # pyright: ignore[reportPrivateUsage]
    _download_answer,  # pyright: ignore[reportPrivateUsage]
    _execution_rows,  # pyright: ignore[reportPrivateUsage]
    _lineage_rows,  # pyright: ignore[reportPrivateUsage]
    _read_trace,  # pyright: ignore[reportPrivateUsage]
    _realized_trace,  # pyright: ignore[reportPrivateUsage]
    _transfer_rows,  # pyright: ignore[reportPrivateUsage]
    stable_document_worker,
)
from infra_mas.tracing.recorder import TraceRecorder

NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
LongBenchTaskFamily = Literal[
    "multi_document_qa",
    "structured_data_analysis",
]
LongBenchWorkflow = Literal[
    "centralized_raw",
    "distributed_retrieval",
    "distributed_compute",
]

MULTIDOC_WORKFLOWS = ("centralized_raw", "distributed_retrieval")
STRUCTURED_WORKFLOWS = ("centralized_raw", "distributed_compute")


class LongBenchRuntime(Protocol):
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

    async def filter_records_on_worker(
        self,
        artifacts: list[ArtifactRef],
        target_worker_id: str,
        *,
        predicates: list[RecordPredicate],
        match: Literal["all", "any"] = "all",
        parent_action_id: str | None = None,
    ) -> ExecutionResult: ...

    async def select_fields_on_worker(
        self,
        artifacts: list[ArtifactRef],
        target_worker_id: str,
        *,
        fields: list[str],
        parent_action_id: str | None = None,
    ) -> ExecutionResult: ...

    async def aggregate_records_on_worker(
        self,
        artifacts: list[ArtifactRef],
        target_worker_id: str,
        *,
        aggregations: list[RecordAggregation],
        group_by: list[str] | None = None,
        parent_action_id: str | None = None,
    ) -> ExecutionResult: ...

    async def derive_fields_on_worker(
        self,
        artifacts: list[ArtifactRef],
        target_worker_id: str,
        *,
        derivations: list[RecordDerivation],
        parent_action_id: str | None = None,
    ) -> ExecutionResult: ...

    async def top_k_records_on_worker(
        self,
        artifacts: list[ArtifactRef],
        target_worker_id: str,
        *,
        order_by: list[RecordSort],
        limit: int,
        parent_action_id: str | None = None,
    ) -> ExecutionResult: ...


class LongBenchInput(BaseModel):
    """One natural document or record shard, never a synthetic token split."""

    model_config = ConfigDict(extra="forbid")

    source_id: NonEmptyString
    artifact_id: NonEmptyString
    artifact_type: NonEmptyString
    path: Path
    worker_id: Literal["a4", "a5", "a28"]
    expected_size_bytes: int = Field(gt=0)
    expected_tokens: int = Field(ge=0)


class StructuredPlan(BaseModel):
    """Answer-independent program for generic deterministic record operators."""

    model_config = ConfigDict(extra="forbid")

    predicates: list[RecordPredicate] = Field(default_factory=lambda: list[RecordPredicate]())
    match: Literal["all", "any"] = "all"
    select_fields: list[NonEmptyString] = Field(default_factory=list)
    aggregations: list[RecordAggregation] = Field(default_factory=lambda: list[RecordAggregation]())
    group_by: list[NonEmptyString] = Field(default_factory=list)
    derivations: list[RecordDerivation] = Field(default_factory=lambda: list[RecordDerivation]())
    order_by: list[RecordSort] = Field(default_factory=lambda: list[RecordSort]())
    limit: int | None = Field(default=None, ge=1, le=1000)

    @model_validator(mode="after")
    def validate_projection(self) -> StructuredPlan:
        if len(self.select_fields) != len(set(self.select_fields)):
            raise ValueError("select_fields must be unique")
        if len(self.group_by) != len(set(self.group_by)):
            raise ValueError("group_by must be unique")
        required = set(self.group_by)
        required.update(item.field for item in self.aggregations if item.field is not None)
        required.update(item.left_field for item in self.derivations)
        required.update(item.right_field for item in self.derivations)
        derived_outputs = {item.output_field for item in self.derivations}
        premature = set(self.select_fields) & derived_outputs
        if premature:
            raise ValueError(
                "select_fields cannot reference fields produced by a later derive step: "
                f"{sorted(premature)}"
            )
        if self.select_fields and not required.issubset(self.select_fields):
            raise ValueError(
                "select_fields must retain every group_by, aggregation, and derivation field"
            )
        produced = set(derived_outputs)
        produced.update(item.output_field for item in self.aggregations)
        available_sort = set(self.select_fields) | produced
        if self.order_by and any(item.field not in available_sort for item in self.order_by):
            raise ValueError("order_by fields must be selected source fields or derivations")
        if bool(self.order_by) != (self.limit is not None):
            raise ValueError("order_by and limit must be configured together")
        if not self.aggregations and not self.order_by:
            raise ValueError("structured_plan must aggregate or select deterministic top-k")
        return self


class LongBenchTask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: NonEmptyString
    task_family: LongBenchTaskFamily
    query: NonEmptyString
    answer_choices: dict[Literal["A", "B", "C", "D"], NonEmptyString]
    evaluator_id: NonEmptyString
    inputs: Annotated[list[LongBenchInput], Field(min_length=1, max_length=128)]
    local_top_k_per_shard: Annotated[int, Field(ge=1, le=50)] | None = None
    structured_plan: StructuredPlan | None = None

    @model_validator(mode="after")
    def validate_family_contract(self) -> LongBenchTask:
        artifact_ids = [item.artifact_id for item in self.inputs]
        source_ids = [item.source_id for item in self.inputs]
        if len(artifact_ids) != len(set(artifact_ids)):
            raise ValueError("artifact IDs must be unique within a task")
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("source IDs must be unique within a task")
        if set(self.answer_choices) != {"A", "B", "C", "D"}:
            raise ValueError("LongBench multiple-choice tasks require choices A, B, C, and D")
        if self.task_family == "multi_document_qa":
            if self.structured_plan is not None:
                raise ValueError("multi-document tasks cannot define structured_plan")
            if self.local_top_k_per_shard is None:
                raise ValueError("multi-document tasks require local_top_k_per_shard")
            if len(set(item.worker_id for item in self.inputs)) < 2:
                raise ValueError("multi-document inputs must be distributed across workers")
            for item in self.inputs:
                if item.worker_id != stable_document_worker(item.source_id):
                    raise ValueError(f"document {item.source_id!r} has non-deterministic placement")
        else:
            if self.structured_plan is None:
                raise ValueError("structured tasks require structured_plan")
            if self.local_top_k_per_shard is not None:
                raise ValueError("structured tasks cannot configure BM25 top-k")
            if set(item.worker_id for item in self.inputs) != set(SHARD_WORKERS):
                raise ValueError("structured record shards must span all three Jetsons")
        return self

    @property
    def workflows(self) -> tuple[LongBenchWorkflow, LongBenchWorkflow]:
        if self.task_family == "multi_document_qa":
            return MULTIDOC_WORKFLOWS
        return STRUCTURED_WORKFLOWS


class LongBenchSweepConfig(BaseModel):
    """One multi-document and one structured LongBench-v2 controlled sweep."""

    model_config = ConfigDict(extra="forbid")

    models_config: Path
    executors_config: Path
    final_model_id: NonEmptyString
    final_executor_id: NonEmptyString
    input_source: Literal["worker_local"] = "worker_local"
    tasks: Annotated[list[LongBenchTask], Field(min_length=2, max_length=2)]
    worlds: Annotated[list[ScopeNetworkWorld], Field(min_length=2, max_length=2)]
    warmup_runs: Literal[1] = 1
    repeats: Literal[3] = 3
    worker_timeout_seconds: float = Field(default=1800, gt=0)
    run_prefix: NonEmptyString = "scope-expansion-v0-longbench-v2"

    @model_validator(mode="after")
    def validate_controlled_sweep(self) -> LongBenchSweepConfig:
        if {task.task_family for task in self.tasks} != {
            "multi_document_qa",
            "structured_data_analysis",
        }:
            raise ValueError("exactly one LongBench task from each task family is required")
        if len({task.task_id for task in self.tasks}) != len(self.tasks):
            raise ValueError("task IDs must be unique")
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
        return self

    @classmethod
    def from_yaml(cls, path: Path) -> LongBenchSweepConfig:
        raw: object = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls.model_validate(raw)


class LongBenchReferenceResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    workflow_id: LongBenchWorkflow
    executions: list[ExecutionResult]
    final_artifact: ArtifactRef
    raw_bytes: int = Field(ge=0)
    raw_tokens: int = Field(ge=0)
    reduced_bytes: int = Field(ge=0)
    reduced_tokens: int = Field(ge=0)
    local_calls: int = Field(ge=0)
    retrieved_document_count: int = Field(ge=0)
    local_sum_ms: float = Field(ge=0)
    local_critical_ms: float = Field(ge=0)
    final_model_ms: float = Field(ge=0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    api_cost_usd: float = Field(ge=0)


def _visible_field(visible: Mapping[str, object], *names: str) -> object:
    for name in names:
        if name in visible:
            return visible[name]
    raise ValueError(f"planner_visible is missing one of {names!r}")


def _normalize_structured_plan(raw: Mapping[str, Any]) -> dict[str, object]:
    """Translate task-bank vocabulary into the generic runtime request vocabulary."""
    aggregations = [
        {
            "output_field": item["alias"],
            "operation": item["function"],
            "field": item.get("field"),
        }
        for item in cast(list[dict[str, Any]], raw.get("aggregations", []))
    ]
    derivations: list[dict[str, object]] = []
    for item in cast(list[dict[str, Any]], raw.get("derived_fields", [])):
        operands = cast(list[str], item["operands"])
        if len(operands) != 2:
            raise ValueError("each derived field requires exactly two named operands")
        derivations.append(
            {
                "output_field": item["alias"],
                "operation": item["operator"],
                "left_field": operands[0],
                "right_field": operands[1],
            }
        )
    direction_aliases = {
        "asc": "ascending",
        "ascending": "ascending",
        "desc": "descending",
        "descending": "descending",
    }
    order_by: list[dict[str, object]] = []
    for item in cast(list[dict[str, Any]], raw.get("order_by", [])):
        direction = str(item.get("direction", "descending"))
        if direction not in direction_aliases:
            raise ValueError(f"unsupported record sort direction: {direction!r}")
        order_by.append(
            {
                "field": item["field"],
                "direction": direction_aliases[direction],
                "nulls": item.get("nulls", "last"),
            }
        )
    return {
        "predicates": raw.get("predicates", []),
        "match": raw.get("match", "all"),
        "select_fields": raw.get("select_fields", []),
        "aggregations": aggregations,
        "group_by": raw.get("group_by", []),
        "derivations": derivations,
        "order_by": order_by,
        "limit": raw.get("limit"),
    }


def generate_longbench_v2_sweep(
    task_bank_path: Path,
    output_path: Path,
    *,
    remote_root: Path = Path("/home/edge/xiaoming/scope_expansion_v0"),
) -> LongBenchSweepConfig:
    """Generate a gold-free sweep from the two LongBench task-bank entries."""
    records = [
        cast(dict[str, Any], json.loads(line))
        for line in task_bank_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    root = PurePosixPath(remote_root.as_posix())
    tasks: list[dict[str, object]] = []
    site_to_worker = {"A4": "a4", "A5": "a5", "A28": "a28"}
    family_dir = {
        "multi_document_qa": "longbench_multidoc",
        "structured_data_analysis": "longbench_structured",
    }
    for record in records:
        visible = cast(dict[str, Any], record.get("planner_visible", {}))
        dataset = str(visible.get("dataset", record.get("dataset", ""))).lower()
        if "longbench" not in dataset:
            continue
        family = str(visible.get("task_family", record.get("task_family", "")))
        if family not in family_dir:
            continue
        artifact_ids = cast(list[str], visible["artifact_ids"])
        artifact_types = cast(dict[str, str], visible["artifact_types"])
        sizes = cast(dict[str, int], visible["artifact_sizes"])
        tokens = cast(dict[str, int], visible["artifact_tokens"])
        placements = cast(dict[str, str], visible["artifact_placement"])
        source_ids = cast(
            dict[str, str],
            visible.get("artifact_document_ids")
            or visible.get("artifact_source_refs")
            or {artifact_id: artifact_id for artifact_id in artifact_ids},
        )
        inputs: list[dict[str, object]] = []
        for artifact_id in artifact_ids:
            site_id = placements[artifact_id]
            inputs.append(
                {
                    "source_id": source_ids[artifact_id],
                    "artifact_id": artifact_id,
                    "artifact_type": artifact_types[artifact_id],
                    "path": str(
                        root / family_dir[family] / "artifacts" / site_id / f"{artifact_id}.json"
                    ),
                    "worker_id": site_to_worker[site_id],
                    "expected_size_bytes": sizes[artifact_id],
                    "expected_tokens": tokens[artifact_id],
                }
            )
        task: dict[str, object] = {
            "task_id": str(visible["task_id"]),
            "task_family": family,
            "query": str(_visible_field(visible, "query", "instruction")),
            "answer_choices": visible["answer_choices"],
            "evaluator_id": str(visible["evaluator_type"]),
            "inputs": inputs,
        }
        if family == "multi_document_qa":
            retrieval = cast(dict[str, Any], visible.get("retrieval_config", {}))
            parameters = cast(dict[str, Any], retrieval.get("parameters", {}))
            task["local_top_k_per_shard"] = int(parameters.get("runtime_local_top_k", 3))
        else:
            task["structured_plan"] = _normalize_structured_plan(
                cast(dict[str, Any], visible["structured_plan"])
            )
        tasks.append(task)
    payload: dict[str, object] = {
        "models_config": "models.yaml",
        "executors_config": "executors.yaml",
        "final_model_id": "edge-text-reasoner",
        "final_executor_id": "a28-vlm",
        "input_source": "worker_local",
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
        "warmup_runs": 1,
        "repeats": 3,
        "worker_timeout_seconds": 1800,
        "run_prefix": "scope-expansion-v0-longbench-v2",
    }
    config = LongBenchSweepConfig.model_validate(payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    return config


def build_longbench_task_interaction(
    task: LongBenchTask,
    artifacts: list[ArtifactRef],
) -> TaskInteractionSpec:
    operators: list[OperatorId] = ["bm25_retrieve", "invoke_model"]
    observations = [
        ObservationSpec(
            observation_id="retrieved_evidence",
            produced_by=["bm25_retrieve"],
            description="Deterministic shard-local evidence selected without gold metadata.",
        )
    ]
    if task.task_family == "structured_data_analysis":
        operators = [
            "filter_records",
            "select_fields",
            "derive_fields",
            "aggregate_records",
            "top_k_records",
            "invoke_model",
        ]
        observations = [
            ObservationSpec(
                observation_id="partial_aggregates",
                produced_by=["aggregate_records"],
                description="Compact deterministic partial aggregates from local shards.",
            )
        ]
        if cast(StructuredPlan, task.structured_plan).order_by:
            observations[0] = ObservationSpec(
                observation_id="partial_top_k",
                produced_by=["top_k_records"],
                description="Compact deterministic local top-k records from each shard.",
            )
    observations.append(
        ObservationSpec(
            observation_id="final_answer",
            produced_by=["invoke_model"],
            description="Answer produced by the common A28 final reasoner.",
        )
    )
    interaction = TaskInteractionSpec(
        task_id=task.task_id,
        objective=task.query,
        initial_artifacts=[
            InitialArtifactSpec(
                artifact_id=artifact.id,
                kind=(
                    "natural_document"
                    if task.task_family == "multi_document_qa"
                    else "natural_record_shard"
                ),
                source_ref=item.source_id,
            )
            for item, artifact in zip(task.inputs, artifacts, strict=True)
        ],
        operators=operators,
        observations=observations,
        runtime_verifier=RuntimeVerifierSpec(level="none"),
        external_evaluator=ExternalEvaluatorSpec(evaluator_id=task.evaluator_id),
    )
    GENERAL_OPERATOR_REGISTRY.validate_task(interaction)
    return interaction


def _structured_final_task(query: str) -> str:
    return (
        f"Question: {query}\n"
        "Use only the supplied records or partial aggregate results. Resolve the answer "
        'choice. Format your response as follows: "The correct answer is '
        '(insert answer here)".'
    )


def _question_with_choices(task: LongBenchTask) -> str:
    choices = "\n".join(f"{label}. {task.answer_choices[cast(Any, label)]}" for label in "ABCD")
    return f"{task.query}\n\nChoices:\n{choices}"


async def execute_longbench_multidoc_reference(
    runtime: LongBenchRuntime,
    *,
    workflow_id: Literal["centralized_raw", "distributed_retrieval"],
    query: str,
    documents: list[ArtifactRef],
    document_workers: list[str],
    raw_tokens: int,
    final_model_id: str,
    local_top_k_per_shard: int,
) -> LongBenchReferenceResult:
    """Run natural-document QA without manufacturing an empty third shard."""
    raw_bytes = sum(item.size_bytes for item in documents)
    retrievals: list[ExecutionResult] = []
    if workflow_id == "centralized_raw":
        final_inputs = documents
        reduced_bytes = raw_bytes
        reduced_tokens = raw_tokens
    elif workflow_id == "distributed_retrieval":
        shards: dict[str, list[ArtifactRef]] = {}
        for artifact, worker_id in zip(documents, document_workers, strict=True):
            shards.setdefault(worker_id, []).append(artifact)
        active_workers = [worker for worker in SHARD_WORKERS if worker in shards]
        retrievals = list(
            await asyncio.gather(
                *(
                    runtime.bm25_retrieve_on_worker(
                        query,
                        shards[worker_id],
                        worker_id,
                        top_k=local_top_k_per_shard,
                    )
                    for worker_id in active_workers
                )
            )
        )
        final_inputs = [item.output_artifacts[0] for item in retrievals]
        reduced_bytes = sum(item.size_bytes for item in final_inputs)
        reduced_tokens = sum(int(item.metadata.get("retrieved_tokens", 0)) for item in retrievals)
    else:
        raise ValueError(f"unsupported multi-document workflow: {workflow_id!r}")

    final = await runtime.invoke(
        InvocationSpec(
            model_id=final_model_id,
            role="scope-expansion-v0-final-reasoner",
            instructions=(
                "You are the common final reasoning stage for an extra-long "
                "multi-document QA experiment. Use only the supplied context and "
                'format your response as follows: "The correct answer is '
                '(insert answer here)".'
            ),
            task=f"Question and choices:\n{query}",
            input_artifacts=final_inputs,
        )
    )
    executions = [*retrievals, final]
    return LongBenchReferenceResult(
        workflow_id=workflow_id,
        executions=executions,
        final_artifact=final.output_artifacts[0],
        raw_bytes=raw_bytes,
        raw_tokens=raw_tokens,
        reduced_bytes=reduced_bytes,
        reduced_tokens=reduced_tokens,
        local_calls=len(retrievals),
        retrieved_document_count=sum(
            int(item.metadata.get("retrieved_document_count", 0)) for item in retrievals
        ),
        local_sum_ms=sum(item.service_ms for item in retrievals),
        local_critical_ms=max((item.service_ms for item in retrievals), default=0.0),
        final_model_ms=final.service_ms,
        input_tokens=int(final.metadata.get("input_tokens", 0)),
        output_tokens=int(final.metadata.get("output_tokens", 0)),
        api_cost_usd=float(final.metadata.get("api_cost_usd", 0.0)),
    )


async def execute_structured_reference(
    runtime: LongBenchRuntime,
    *,
    workflow_id: Literal["centralized_raw", "distributed_compute"],
    query: str,
    shards: list[ArtifactRef],
    shard_workers: list[str],
    raw_tokens: int,
    structured_plan: StructuredPlan,
    final_model_id: str,
) -> LongBenchReferenceResult:
    """Execute raw centralization or parallel deterministic local record compute."""
    raw_bytes = sum(item.size_bytes for item in shards)
    local_chains: list[list[ExecutionResult]] = []
    if workflow_id == "centralized_raw":
        final_inputs = shards
        reduced_bytes = raw_bytes
        reduced_tokens = raw_tokens
    elif workflow_id == "distributed_compute":
        by_worker: dict[str, list[ArtifactRef]] = {worker: [] for worker in SHARD_WORKERS}
        for artifact, worker_id in zip(shards, shard_workers, strict=True):
            by_worker[worker_id].append(artifact)

        async def process(worker_id: str) -> list[ExecutionResult]:
            current = by_worker[worker_id]
            chain: list[ExecutionResult] = []
            if structured_plan.predicates:
                filtered = await runtime.filter_records_on_worker(
                    current,
                    worker_id,
                    predicates=structured_plan.predicates,
                    match=structured_plan.match,
                )
                chain.append(filtered)
                current = filtered.output_artifacts
            if structured_plan.select_fields:
                selected = await runtime.select_fields_on_worker(
                    current,
                    worker_id,
                    fields=structured_plan.select_fields,
                )
                chain.append(selected)
                current = selected.output_artifacts
            if structured_plan.derivations:
                derived = await runtime.derive_fields_on_worker(
                    current,
                    worker_id,
                    derivations=structured_plan.derivations,
                )
                chain.append(derived)
                current = derived.output_artifacts
            if structured_plan.aggregations:
                aggregated = await runtime.aggregate_records_on_worker(
                    current,
                    worker_id,
                    aggregations=structured_plan.aggregations,
                    group_by=structured_plan.group_by or None,
                )
                chain.append(aggregated)
                current = aggregated.output_artifacts
            if structured_plan.order_by:
                top_k = await runtime.top_k_records_on_worker(
                    current,
                    worker_id,
                    order_by=structured_plan.order_by,
                    limit=cast(int, structured_plan.limit),
                )
                chain.append(top_k)
            return chain

        local_chains = list(
            await asyncio.gather(*(process(worker_id) for worker_id in SHARD_WORKERS))
        )
        final_inputs = [chain[-1].output_artifacts[0] for chain in local_chains]
        reduced_bytes = sum(item.size_bytes for item in final_inputs)
        reduced_tokens = sum(
            int(chain[-1].metadata.get("output_tokens", 0)) for chain in local_chains
        )
    else:
        raise ValueError(f"unsupported structured workflow: {workflow_id!r}")

    final = await runtime.invoke(
        InvocationSpec(
            model_id=final_model_id,
            role="scope-expansion-v0-final-reasoner",
            instructions=(
                "You are the common final reasoning stage for a structured-data QA "
                "experiment. Do not use external knowledge; follow the output format."
            ),
            task=_structured_final_task(query),
            input_artifacts=final_inputs,
        )
    )
    local_results = [item for chain in local_chains for item in chain]
    executions = [*local_results, final]
    return LongBenchReferenceResult(
        workflow_id=workflow_id,
        executions=executions,
        final_artifact=final.output_artifacts[0],
        raw_bytes=raw_bytes,
        raw_tokens=raw_tokens,
        reduced_bytes=reduced_bytes,
        reduced_tokens=reduced_tokens,
        local_calls=len(local_results),
        retrieved_document_count=0,
        local_sum_ms=sum(item.service_ms for item in local_results),
        local_critical_ms=max(
            (sum(item.service_ms for item in chain) for chain in local_chains),
            default=0.0,
        ),
        final_model_ms=final.service_ms,
        input_tokens=int(final.metadata.get("input_tokens", 0)),
        output_tokens=int(final.metadata.get("output_tokens", 0)),
        api_cost_usd=float(final.metadata.get("api_cost_usd", 0.0)),
    )


def _validate_deployment(
    config: LongBenchSweepConfig,
    registry: ExecutorRegistry,
    models: ModelRegistry,
) -> None:
    sites = registry.worker_sites()
    if {worker: sites.get(worker) for worker in SHARD_WORKERS} != {
        "a4": "A4",
        "a5": "A5",
        "a28": "A28",
    }:
        raise ValueError("LongBench workers must resolve to A4, A5, and A28")
    if any(site == "4090" for site in sites.values()):
        raise ValueError("Scope Expansion v0 LongBench-v2 is Jetson-only")
    if config.final_model_id not in {item.model_id for item in models.list()}:
        raise ValueError(f"unknown final model: {config.final_model_id!r}")
    executor = registry.get(config.final_executor_id)
    if executor.worker_id != "a28" or executor.site != "A28":
        raise ValueError("final_executor_id must be hosted by A28")
    if executor.model_id != config.final_model_id:
        raise ValueError("final executor must serve final_model_id")


async def _preflight(
    config: LongBenchSweepConfig,
    registry: ExecutorRegistry,
    clients: Mapping[str, Any],
) -> None:
    if set(registry.worker_endpoints()) != set(SHARD_WORKERS):
        raise ValueError("executors_config must contain exactly the three Jetson workers")
    required = {"bm25_retrieve"}
    if any(task.task_family == "structured_data_analysis" for task in config.tasks):
        required.update(
            {
                "filter_records",
                "select_fields",
                "derive_fields",
                "aggregate_records",
                "top_k_records",
            }
        )

    async def check(worker_id: str) -> None:
        client = clients[worker_id]
        await client.health()
        await client.ready()
        status = await client.status()
        if status.worker_id != worker_id:
            raise ValueError(f"endpoint {worker_id!r} identifies as {status.worker_id!r}")
        missing = required - set(status.operators)
        if missing:
            raise ValueError(f"worker {worker_id!r} lacks operators {sorted(missing)}")
        if worker_id == "a28" and config.final_executor_id not in status.executors:
            raise ValueError("A28 does not expose the configured final executor")

    await asyncio.gather(*(check(worker_id) for worker_id in SHARD_WORKERS))


def _run_id(
    task: LongBenchTask,
    world: ScopeNetworkWorld,
    workflow: LongBenchWorkflow,
    repeat: int,
    warmup: bool,
) -> str:
    world_code = "h1" if world.world_id == "H1_distributed_constrained" else "h2"
    workflow_code = {
        "centralized_raw": "cr",
        "distributed_retrieval": "dr",
        "distributed_compute": "dc",
    }[workflow]
    attempt = "warmup" if warmup else f"r{repeat}"
    return f"lb2-{task.task_id[-8:]}-{world_code}-{workflow_code}-{attempt}-{uuid4().hex[:8]}"


def _completed_keys(path: Path, series_id: str) -> set[tuple[str, str, str, int, bool]]:
    if not path.is_file():
        return set()
    rows = (
        cast(dict[str, object], json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    return {
        (
            str(row["task_id"]),
            str(row["workflow_id"]),
            str(row["world_id"]),
            int(cast(int, row["repeat"])),
            bool(row["warmup"]),
        )
        for row in rows
        if row.get("status") == "completed" and row.get("measurement_series_id") == series_id
    }


async def run_longbench_v2_sweep(
    sweep_path: Path,
    output_directory: Path,
) -> dict[str, object]:
    """Run every LongBench-v2 cell with one warm-up and three measured trials."""
    sweep_path = sweep_path.resolve()
    config = LongBenchSweepConfig.from_yaml(sweep_path)
    base = sweep_path.parent
    models_path = (
        config.models_config
        if config.models_config.is_absolute()
        else (base / config.models_config).resolve()
    )
    executors_path = (
        config.executors_config
        if config.executors_config.is_absolute()
        else (base / config.executors_config).resolve()
    )
    models = ModelRegistry.from_yaml(models_path)
    registry = ExecutorRegistry.from_yaml(executors_path)
    _validate_deployment(config, registry, models)
    final_executor = registry.get(config.final_executor_id)
    scheduler = FixedScheduler(registry, {config.final_model_id: final_executor.id})
    output_directory = output_directory.resolve()
    family_outputs = {
        "multi_document_qa": output_directory / "longbench_multidoc",
        "structured_data_analysis": output_directory / "longbench_structured",
    }
    raw_paths = {
        family: directory / "raw_runs.jsonl" for family, directory in family_outputs.items()
    }
    completed: set[tuple[str, str, str, int, bool]] = set()
    for path in raw_paths.values():
        completed.update(_completed_keys(path, config.run_prefix))
    clients = create_worker_clients(registry, config.worker_timeout_seconds)
    written = 0
    try:
        await _preflight(config, registry, clients)
        for task in config.tasks:
            task_output = family_outputs[task.task_family]
            raw_path = raw_paths[task.task_family]
            traces_root = task_output / "traces"
            temporary_root = task_output / ".runtime"
            paths = [item.path for item in task.inputs]
            worker_ids = [item.worker_id for item in task.inputs]
            raw_tokens = sum(item.expected_tokens for item in task.inputs)
            per_agent = {
                site: sum(item.worker_id == worker for item in task.inputs)
                for worker, site in zip(SHARD_WORKERS, SHARD_SITES, strict=True)
            }
            for world in config.worlds:
                for repeat, warmup in _attempt_schedule(config.repeats, config.warmup_runs):
                    for workflow_id in task.workflows:
                        key = (task.task_id, workflow_id, world.world_id, repeat, warmup)
                        if key in completed:
                            continue
                        run_id = _run_id(task, world, workflow_id, repeat, warmup)
                        trace = TraceRecorder(traces_root, run_id, exclusive=True)
                        await trace.start(
                            {
                                "mode": "scope_expansion_v0_longbench_reference",
                                "planner_constructed": False,
                                "measurement_series_id": config.run_prefix,
                                "measurement_protocol": MEASUREMENT_PROTOCOL,
                                "task_id": task.task_id,
                                "task_family": task.task_family,
                                "workflow_id": workflow_id,
                                "world_id": world.world_id,
                                "repeat": repeat,
                                "warmup": warmup,
                                "configured_bandwidth_mbps": world.bandwidth_mbps,
                                "configured_added_rtt_ms": world.rtt_ms,
                            }
                        )
                        started = perf_counter()
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
                            build_longbench_task_interaction(task, artifacts)
                            started = perf_counter()
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
                            question = _question_with_choices(task)
                            if task.task_family == "multi_document_qa":
                                result = await execute_longbench_multidoc_reference(
                                    runtime,
                                    workflow_id=cast(Any, workflow_id),
                                    query=question,
                                    documents=artifacts,
                                    document_workers=worker_ids,
                                    raw_tokens=raw_tokens,
                                    final_model_id=config.final_model_id,
                                    local_top_k_per_shard=cast(int, task.local_top_k_per_shard),
                                )
                            else:
                                result = await execute_structured_reference(
                                    runtime,
                                    workflow_id=cast(Any, workflow_id),
                                    query=question,
                                    shards=artifacts,
                                    shard_workers=worker_ids,
                                    raw_tokens=raw_tokens,
                                    structured_plan=cast(StructuredPlan, task.structured_plan),
                                    final_model_id=config.final_model_id,
                                )
                            e2e_ms = (perf_counter() - started) * 1000
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
                            executions = _execution_rows(events, registry, initial_sizes, transfers)
                            initial_workers = {
                                artifact.id: worker
                                for artifact, worker in zip(artifacts, worker_ids, strict=True)
                            }
                            lineage = _lineage_rows(executions, initial_workers, initial_sizes)
                            realized = _realized_trace(
                                run_id=run_id,
                                task_id=task.task_id,
                                workflow_id=cast(Any, workflow_id),
                                warmup=warmup,
                                executions=executions,
                                transfers=transfers,
                                lineage=lineage,
                                initial_workers=initial_workers,
                                e2e_latency_ms=e2e_ms,
                            )
                            metadata = cast(dict[str, object], realized["metadata"])
                            critical = cast(dict[str, object], metadata["critical_path_summary"])
                            row: dict[str, object] = {
                                "schema_version": "scope-expansion-v0-run-v1",
                                "run_id": run_id,
                                "measurement_series_id": config.run_prefix,
                                "protocol_id": MEASUREMENT_PROTOCOL,
                                "task_id": task.task_id,
                                "task_family": task.task_family,
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
                                    "documents_per_agent": per_agent,
                                },
                                "artifacts": {
                                    "raw_bytes": result.raw_bytes,
                                    "reduced_bytes": result.reduced_bytes,
                                    "reduced_tokens": result.reduced_tokens,
                                    "absolute_reducible_bytes": (
                                        result.raw_bytes - result.reduced_bytes
                                    ),
                                },
                                "transfers": {
                                    "count": len(transfers),
                                    "bytes": sum(
                                        int(cast(int | str, item["bytes"])) for item in transfers
                                    ),
                                    "latency_ms": sum(
                                        float(cast(float | int, item["latency_ms"]))
                                        for item in transfers
                                    ),
                                    "critical_ms": float(
                                        cast(
                                            float | int | str,
                                            critical["transfer_critical_ms"],
                                        )
                                    ),
                                    "records": transfers,
                                },
                                "service": {
                                    "local_preprocessing_ms": result.local_sum_ms,
                                    "final_model_ms": result.final_model_ms,
                                    "total_ms": sum(item.service_ms for item in result.executions),
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
                                "trace": realized,
                                "metadata": {
                                    "planner_constructed": False,
                                    "natural_boundaries_required": True,
                                    "configured_bandwidth_mbps": world.bandwidth_mbps,
                                    "configured_added_rtt_ms": world.rtt_ms,
                                    "network_shaping": ("worker_application_layer_wall_clock"),
                                    "final_model_id": config.final_model_id,
                                    "final_executor_id": config.final_executor_id,
                                    "e2e_boundary": ("workflow_start_to_final_artifact_created"),
                                    "answer_download_in_e2e": False,
                                },
                            }
                            if task.task_family == "multi_document_qa":
                                row["retrieval"] = {
                                    "algorithm": "bm25",
                                    "top_k_per_shard": task.local_top_k_per_shard,
                                    "calls": result.local_calls,
                                    "sum_ms": result.local_sum_ms,
                                    "critical_ms": result.local_critical_ms,
                                    "retrieved_document_count": (result.retrieved_document_count),
                                }
                            else:
                                row["local_compute"] = {
                                    "calls": result.local_calls,
                                    "sum_ms": result.local_sum_ms,
                                    "critical_ms": result.local_critical_ms,
                                }
                        except Exception as error:
                            e2e_ms = (perf_counter() - started) * 1000
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
                                "task_family": task.task_family,
                                "workflow_id": workflow_id,
                                "world_id": world.world_id,
                                "repeat": repeat,
                                "warmup": warmup,
                                "status": "failed",
                                "quality": None,
                                "error": f"{type(error).__name__}: {error}",
                                "metadata": {
                                    "planner_constructed": False,
                                    "configured_bandwidth_mbps": world.bandwidth_mbps,
                                    "configured_added_rtt_ms": world.rtt_ms,
                                },
                            }
                        _append_jsonl(raw_path, row)
                        if row["status"] == "completed":
                            completed.add(key)
                        written += 1
    finally:
        await close_worker_clients(clients)
    return {
        "raw_runs": {family: str(path) for family, path in raw_paths.items()},
        "new_rows": written,
        "completed_cells": len(completed),
    }
