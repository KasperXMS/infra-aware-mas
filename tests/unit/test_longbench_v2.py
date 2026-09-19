from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from infra_mas.core.artifact import ArtifactRef
from infra_mas.core.execution import (
    ExecutionResult,
    InvocationSpec,
    RecordDerivation,
    RecordPredicate,
    RecordSort,
)
from infra_mas.longbench_v2 import (
    LongBenchSweepConfig,
    StructuredPlan,
    execute_longbench_multidoc_reference,
    execute_structured_reference,
    generate_longbench_v2_sweep,
)
from infra_mas.scope_expansion_v0 import SHARD_WORKERS, stable_document_worker


class FakeStructuredRuntime:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.invocations: list[InvocationSpec] = []
        self.active = 0
        self.max_active = 0

    async def _operator(
        self, name: str, artifacts: list[ArtifactRef], worker: str
    ) -> ExecutionResult:
        self.calls.append((name, worker))
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(0.005)
        self.active -= 1
        return ExecutionResult(
            request_id=f"{worker}-{name}",
            executor_id=f"{worker}:{name}",
            output_artifacts=[
                ArtifactRef(
                    id=f"run/{worker}-{name}.json",
                    artifact_type="application/json",
                    size_bytes=max(10, artifacts[0].size_bytes // 2),
                    locations=[worker],
                )
            ],
            queue_ms=0,
            service_ms=5,
            metadata={"semantic_operator": name, "output_tokens": 5},
        )

    async def filter_records_on_worker(
        self, artifacts: list[ArtifactRef], target_worker_id: str, **_: Any
    ) -> ExecutionResult:
        return await self._operator("filter_records", artifacts, target_worker_id)

    async def bm25_retrieve_on_worker(
        self,
        query: str,
        artifacts: list[ArtifactRef],
        target_worker_id: str,
        *,
        top_k: int = 3,
        parent_action_id: str | None = None,
    ) -> ExecutionResult:
        del query, top_k, parent_action_id
        return await self._operator("bm25_retrieve", artifacts, target_worker_id)

    async def select_fields_on_worker(
        self, artifacts: list[ArtifactRef], target_worker_id: str, **_: Any
    ) -> ExecutionResult:
        return await self._operator("select_fields", artifacts, target_worker_id)

    async def derive_fields_on_worker(
        self, artifacts: list[ArtifactRef], target_worker_id: str, **_: Any
    ) -> ExecutionResult:
        return await self._operator("derive_fields", artifacts, target_worker_id)

    async def aggregate_records_on_worker(
        self, artifacts: list[ArtifactRef], target_worker_id: str, **_: Any
    ) -> ExecutionResult:
        return await self._operator("aggregate_records", artifacts, target_worker_id)

    async def top_k_records_on_worker(
        self, artifacts: list[ArtifactRef], target_worker_id: str, **_: Any
    ) -> ExecutionResult:
        return await self._operator("top_k_records", artifacts, target_worker_id)

    async def invoke(
        self,
        invocation: InvocationSpec,
        *,
        parent_action_id: str | None = None,
    ) -> ExecutionResult:
        del parent_action_id
        self.invocations.append(invocation)
        return ExecutionResult(
            request_id="final",
            executor_id="a28-vlm",
            output_artifacts=[
                ArtifactRef(
                    id="run/answer.txt",
                    artifact_type="text/plain",
                    size_bytes=10,
                    locations=["a28"],
                )
            ],
            queue_ms=0,
            service_ms=20,
            metadata={"input_tokens": 30, "output_tokens": 3},
        )


def _shards() -> tuple[list[ArtifactRef], list[str]]:
    return (
        [
            ArtifactRef(
                id=f"run/{worker}.json",
                artifact_type="application/json",
                size_bytes=1_000,
                locations=[worker],
            )
            for worker in SHARD_WORKERS
        ],
        list(SHARD_WORKERS),
    )


async def test_structured_workflows_share_final_model_and_parallelize_local_compute() -> None:
    shards, workers = _shards()
    plan = StructuredPlan(
        predicates=[RecordPredicate(field="liability", operator="gt", value=0)],
        select_fields=["assets", "liability", "symbol"],
        derivations=[
            RecordDerivation(
                output_field="ratio",
                operation="divide",
                left_field="assets",
                right_field="liability",
            )
        ],
        order_by=[RecordSort(field="ratio", direction="descending", nulls="last")],
        limit=1,
    )
    central_runtime = FakeStructuredRuntime()
    distributed_runtime = FakeStructuredRuntime()

    central = await execute_structured_reference(
        central_runtime,
        workflow_id="centralized_raw",
        query="Which symbol has the largest ratio?",
        shards=shards,
        shard_workers=workers,
        raw_tokens=300,
        structured_plan=plan,
        final_model_id="edge-text-reasoner",
    )
    distributed = await execute_structured_reference(
        distributed_runtime,
        workflow_id="distributed_compute",
        query="Which symbol has the largest ratio?",
        shards=shards,
        shard_workers=workers,
        raw_tokens=300,
        structured_plan=plan,
        final_model_id="edge-text-reasoner",
    )

    assert central_runtime.calls == []
    assert central.local_calls == 0
    assert central.reduced_bytes == central.raw_bytes
    assert len(central_runtime.invocations[0].input_artifacts) == 3

    assert distributed.local_calls == 12
    assert distributed_runtime.max_active == 3
    assert {worker for _, worker in distributed_runtime.calls} == set(SHARD_WORKERS)
    assert [name for name, worker in distributed_runtime.calls if worker == "a4"] == [
        "filter_records",
        "select_fields",
        "derive_fields",
        "top_k_records",
    ]
    assert len(distributed_runtime.invocations[0].input_artifacts) == 3
    assert distributed.local_sum_ms == 60
    assert distributed.local_critical_ms == 20

    central_invocation = central_runtime.invocations[0]
    distributed_invocation = distributed_runtime.invocations[0]
    assert central_invocation.model_id == distributed_invocation.model_id
    assert central_invocation.instructions == distributed_invocation.instructions
    assert central_invocation.task == distributed_invocation.task
    assert '"The correct answer is (insert answer here)"' in central_invocation.task


async def test_multidoc_retrieval_only_calls_workers_with_natural_documents() -> None:
    runtime = FakeStructuredRuntime()
    documents = [
        ArtifactRef(
            id="run/a4.json",
            artifact_type="application/json",
            size_bytes=1_000,
            locations=["a4"],
        ),
        ArtifactRef(
            id="run/a28.json",
            artifact_type="application/json",
            size_bytes=1_000,
            locations=["a28"],
        ),
    ]

    result = await execute_longbench_multidoc_reference(
        runtime,
        workflow_id="distributed_retrieval",
        query="Question with choices",
        documents=documents,
        document_workers=["a4", "a28"],
        raw_tokens=200,
        final_model_id="edge-text-reasoner",
        local_top_k_per_shard=1,
    )

    assert result.local_calls == 2
    assert {worker for name, worker in runtime.calls if name == "bm25_retrieve"} == {
        "a4",
        "a28",
    }
    assert "a5" not in {worker for _, worker in runtime.calls}


def _ids_across_workers() -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    index = 0
    while seen != set(SHARD_WORKERS):
        source_id = f"natural-document-{index}"
        ids.append(source_id)
        seen.add(stable_document_worker(source_id))
        index += 1
    return ids


def _task_bank_rows() -> list[dict[str, object]]:
    doc_ids = _ids_across_workers()
    doc_artifacts = [f"doc-{index}" for index in range(len(doc_ids))]
    doc_placement = {
        artifact: stable_document_worker(source).replace("a", "A", 1)
        for artifact, source in zip(doc_artifacts, doc_ids, strict=True)
    }
    structured_artifacts = ["records-a4", "records-a5", "records-a28"]
    return [
        {
            "task_family": "multi_document_qa",
            "planner_visible": {
                "task_id": "longbench-multidoc",
                "dataset": "LongBench-v2",
                "task_family": "multi_document_qa",
                "instruction": "Answer from all natural documents.",
                "query": "Which option follows from the documents?",
                "answer_choices": {label: f"Choice {label}" for label in "ABCD"},
                "evaluator_type": "longbench_v2_choice",
                "artifact_ids": doc_artifacts,
                "artifact_types": {item: "document" for item in doc_artifacts},
                "artifact_sizes": {item: 100 for item in doc_artifacts},
                "artifact_tokens": {item: 20 for item in doc_artifacts},
                "artifact_placement": doc_placement,
                "artifact_source_refs": dict(zip(doc_artifacts, doc_ids, strict=True)),
            },
            "evaluator_only": {"answer": "MUST_NOT_LEAK"},
        },
        {
            "task_family": "structured_data_analysis",
            "planner_visible": {
                "task_id": "longbench-structured",
                "dataset": "LongBench-v2",
                "task_family": "structured_data_analysis",
                "instruction": "Find the maximum derived ratio.",
                "query": "Which option names the maximum ratio?",
                "answer_choices": {label: f"Choice {label}" for label in "ABCD"},
                "evaluator_type": "longbench_v2_choice",
                "artifact_ids": structured_artifacts,
                "artifact_types": {item: "records" for item in structured_artifacts},
                "artifact_sizes": {item: 1_000 for item in structured_artifacts},
                "artifact_tokens": {item: 100 for item in structured_artifacts},
                "artifact_placement": dict(
                    zip(structured_artifacts, ["A4", "A5", "A28"], strict=True)
                ),
                "artifact_source_refs": {
                    item: f"natural-shard-{index}"
                    for index, item in enumerate(structured_artifacts)
                },
                "structured_plan": {
                    "predicates": [{"field": "liability", "operator": "gt", "value": 0}],
                    "select_fields": ["assets", "liability", "symbol"],
                    "aggregations": [],
                    "group_by": [],
                    "derived_fields": [
                        {
                            "alias": "ratio",
                            "operator": "divide",
                            "operands": ["assets", "liability"],
                        }
                    ],
                    "order_by": [{"field": "ratio", "direction": "descending"}],
                    "limit": 1,
                },
            },
            "evaluator_only": {"answer": "ALSO_MUST_NOT_LEAK"},
        },
    ]


def test_generator_normalizes_generic_plan_without_gold(tmp_path: Path) -> None:
    task_bank = tmp_path / "task_bank.jsonl"
    task_bank.write_text(
        "".join(json.dumps(row) + "\n" for row in _task_bank_rows()),
        encoding="utf-8",
    )
    output = tmp_path / "longbench.yaml"

    config = generate_longbench_v2_sweep(task_bank, output)

    assert isinstance(config, LongBenchSweepConfig)
    assert {task.task_family for task in config.tasks} == {
        "multi_document_qa",
        "structured_data_analysis",
    }
    structured = next(
        task for task in config.tasks if task.task_family == "structured_data_analysis"
    )
    assert structured.structured_plan is not None
    assert structured.structured_plan.derivations[0].output_field == "ratio"
    assert structured.workflows == ("centralized_raw", "distributed_compute")
    rendered = output.read_text(encoding="utf-8")
    assert "MUST_NOT_LEAK" not in rendered
    assert "evaluator_only" not in rendered
    assert "\\home\\edge" not in rendered


async def test_multidoc_uses_official_longbench_v2_answer_contract() -> None:
    runtime = FakeStructuredRuntime()
    documents, workers = _shards()

    await execute_longbench_multidoc_reference(
        runtime,
        workflow_id="centralized_raw",
        query="Question with choices",
        documents=documents,
        document_workers=workers,
        raw_tokens=300,
        final_model_id="edge-text-reasoner",
        local_top_k_per_shard=1,
    )

    assert '"The correct answer is (insert answer here)"' in runtime.invocations[0].instructions


def test_sweep_rejects_missing_family_and_wrong_protocol() -> None:
    rows = _task_bank_rows()
    # A direct model-validation smoke test is covered by the generator.  This checks
    # that the fixed protocol cannot silently become a convenience run.
    assert rows
    with pytest.raises(ValueError):
        LongBenchSweepConfig.model_validate(
            {
                "models_config": "models.yaml",
                "executors_config": "executors.yaml",
                "final_model_id": "edge-text-reasoner",
                "final_executor_id": "a28-vlm",
                "tasks": [],
                "worlds": [],
                "warmup_runs": 0,
                "repeats": 1,
            }
        )


def test_structured_plan_rejects_projection_of_not_yet_derived_field() -> None:
    with pytest.raises(ValueError, match="later derive step"):
        StructuredPlan(
            select_fields=["assets", "liability", "ratio"],
            derivations=[
                RecordDerivation(
                    output_field="ratio",
                    operation="divide",
                    left_field="assets",
                    right_field="liability",
                )
            ],
            order_by=[RecordSort(field="ratio")],
            limit=1,
        )
