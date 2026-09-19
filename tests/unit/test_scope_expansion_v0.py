from __future__ import annotations

import asyncio
import importlib
import json
import sys
from pathlib import Path
from typing import Any, cast

import pytest

from infra_mas.core.artifact import ArtifactRef
from infra_mas.core.execution import BM25RetrieveRequest, ExecutionResult, InvocationSpec
from infra_mas.core.executor import ExecutorSpec
from infra_mas.core.model import ModelSpec
from infra_mas.execution.executor_registry import ExecutorRegistry
from infra_mas.runtime.model_registry import ModelRegistry
from infra_mas.scope_expansion_v0 import (
    ScopeExpansionSweepConfig,
    _execution_rows,  # pyright: ignore[reportPrivateUsage]
    _lineage_rows,  # pyright: ignore[reportPrivateUsage]
    _physical_run_id,  # pyright: ignore[reportPrivateUsage]
    _realized_trace,  # pyright: ignore[reportPrivateUsage]
    _transfer_rows,  # pyright: ignore[reportPrivateUsage]
    _validate_deployment,  # pyright: ignore[reportPrivateUsage]
    execute_multidoc_reference,
    generate_scope_expansion_sweep,
    stable_document_worker,
)
from infra_mas.worker.artifact_store import ArtifactStore
from infra_mas.worker.service import WorkerService


def test_physical_run_id_is_short_but_semantically_distinct() -> None:
    task_id = "multihop-rag-train-9bae0079038050a37a1ae583"
    central = _physical_run_id(
        task_id,
        "H1_distributed_constrained",
        "centralized_raw",
        0,
        True,
        nonce="12345678",
    )
    distributed = _physical_run_id(
        task_id,
        "H2_distributed_favorable",
        "distributed_retrieval",
        3,
        False,
        nonce="12345678",
    )

    assert central == "se0-7a1ae583-h1-cr-warmup-12345678"
    assert distributed == "se0-7a1ae583-h2-dr-r3-12345678"
    assert max(len(central), len(distributed)) < 64


async def test_worker_bm25_is_deterministic_and_reports_retrieval_metadata(
    tmp_path: Path,
) -> None:
    service = WorkerService("a4", ArtifactStore(tmp_path / "store", "a4"), [])
    doc_b = await service.artifact_store.put_text("run/doc-b.txt", "alpha shared")
    doc_a = await service.artifact_store.put_text("run/doc-a.txt", "alpha shared")
    doc_c = await service.artifact_store.put_text("run/doc-c.txt", "unrelated")

    result = await service.bm25_retrieve(
        BM25RetrieveRequest(
            request_id="run/retrieve",
            query="ALPHA",
            input_artifacts=[doc_b, doc_c, doc_a],
            output_artifact_id="run/evidence.json",
            top_k=2,
        )
    )

    payload = json.loads(
        (await service.artifact_store.get_path(result.output_artifacts[0].id)).read_text(
            encoding="utf-8"
        )
    )
    assert [item["artifact_id"] for item in payload["documents"]] == [
        "run/doc-a.txt",
        "run/doc-b.txt",
    ]
    assert result.executor_id == "a4:bm25_retrieve"
    assert result.metadata == {
        "semantic_operator": "bm25_retrieve",
        "algorithm": "bm25",
        "top_k": 2,
        "k1": 1.5,
        "b": 0.75,
        "candidate_document_count": 3,
        "candidate_tokens": 5,
        "retrieved_document_count": 2,
        "retrieved_tokens": 4,
        "retrieved_artifact_ids": ["run/doc-a.txt", "run/doc-b.txt"],
    }
    assert "bm25_retrieve" in service.status().operators


class FakeScopeRuntime:
    def __init__(self) -> None:
        self.retrievals: list[tuple[str, list[str], str, int]] = []
        self.invocations: list[InvocationSpec] = []
        self.active_retrievals = 0
        self.max_active_retrievals = 0

    async def bm25_retrieve_on_worker(
        self,
        query: str,
        artifacts: list[ArtifactRef],
        target_worker_id: str,
        *,
        top_k: int = 3,
        parent_action_id: str | None = None,
    ) -> ExecutionResult:
        del parent_action_id
        self.retrievals.append(
            (query, [artifact.id for artifact in artifacts], target_worker_id, top_k)
        )
        self.active_retrievals += 1
        self.max_active_retrievals = max(
            self.max_active_retrievals, self.active_retrievals
        )
        await asyncio.sleep(0.01)
        self.active_retrievals -= 1
        return ExecutionResult(
            request_id=f"retrieve-{target_worker_id}",
            executor_id=f"{target_worker_id}:bm25_retrieve",
            output_artifacts=[
                ArtifactRef(
                    id=f"run/evidence-{target_worker_id}.json",
                    artifact_type="application/json",
                    size_bytes=100,
                    locations=[target_worker_id],
                )
            ],
            queue_ms=0,
            service_ms=10,
            metadata={
                "semantic_operator": "bm25_retrieve",
                "retrieved_document_count": 3,
                "retrieved_tokens": 30,
            },
        )

    async def invoke(
        self,
        invocation: InvocationSpec,
        *,
        parent_action_id: str | None = None,
    ) -> ExecutionResult:
        del parent_action_id
        self.invocations.append(invocation)
        return ExecutionResult(
            request_id=f"final-{len(self.invocations)}",
            executor_id="a28-vlm",
            output_artifacts=[
                ArtifactRef(
                    id=f"run/answer-{len(self.invocations)}.txt",
                    artifact_type="text/plain",
                    size_bytes=12,
                    locations=["a28"],
                )
            ],
            queue_ms=0,
            service_ms=20,
            metadata={"input_tokens": 50, "output_tokens": 4, "api_cost_usd": 0.0},
        )


def _documents() -> tuple[list[ArtifactRef], list[str]]:
    workers = ["a4", "a4", "a5", "a5", "a28", "a28"]
    return (
        [
            ArtifactRef(
                id=f"run/doc-{index}.json",
                artifact_type="application/json",
                size_bytes=1_000,
                locations=[worker],
            )
            for index, worker in enumerate(workers)
        ],
        workers,
    )


async def test_references_share_prompt_and_model_but_change_information_flow() -> None:
    documents, workers = _documents()
    centralized = FakeScopeRuntime()
    distributed = FakeScopeRuntime()

    central_result = await execute_multidoc_reference(
        centralized,
        workflow_id="centralized_raw",
        query="Which organization is linked across the documents?",
        documents=documents,
        document_workers=workers,
        raw_tokens=600,
        final_model_id="edge-text-reasoner",
    )
    distributed_result = await execute_multidoc_reference(
        distributed,
        workflow_id="distributed_retrieval",
        query="Which organization is linked across the documents?",
        documents=documents,
        document_workers=workers,
        raw_tokens=600,
        final_model_id="edge-text-reasoner",
    )

    assert centralized.retrievals == []
    assert central_result.retrieval_calls == 0
    assert central_result.reduced_bytes == central_result.raw_bytes
    assert len(centralized.invocations[0].input_artifacts) == 6

    assert len(distributed.retrievals) == 3
    assert {item[2] for item in distributed.retrievals} == {"a4", "a5", "a28"}
    assert {item[3] for item in distributed.retrievals} == {3}
    assert distributed.max_active_retrievals == 3
    assert distributed_result.retrieval_calls == 3
    assert distributed_result.retrieved_document_count == 9
    assert distributed_result.reduced_bytes == 300
    assert len(distributed.invocations[0].input_artifacts) == 3

    central_invocation = centralized.invocations[0]
    distributed_invocation = distributed.invocations[0]
    assert central_invocation.model_id == distributed_invocation.model_id
    assert central_invocation.instructions == distributed_invocation.instructions
    assert central_invocation.task == distributed_invocation.task
    assert {item.locations[0] for item in central_invocation.input_artifacts} == {
        "a4",
        "a5",
        "a28",
    }
    assert {item.locations[0] for item in distributed_invocation.input_artifacts} == {
        "a4",
        "a5",
        "a28",
    }


def _config_payload() -> dict[str, object]:
    inputs: list[dict[str, object]] = []
    for index in range(30):
        document_id = f"document-{index}"
        worker_id = stable_document_worker(document_id)
        inputs.append(
            {
                "document_id": document_id,
                "artifact_id": f"artifact-{index}",
                "path": f"/home/edge/xiaoming/scope_expansion_v0/{index}.json",
                "worker_id": worker_id,
                "expected_size_bytes": 100,
                "expected_tokens": 10,
            }
        )
    return {
        "models_config": "models.yaml",
        "executors_config": "executors.yaml",
        "final_model_id": "edge-text-reasoner",
        "final_executor_id": "a28-vlm",
        "tasks": [
            {
                "task_id": "opaque-task",
                "query": "What happened?",
                "evaluator_id": "multihop_rag_official_token_intersection",
                "inputs": inputs,
            }
        ],
        "worlds": [
            {
                "world_id": "H1_distributed_constrained",
                "bandwidth_mbps": 3,
                "rtt_ms": 50,
            },
            {
                "world_id": "H2_distributed_favorable",
                "bandwidth_mbps": 100,
                "rtt_ms": 0,
            },
        ],
    }


def test_sweep_config_has_fixed_top_k_rejects_gold_and_4090() -> None:
    payload = _config_payload()
    config = ScopeExpansionSweepConfig.model_validate(payload)
    assert config.local_top_k_per_shard == 3
    assert config.warmup_runs == 1
    assert config.repeats == 3

    payload["local_top_k_per_shard"] = 4
    with pytest.raises(ValueError, match="Input should be 3"):
        ScopeExpansionSweepConfig.model_validate(payload)
    payload["local_top_k_per_shard"] = 3
    task = cast(list[dict[str, object]], payload["tasks"])[0]
    task["evidence_document_count"] = 2
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        ScopeExpansionSweepConfig.model_validate(payload)

    models = ModelRegistry(
        [
            ModelSpec(
                model_id="edge-text-reasoner",
                description="test",
                input_modalities=["text"],
                output_modalities=["text"],
                context_window=8192,
            )
        ]
    )
    registry = ExecutorRegistry(
        [
            ExecutorSpec(
                id="a28-vlm",
                capability="text_reasoning",
                worker_id="a28",
                model_id="edge-text-reasoner",
                device="orin",
                site="A28",
            )
        ],
        {
            "a4": "http://a4.test",
            "a5": "http://a5.test",
            "a28": "http://a28.test",
            "gpu": "http://gpu.test",
        },
        {"a4": "A4", "a5": "A5", "a28": "A28", "gpu": "4090"},
    )
    with pytest.raises(ValueError, match="Jetson-only"):
        _validate_deployment(config, registry, models)


def test_generator_uses_only_visible_fields_and_writes_posix_orin_paths(
    tmp_path: Path,
) -> None:
    rows: list[dict[str, object]] = []
    for task_index in range(3):
        artifact_ids = [f"task-{task_index}-doc-{index}" for index in range(30)]
        placements = {
            artifact_id: {
                "a4": "A4",
                "a5": "A5",
                "a28": "A28",
            }[stable_document_worker(artifact_id)]
            for artifact_id in artifact_ids
        }
        rows.append(
            {
                "planner_visible": {
                    "task_id": f"task-{task_index}",
                    "query": "Visible query",
                    "evaluator_type": "multihop_rag_official_token_intersection",
                    "artifact_ids": artifact_ids,
                    "artifact_document_ids": {
                        artifact_id: artifact_id for artifact_id in artifact_ids
                    },
                    "artifact_sizes": {artifact_id: 100 for artifact_id in artifact_ids},
                    "artifact_tokens": {artifact_id: 10 for artifact_id in artifact_ids},
                    "artifact_placement": placements,
                },
                "evaluator_only": {"answer": "MUST_NOT_LEAK"},
            }
        )
    task_bank = tmp_path / "task_bank.jsonl"
    task_bank.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    output = tmp_path / "sweep.yaml"

    config = generate_scope_expansion_sweep(task_bank, output)

    rendered = output.read_text(encoding="utf-8")
    assert len(config.tasks) == 3
    assert config.final_model_id == "edge-text-reasoner"
    assert [(world.bandwidth_mbps, world.rtt_ms) for world in config.worlds] == [
        (3.0, 83.0),
        (100.0, 33.0),
    ]
    assert "MUST_NOT_LEAK" not in rendered
    assert "evidence_document_count" not in rendered
    assert "/home/edge/xiaoming/scope_expansion_v0/" in rendered
    assert "\\home\\edge" not in rendered


def test_trace_has_standard_spans_and_row_lineage_uses_input_bytes() -> None:
    registry = ExecutorRegistry.from_yaml(
        Path(__file__).parents[2] / "configs" / "scope_expansion_v0" / "executors.yaml"
    )
    events = [
        {
            "event_type": "execution.request",
            "action_id": "retrieve-a",
            "semantic_operator": "bm25_retrieve",
            "input_artifacts": ["raw-a"],
        },
        {
            "event_type": "worker.execution.start",
            "action_id": "retrieve-a",
            "timestamp": "2026-01-01T00:00:00+00:00",
        },
        {
            "event_type": "worker.execution.end",
            "action_id": "retrieve-a",
            "timestamp": "2026-01-01T00:00:00.010000+00:00",
            "success": True,
            "executor": "a4:bm25_retrieve",
            "worker_id": "a4",
            "service_ms": 10.0,
            "output_artifacts": ["evidence-a"],
        },
        {
            "event_type": "artifact.created",
            "action_id": "retrieve-a",
            "artifact_id": "evidence-a",
            "size_bytes": 100,
        },
        {
            "event_type": "execution.request",
            "action_id": "final",
            "semantic_operator": "invoke_model",
            "model_id": "edge-text-reasoner",
            "input_artifacts": ["evidence-a"],
        },
        {
            "event_type": "artifact.transfer.start",
            "action_id": "final",
            "artifact_id": "evidence-a",
            "source_worker_id": "a4",
            "target_worker_id": "a28",
            "timestamp": "2026-01-01T00:00:00.010000+00:00",
        },
        {
            "event_type": "artifact.transfer.end",
            "action_id": "final",
            "artifact_id": "evidence-a",
            "source_worker_id": "a4",
            "target_worker_id": "a28",
            "timestamp": "2026-01-01T00:00:00.015000+00:00",
            "success": True,
            "bytes_transferred": 100,
            "transfer_ms": 5.0,
        },
        {
            "event_type": "worker.execution.start",
            "action_id": "final",
            "timestamp": "2026-01-01T00:00:00.015000+00:00",
        },
        {
            "event_type": "worker.execution.end",
            "action_id": "final",
            "timestamp": "2026-01-01T00:00:00.017000+00:00",
            "success": True,
            "executor": "a28-vlm",
            "worker_id": "a28",
            "service_ms": 2.0,
            "output_artifacts": ["answer"],
        },
        {
            "event_type": "artifact.created",
            "action_id": "final",
            "artifact_id": "answer",
            "size_bytes": 120,
        },
    ]
    transfers = _transfer_rows(events, registry)
    executions = _execution_rows(events, registry, {"raw-a": 1_000}, transfers)
    lineage = _lineage_rows(executions, {"raw-a": "a4"}, {"raw-a": 1_000})
    trace = _realized_trace(
        run_id="run",
        task_id="task",
        workflow_id="distributed_retrieval",
        warmup=False,
        executions=executions,
        transfers=transfers,
        lineage=lineage,
        initial_workers={"raw-a": "a4"},
        e2e_latency_ms=17,
    )

    assert set(trace) == {
        "schema_version",
        "run_id",
        "task_id",
        "workflow_id",
        "warmup",
        "dependency_evidence",
        "timestamp_evidence",
        "trace_coverage",
        "spans",
        "e2e_latency_ms",
        "quality",
        "metadata",
    }
    assert len(cast(list[object], trace["spans"])) == 3
    assert executions[1]["depends_on"] == ["final:transfer-1"]
    assert transfers[0]["depends_on"] == ["retrieve-a"]
    assert lineage[0]["artifact_bytes"] == 1_000
    assert lineage[1]["artifact_bytes"] == 100


def test_runtime_row_shape_validates_with_infra_bench() -> None:
    bench_source = Path(__file__).parents[3] / "infra-bench" / "src"
    sys.path.insert(0, str(bench_source))
    schemas = importlib.import_module("infra_bench.schemas")
    scope_run: Any = getattr(schemas, "ScopeRun")
    execution: dict[str, object] = {
        "action_id": "final",
        "operator_id": "invoke_model",
        "executor_id": "a28-vlm",
        "worker_id": "a28",
        "site_id": "A28",
        "service_ms": 20.0,
        "input_bytes": 100,
        "output_bytes": 12,
        "input_tokens": 10,
        "output_tokens": 2,
        "input_artifacts": ["raw-a"],
        "output_artifacts": ["answer"],
        "started_at": "2026-01-01T00:00:00+00:00",
        "finished_at": "2026-01-01T00:00:00.020000+00:00",
        "depends_on": [],
        "model_id": "edge-text-reasoner",
    }
    lineage: dict[str, object] = {
        "producer_agent": "a28",
        "consumer_agent": "a28",
        "input_artifact": "raw-a",
        "derived_artifact": "answer",
        "artifact_bytes": 100,
        "operator": "invoke_model",
    }
    trace = _realized_trace(
        run_id="run",
        task_id="task",
        workflow_id="centralized_raw",
        warmup=False,
        executions=[execution],
        transfers=[],
        lineage=[lineage],
        initial_workers={"raw-a": "a28"},
        e2e_latency_ms=21,
    )
    row: dict[str, object] = {
        "schema_version": "scope-expansion-v0-run-v1",
        "run_id": "run",
        "measurement_series_id": "scope-expansion-v0-multihop-rag",
        "protocol_id": "steady_state_1_warmup_3_measured_v1",
        "task_id": "task",
        "task_family": "multi_document_qa",
        "workflow_id": "centralized_raw",
        "world_id": "H1_distributed_constrained",
        "repeat": 1,
        "warmup": False,
        "status": "completed",
        "quality": None,
        "demand": {
            "artifact_count": 1,
            "raw_bytes": 100,
            "raw_tokens": 10,
            "documents_per_agent": {"A4": 0, "A5": 0, "A28": 1},
        },
        "retrieval": {
            "algorithm": "bm25",
            "top_k_per_shard": 3,
            "calls": 0,
            "sum_ms": 0.0,
            "critical_ms": 0.0,
            "retrieved_document_count": 0,
        },
        "artifacts": {
            "raw_bytes": 100,
            "reduced_bytes": 100,
            "reduced_tokens": 10,
            "absolute_reducible_bytes": 0,
            "records": [
                {
                    "artifact_id": "raw-a",
                    "document_id": "doc-a",
                    "kind": "raw_document",
                    "site_id": "A28",
                    "bytes": 100,
                    "tokens": 10,
                }
            ],
        },
        "transfers": {
            "count": 0,
            "bytes": 0,
            "latency_ms": 0.0,
            "critical_ms": 0.0,
            "records": [],
        },
        "service": {
            "local_preprocessing_ms": 0.0,
            "final_model_ms": 20.0,
            "total_ms": 20.0,
        },
        "usage": {"input_tokens": 10, "output_tokens": 2, "api_cost_usd": 0.0},
        "e2e_latency_ms": 21.0,
        "executions": [execution],
        "lineage": [lineage],
        "final_answer": "ANSWER: test",
        "trace": trace,
        "metadata": {
            "configured_bandwidth_mbps": 3.0,
            "configured_added_rtt_ms": 83.0,
        },
    }

    validated = scope_run.model_validate(row)
    assert validated.trace.trace_coverage == "complete"
