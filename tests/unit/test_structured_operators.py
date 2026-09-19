"""Tests for dataset-neutral Worker-local structured record operators."""

import json
import re
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import ValidationError

from infra_mas.core.errors import ExecutionFailedError
from infra_mas.core.execution import (
    AggregateRecordsRequest,
    DeriveFieldsRequest,
    FilterRecordsRequest,
    RecordAggregation,
    RecordDerivation,
    RecordPredicate,
    RecordSort,
    SelectFieldsRequest,
    TopKRecordsRequest,
)
from infra_mas.execution.worker_client import WorkerClient
from infra_mas.operators import GENERAL_OPERATOR_REGISTRY, TaskInteractionSpec
from infra_mas.worker.artifact_store import ArtifactStore
from infra_mas.worker.server import create_app
from infra_mas.worker.service import WorkerService


async def _output_payload(
    service: WorkerService, artifact_id: str
) -> dict[str, Any]:
    path = await service.artifact_store.get_path(artifact_id)
    return json.loads(path.read_text(encoding="utf-8"))


async def test_structured_filter_select_derive_top_k_chain(tmp_path: Path) -> None:
    service = WorkerService("a4", ArtifactStore(tmp_path / "store", "a4"), [])
    source = await service.artifact_store.put_bytes(
        "run/records.jsonl",
        b"\n".join(
            [
                json.dumps(
                    {
                        "Symbol": "AAA",
                        "EndDate": "2025-12-31",
                        "TotalAssets": 100,
                        "TotalLiabilities": 20,
                        "nested": {"sector": "tech"},
                    }
                ).encode(),
                json.dumps(
                    {
                        "Symbol": "BBB",
                        "EndDate": "2025-12-31",
                        "TotalAssets": 100,
                        "TotalLiabilities": 40,
                        "nested": {"sector": "retail"},
                    }
                ).encode(),
                json.dumps(
                    {
                        "Symbol": "CCC",
                        "EndDate": "2024-12-31",
                        "TotalAssets": 900,
                        "TotalLiabilities": 10,
                        "nested": {"sector": "tech"},
                    }
                ).encode(),
            ]
        ),
        "application/x-ndjson",
    )

    filtered = await service.filter_records(
        FilterRecordsRequest(
            request_id="run/filter",
            input_artifacts=[source],
            output_artifact_id="run/filtered.json",
            predicates=[
                RecordPredicate(field="EndDate", operator="eq", value="2025-12-31"),
                RecordPredicate(field="TotalLiabilities", operator="gt", value=0),
            ],
        )
    )
    selected = await service.select_fields(
        SelectFieldsRequest(
            request_id="run/select",
            input_artifacts=filtered.output_artifacts,
            output_artifact_id="run/selected.json",
            fields=["Symbol", "TotalAssets", "TotalLiabilities", "nested.sector"],
        )
    )
    derived = await service.derive_fields(
        DeriveFieldsRequest(
            request_id="run/derive",
            input_artifacts=selected.output_artifacts,
            output_artifact_id="run/derived.json",
            derivations=[
                RecordDerivation(
                    output_field="asset_liability_ratio",
                    operation="divide",
                    left_field="TotalAssets",
                    right_field="TotalLiabilities",
                )
            ],
        )
    )
    top = await service.top_k_records(
        TopKRecordsRequest(
            request_id="run/top",
            input_artifacts=derived.output_artifacts,
            output_artifact_id="run/top.json",
            order_by=[RecordSort(field="asset_liability_ratio")],
            limit=1,
        )
    )

    filtered_payload = await _output_payload(service, "run/filtered.json")
    selected_payload = await _output_payload(service, "run/selected.json")
    top_payload = await _output_payload(service, "run/top.json")
    assert filtered.metadata["input_record_count"] == 3
    assert filtered.metadata["output_record_count"] == 2
    assert len(filtered_payload["records"]) == 2
    assert selected_payload["records"][0]["nested.sector"] == "tech"
    assert top_payload["records"] == [
        {
            "Symbol": "AAA",
            "TotalAssets": 100,
            "TotalLiabilities": 20,
            "asset_liability_ratio": 5.0,
            "nested.sector": "tech",
        }
    ]
    assert top.executor_id == "a4:top_k_records"
    assert top.metadata["input_record_count"] == 2
    for result in (filtered, selected, derived, top):
        content = (
            await service.artifact_store.get_path(result.output_artifacts[0].id)
        ).read_text(encoding="utf-8")
        expected_tokens = len(re.findall(r"[a-z0-9]+", content.lower()))
        assert expected_tokens > 0
        assert result.metadata["output_tokens"] == expected_tokens
        assert result.metadata["output_tokens_lexical"] == expected_tokens


async def test_aggregate_records_groups_and_computes_generic_statistics(
    tmp_path: Path,
) -> None:
    service = WorkerService("a5", ArtifactStore(tmp_path / "store", "a5"), [])
    source = await service.artifact_store.put_text(
        "run/records.json",
        json.dumps(
            [
                {"group": "b", "amount": 2, "name": "x"},
                {"group": "a", "amount": 3, "name": "x"},
                {"group": "a", "amount": 7, "name": "y"},
            ]
        ),
        "application/json",
    )
    result = await service.aggregate_records(
        AggregateRecordsRequest(
            request_id="run/aggregate",
            input_artifacts=[source],
            output_artifact_id="run/aggregated.json",
            group_by=["group"],
            aggregations=[
                RecordAggregation(output_field="rows", operation="count"),
                RecordAggregation(output_field="total", operation="sum", field="amount"),
                RecordAggregation(output_field="average", operation="mean", field="amount"),
                RecordAggregation(
                    output_field="unique_names",
                    operation="count_distinct",
                    field="name",
                ),
            ],
        )
    )

    payload = await _output_payload(service, "run/aggregated.json")
    assert payload["groups"] == [
        {"average": 5.0, "group": "a", "rows": 2, "total": 10, "unique_names": 2},
        {"average": 2.0, "group": "b", "rows": 1, "total": 2, "unique_names": 1},
    ]
    assert result.metadata["input_record_count"] == 3
    assert result.metadata["output_group_count"] == 2
    encoded = (
        await service.artifact_store.get_path(result.output_artifacts[0].id)
    ).read_text(encoding="utf-8")
    expected_tokens = len(re.findall(r"[a-z0-9]+", encoded.lower()))
    assert result.metadata["output_tokens"] == expected_tokens
    assert result.metadata["output_tokens_lexical"] == expected_tokens


async def test_structured_operators_reject_unsafe_or_invalid_inputs(tmp_path: Path) -> None:
    service = WorkerService("a28", ArtifactStore(tmp_path / "store", "a28"), [])
    text = await service.artifact_store.put_text(
        "run/records.txt", "not structured", "text/plain"
    )
    with pytest.raises(ExecutionFailedError, match="require JSON"):
        await service.filter_records(
            FilterRecordsRequest(
                request_id="run/filter",
                input_artifacts=[text],
                output_artifact_id="run/out.json",
                predicates=[RecordPredicate(field="x", operator="eq", value=1)],
            )
        )

    source = await service.artifact_store.put_text(
        "run/divide.json",
        json.dumps([{"numerator": 1, "denominator": 0}]),
        "application/json",
    )
    with pytest.raises(ExecutionFailedError, match="divides by zero"):
        await service.derive_fields(
            DeriveFieldsRequest(
                request_id="run/derive",
                input_artifacts=[source],
                output_artifact_id="run/derived.json",
                derivations=[
                    RecordDerivation(
                        output_field="ratio",
                        operation="divide",
                        left_field="numerator",
                        right_field="denominator",
                    )
                ],
            )
        )

    with pytest.raises(ValidationError, match="requires a list value"):
        RecordPredicate(field="x", operator="in", value="not-a-list")
    with pytest.raises(ValidationError, match="requires a field"):
        RecordAggregation(output_field="total", operation="sum")


async def test_worker_http_exposes_structured_operator_contract(tmp_path: Path) -> None:
    service = WorkerService("a28", ArtifactStore(tmp_path / "store", "a28"), [])
    source = await service.artifact_store.put_text(
        "run/records.json", json.dumps([{"value": 1}, {"value": 3}]), "application/json"
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_app(service)),
        base_url="http://worker.test",
    ) as http_client:
        client = WorkerClient(client=http_client)
        status = await client.status()
        result = await client.top_k_records(
            TopKRecordsRequest(
                request_id="run/top",
                input_artifacts=[source],
                output_artifact_id="run/top.json",
                order_by=[RecordSort(field="value")],
                limit=1,
            )
        )

    assert {
        "filter_records",
        "select_fields",
        "aggregate_records",
        "derive_fields",
        "top_k_records",
    } <= set(status.operators)
    assert result.executor_id == "a28:top_k_records"
    assert (await _output_payload(service, "run/top.json"))["records"] == [
        {"value": 3}
    ]


def test_general_registry_realizes_structured_operator_surface() -> None:
    task = TaskInteractionSpec.model_validate(
        {
            "task_id": "structured-generic",
            "objective": "Filter, project, derive, rank, and aggregate records.",
            "initial_artifacts": [
                {
                    "artifact_id": "records",
                    "kind": "application/json",
                    "source_ref": "records.json",
                }
            ],
            "operators": [
                "filter_records",
                "select_fields",
                "derive_fields",
                "top_k_records",
                "aggregate_records",
                "invoke_model",
            ],
            "observations": [
                {
                    "observation_id": "partial",
                    "produced_by": ["top_k_records", "aggregate_records"],
                    "description": "Compact structured partials.",
                }
            ],
            "runtime_verifier": {"level": "partial", "signals": []},
            "external_evaluator": {"evaluator_id": "benchmark"},
        }
    )
    GENERAL_OPERATOR_REGISTRY.validate_task(task)
