"""Unit tests for the Phase 1 core schemas."""

import pytest
from pydantic import BaseModel, ValidationError

from infra_mas.core import (
    AgentSpec,
    ArtifactRef,
    ExecutionRequest,
    ExecutionResult,
    ExecutorSpec,
    ModelRequest,
    ModelResult,
)


def artifact() -> ArtifactRef:
    return ArtifactRef(
        id="run-001/evidence-001",
        artifact_type="text/plain",
        size_bytes=18_000,
        locations=["orin-1"],
    )


INVALID_MODEL_CASES: list[tuple[type[BaseModel], dict[str, object]]] = [
    (
        AgentSpec,
        {
            "name": "   ",
            "capability": "reasoning",
            "instructions": "Reason over evidence.",
        },
    ),
    (
        ArtifactRef,
        {
            "id": "run-001/input-001",
            "artifact_type": "video/mp4",
            "size_bytes": -1,
            "locations": ["orin-1"],
        },
    ),
    (
        ExecutorSpec,
        {
            "id": "gpu-1-llm",
            "capability": "reasoning",
            "worker_id": "gpu-1",
            "model": "",
            "device": "rtx4090",
            "site": "remote",
        },
    ),
    (
        ExecutionRequest,
        {
            "request_id": "request-001",
            "agent": "reasoner",
            "capability": "reasoning",
            "task": "",
            "inputs": [],
        },
    ),
    (
        ExecutionResult,
        {
            "request_id": "request-001",
            "executor_id": "gpu-1-llm",
            "output_artifacts": [],
            "queue_ms": -0.1,
            "compute_ms": 10.0,
        },
    ),
    (ModelRequest, {"task": "", "input_paths": []}),
    (ModelResult, {"output_text": "answer", "latency_ms": float("nan")}),
]


@pytest.mark.parametrize(
    "model",
    [
        AgentSpec(
            name="vision_extractor",
            capability="visual_understanding",
            instructions="Extract task-relevant visual evidence.",
        ),
        artifact(),
        ExecutorSpec(
            id="orin-1-vlm",
            capability="visual_understanding",
            worker_id="orin-1",
            model="local-vlm",
            device="agx-orin",
            site="edge",
        ),
        ExecutionRequest(
            request_id="request-001",
            agent="vision_extractor",
            capability="visual_understanding",
            task="Extract relevant evidence.",
            inputs=[artifact()],
        ),
        ExecutionResult(
            request_id="request-001",
            executor_id="orin-1-vlm",
            output_artifacts=[artifact()],
            queue_ms=1.5,
            compute_ms=100.0,
        ),
        ModelRequest(task="Extract relevant evidence.", input_paths=["data/video-001.mp4"]),
        ModelResult(output_text="Evidence found.", latency_ms=100.0),
    ],
)
def test_json_serialization_round_trip(model: BaseModel) -> None:
    restored = type(model).model_validate_json(model.model_dump_json())
    assert restored == model


@pytest.mark.parametrize(
    ("model_type", "payload"),
    INVALID_MODEL_CASES,
)
def test_invalid_values_are_rejected(
    model_type: type[BaseModel], payload: dict[str, object]
) -> None:
    with pytest.raises(ValidationError):
        model_type.model_validate(payload)


def test_artifact_requires_a_location() -> None:
    with pytest.raises(ValidationError, match="at least 1 item"):
        ArtifactRef(
            id="run-001/input-001",
            artifact_type="video/mp4",
            size_bytes=1,
            locations=[],
        )


def test_artifact_locations_are_unique() -> None:
    with pytest.raises(ValidationError, match="locations must be unique"):
        ArtifactRef(
            id="run-001/input-001",
            artifact_type="video/mp4",
            size_bytes=1,
            locations=["orin-1", "orin-1"],
        )


def test_unknown_fields_are_rejected() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        AgentSpec.model_validate(
            {
                "name": "reasoner",
                "capability": "reasoning",
                "instructions": "Reason over evidence.",
                "executor_id": "gpu-1-llm",
            }
        )


def test_execution_result_metadata_defaults_are_isolated() -> None:
    first = ExecutionResult(
        request_id="request-001",
        executor_id="gpu-1-llm",
        output_artifacts=[],
        queue_ms=0,
        compute_ms=10,
    )
    second = ExecutionResult(
        request_id="request-002",
        executor_id="gpu-1-llm",
        output_artifacts=[],
        queue_ms=0,
        compute_ms=10,
    )

    first.metadata["attempt"] = 1

    assert second.metadata == {}
