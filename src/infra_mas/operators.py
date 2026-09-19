"""Stable semantic operator contract shared with infra-bench on the wire."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

VerifierLevel = Literal["none", "partial", "terminal"]
OperatorId = Literal[
    "search_code",
    "read_file",
    "edit_file",
    "apply_patch",
    "run_targeted_test",
    "run_full_test",
    "invoke_model",
    "submit_patch",
    "read_artifact",
    "sample_frames",
    "make_contact_sheet",
    "extract_clip",
    "process_local_artifact",
    "aggregate_artifacts",
    "bm25_retrieve",
    "filter_records",
    "select_fields",
    "aggregate_records",
    "derive_fields",
    "top_k_records",
]


class InitialArtifactSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    artifact_id: str
    kind: str
    source_ref: str


class ObservationSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    observation_id: str
    produced_by: list[OperatorId]
    description: str


class RuntimeVerifierSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    level: VerifierLevel
    signals: list[str] = Field(default_factory=list)


class ExternalEvaluatorSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    evaluator_id: str
    visibility: Literal["external_only"] = "external_only"


class TaskInteractionSpec(BaseModel):
    """Task semantics only; infrastructure remains in the separate world spec."""

    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["task-interaction-v1"] = "task-interaction-v1"
    task_id: str
    objective: str
    initial_artifacts: list[InitialArtifactSpec]
    operators: list[OperatorId]
    observations: list[ObservationSpec]
    runtime_verifier: RuntimeVerifierSpec
    external_evaluator: ExternalEvaluatorSpec

    @model_validator(mode="after")
    def validate_references(self) -> TaskInteractionSpec:
        if len(self.operators) != len(set(self.operators)):
            raise ValueError("TaskInteractionSpec operators must be unique")
        allowed = set(self.operators)
        for observation in self.observations:
            unknown = set(observation.produced_by) - allowed
            if unknown:
                raise ValueError(
                    f"observation {observation.observation_id!r} references "
                    f"disallowed operators: {sorted(unknown)}"
                )
        return self


class OperatorBinding(BaseModel):
    model_config = ConfigDict(extra="forbid")
    operator_id: OperatorId
    runtime_tool: str


class OperatorRegistry:
    def __init__(self, bindings: list[OperatorBinding]) -> None:
        self._bindings = {item.operator_id: item for item in bindings}
        if len(self._bindings) != len(bindings):
            raise ValueError("operator bindings must be unique")

    def validate_task(self, task: TaskInteractionSpec) -> None:
        missing = sorted(set(task.operators) - self._bindings.keys())
        if missing:
            raise ValueError(f"not_realizable: missing MAS bindings for {missing}")

    def semantic_operator(self, runtime_tool: str) -> OperatorId:
        for binding in self._bindings.values():
            if binding.runtime_tool == runtime_tool:
                return binding.operator_id
        raise KeyError(runtime_tool)


CODE_OPERATOR_REGISTRY = OperatorRegistry(
    [
        OperatorBinding(operator_id="search_code", runtime_tool="search_code"),
        OperatorBinding(operator_id="read_file", runtime_tool="read_file"),
        OperatorBinding(operator_id="edit_file", runtime_tool="edit_file"),
        OperatorBinding(operator_id="apply_patch", runtime_tool="apply_patch"),
        OperatorBinding(
            operator_id="run_targeted_test", runtime_tool="run_targeted_test"
        ),
        OperatorBinding(operator_id="run_full_test", runtime_tool="run_full_test"),
        OperatorBinding(operator_id="invoke_model", runtime_tool="planner.llm"),
        OperatorBinding(operator_id="submit_patch", runtime_tool="submit_patch"),
    ]
)

GENERAL_OPERATOR_REGISTRY = OperatorRegistry(
    [
        OperatorBinding(operator_id="invoke_model", runtime_tool="delegate"),
        OperatorBinding(operator_id="read_artifact", runtime_tool="inspect_artifact"),
        OperatorBinding(operator_id="sample_frames", runtime_tool="worker.sample_frames"),
        OperatorBinding(
            operator_id="make_contact_sheet", runtime_tool="worker.make_contact_sheet"
        ),
        OperatorBinding(operator_id="extract_clip", runtime_tool="worker.extract_clip"),
        OperatorBinding(
            operator_id="process_local_artifact", runtime_tool="process_local_artifact"
        ),
        OperatorBinding(
            operator_id="aggregate_artifacts", runtime_tool="worker.aggregate_artifacts"
        ),
        OperatorBinding(operator_id="bm25_retrieve", runtime_tool="worker.bm25_retrieve"),
        OperatorBinding(operator_id="filter_records", runtime_tool="worker.filter_records"),
        OperatorBinding(operator_id="select_fields", runtime_tool="worker.select_fields"),
        OperatorBinding(
            operator_id="aggregate_records", runtime_tool="worker.aggregate_records"
        ),
        OperatorBinding(operator_id="derive_fields", runtime_tool="worker.derive_fields"),
        OperatorBinding(operator_id="top_k_records", runtime_tool="worker.top_k_records"),
    ]
)
