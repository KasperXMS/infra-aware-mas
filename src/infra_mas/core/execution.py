"""MAS-level execution models."""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from infra_mas.core.artifact import ArtifactRef

NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
NonNegativeFloat = Annotated[float, Field(ge=0, allow_inf_nan=False)]


class ExecutionRequest(BaseModel):
    """Request semantic work without binding it to a physical executor."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    request_id: NonEmptyString
    agent: NonEmptyString
    model_id: NonEmptyString
    capability: NonEmptyString
    instructions: NonEmptyString
    task: NonEmptyString
    inputs: list[ArtifactRef]


class InvocationSpec(BaseModel):
    """Describe one planner-created logical model invocation."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    model_id: NonEmptyString
    role: NonEmptyString
    instructions: NonEmptyString
    task: NonEmptyString
    input_artifacts: list[ArtifactRef]
    semantic_operator: Literal["invoke_model", "process_local_artifact"] = "invoke_model"


class ExecutionResult(BaseModel):
    """Record the artifacts and timings produced by physical execution."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    request_id: NonEmptyString
    executor_id: NonEmptyString
    output_artifacts: list[ArtifactRef]
    queue_ms: NonNegativeFloat = Field(
        description=(
            "Observed scheduler/worker queue time; zero when it cannot be observed separately"
        )
    )
    service_ms: NonNegativeFloat = Field(
        description=(
            "Elapsed model service time, including model-server queueing, inference, "
            "and RPC latency"
        )
    )
    transfer_ms: NonNegativeFloat = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    """Represent the worker health endpoint response."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ok"] = "ok"


class WorkerStatus(BaseModel):
    """Expose static worker identity and locally hosted executors."""

    model_config = ConfigDict(extra="forbid")

    worker_id: NonEmptyString
    executors: list[NonEmptyString]
    operators: list[NonEmptyString] = Field(default_factory=list)


class TransferResult(BaseModel):
    """Measure one explicit data-plane artifact transfer."""

    model_config = ConfigDict(extra="forbid")

    bytes_transferred: Annotated[int, Field(ge=0)]
    transfer_ms: NonNegativeFloat


class ArtifactPullRequest(BaseModel):
    """Instruct a target Worker to pull an artifact directly from a source Worker."""

    model_config = ConfigDict(extra="forbid")

    artifact: ArtifactRef
    source_worker_id: NonEmptyString
    source_endpoint: NonEmptyString
    bandwidth_mbps: Annotated[float, Field(gt=0, allow_inf_nan=False)] | None = None
    rtt_ms: NonNegativeFloat = 0.0


class SampleFramesRequest(BaseModel):
    """Uniformly sample a video into one contact-sheet artifact on a Worker."""

    model_config = ConfigDict(extra="forbid")

    request_id: NonEmptyString
    input_artifact: ArtifactRef
    output_artifact_id: NonEmptyString
    duration_s: Annotated[float, Field(gt=0, allow_inf_nan=False)] | None = None
    sample_count: Annotated[int, Field(ge=1, le=64)] = 20
    columns: Annotated[int, Field(ge=1, le=16)] = 5
    frame_width: Annotated[int, Field(ge=64, le=1920)] = 448


class MakeContactSheetRequest(BaseModel):
    """Compose image artifacts into one labelled contact sheet on a Worker."""

    model_config = ConfigDict(extra="forbid")

    request_id: NonEmptyString
    input_artifacts: Annotated[list[ArtifactRef], Field(min_length=1, max_length=64)]
    output_artifact_id: NonEmptyString
    columns: Annotated[int, Field(ge=1, le=16)] = 5
    duration_s: Annotated[float, Field(gt=0, allow_inf_nan=False)] | None = None


class ExtractClipRequest(BaseModel):
    """Extract a fixed time interval from one video artifact on a Worker."""

    model_config = ConfigDict(extra="forbid")

    request_id: NonEmptyString
    input_artifact: ArtifactRef
    output_artifact_id: NonEmptyString
    start_s: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    end_s: Annotated[float, Field(gt=0, allow_inf_nan=False)]

    @model_validator(mode="after")
    def validate_interval(self) -> "ExtractClipRequest":
        if self.end_s <= self.start_s:
            raise ValueError("extract_clip end_s must be greater than start_s")
        return self

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


class AggregateArtifactsRequest(BaseModel):
    """Combine textual semantic evidence into one structured artifact."""

    model_config = ConfigDict(extra="forbid")

    request_id: NonEmptyString
    input_artifacts: Annotated[list[ArtifactRef], Field(min_length=1, max_length=128)]
    output_artifact_id: NonEmptyString


class BM25RetrieveRequest(BaseModel):
    """Rank one Worker-local text shard with deterministic BM25."""

    model_config = ConfigDict(extra="forbid")

    request_id: NonEmptyString
    query: NonEmptyString
    input_artifacts: Annotated[list[ArtifactRef], Field(min_length=1, max_length=128)]
    output_artifact_id: NonEmptyString
    top_k: Annotated[int, Field(ge=1, le=50)] = 3
    k1: Annotated[float, Field(gt=0, allow_inf_nan=False)] = 1.5
    b: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)] = 0.75


RecordScalar = str | int | float | bool | None
RecordPredicateOperator = Literal[
    "eq",
    "ne",
    "lt",
    "lte",
    "gt",
    "gte",
    "contains",
    "not_contains",
    "in",
    "not_in",
    "is_null",
    "not_null",
]


class RecordPredicate(BaseModel):
    """Describe one dataset-neutral predicate over a dotted record field path."""

    model_config = ConfigDict(extra="forbid")

    field: NonEmptyString
    operator: RecordPredicateOperator
    value: RecordScalar | list[RecordScalar] = None

    @model_validator(mode="after")
    def validate_value_shape(self) -> "RecordPredicate":
        if self.operator in {"in", "not_in"} and not isinstance(self.value, list):
            raise ValueError(f"{self.operator} requires a list value")
        if self.operator in {"is_null", "not_null"} and self.value is not None:
            raise ValueError(f"{self.operator} does not accept a value")
        return self


class FilterRecordsRequest(BaseModel):
    """Filter generic JSON records without dataset-specific logic."""

    model_config = ConfigDict(extra="forbid")

    request_id: NonEmptyString
    input_artifacts: Annotated[list[ArtifactRef], Field(min_length=1, max_length=128)]
    output_artifact_id: NonEmptyString
    predicates: Annotated[list[RecordPredicate], Field(min_length=1, max_length=32)]
    match: Literal["all", "any"] = "all"


class SelectFieldsRequest(BaseModel):
    """Project generic JSON records onto an explicit set of field paths."""

    model_config = ConfigDict(extra="forbid")

    request_id: NonEmptyString
    input_artifacts: Annotated[list[ArtifactRef], Field(min_length=1, max_length=128)]
    output_artifact_id: NonEmptyString
    fields: Annotated[list[NonEmptyString], Field(min_length=1, max_length=64)]

    @model_validator(mode="after")
    def validate_unique_fields(self) -> "SelectFieldsRequest":
        if len(self.fields) != len(set(self.fields)):
            raise ValueError("select_fields fields must be unique")
        return self


RecordAggregationOperation = Literal[
    "count",
    "count_distinct",
    "sum",
    "min",
    "max",
    "mean",
]


class RecordAggregation(BaseModel):
    """Describe one generic aggregation and its output field."""

    model_config = ConfigDict(extra="forbid")

    output_field: NonEmptyString
    operation: RecordAggregationOperation
    field: NonEmptyString | None = None

    @model_validator(mode="after")
    def validate_field(self) -> "RecordAggregation":
        if self.operation != "count" and self.field is None:
            raise ValueError(f"{self.operation} requires a field")
        return self


class AggregateRecordsRequest(BaseModel):
    """Group and aggregate generic JSON records deterministically."""

    model_config = ConfigDict(extra="forbid")

    request_id: NonEmptyString
    input_artifacts: Annotated[list[ArtifactRef], Field(min_length=1, max_length=128)]
    output_artifact_id: NonEmptyString
    aggregations: Annotated[list[RecordAggregation], Field(min_length=1, max_length=32)]
    group_by: Annotated[list[NonEmptyString], Field(max_length=16)] = Field(
        default_factory=list
    )

    @model_validator(mode="after")
    def validate_unique_outputs(self) -> "AggregateRecordsRequest":
        output_fields = [item.output_field for item in self.aggregations]
        if len(output_fields) != len(set(output_fields)):
            raise ValueError("aggregate_records output fields must be unique")
        if len(self.group_by) != len(set(self.group_by)):
            raise ValueError("aggregate_records group_by fields must be unique")
        overlap = set(output_fields) & set(self.group_by)
        if overlap:
            raise ValueError(
                "aggregate_records output fields overlap group_by fields: "
                f"{sorted(overlap)}"
            )
        return self


class RecordDerivation(BaseModel):
    """Describe safe field-to-field arithmetic without arbitrary code evaluation."""

    model_config = ConfigDict(extra="forbid")

    output_field: NonEmptyString
    operation: Literal["add", "subtract", "multiply", "divide"]
    left_field: NonEmptyString
    right_field: NonEmptyString


class DeriveFieldsRequest(BaseModel):
    """Add deterministic arithmetic fields to generic JSON records."""

    model_config = ConfigDict(extra="forbid")

    request_id: NonEmptyString
    input_artifacts: Annotated[list[ArtifactRef], Field(min_length=1, max_length=128)]
    output_artifact_id: NonEmptyString
    derivations: Annotated[list[RecordDerivation], Field(min_length=1, max_length=32)]

    @model_validator(mode="after")
    def validate_unique_outputs(self) -> "DeriveFieldsRequest":
        outputs = [item.output_field for item in self.derivations]
        if len(outputs) != len(set(outputs)):
            raise ValueError("derive_fields output fields must be unique")
        return self


class RecordSort(BaseModel):
    """Describe deterministic ordering of one record field."""

    model_config = ConfigDict(extra="forbid")

    field: NonEmptyString
    direction: Literal["ascending", "descending"] = "descending"
    nulls: Literal["first", "last"] = "last"


class TopKRecordsRequest(BaseModel):
    """Order generic JSON records and retain a bounded prefix."""

    model_config = ConfigDict(extra="forbid")

    request_id: NonEmptyString
    input_artifacts: Annotated[list[ArtifactRef], Field(min_length=1, max_length=128)]
    output_artifact_id: NonEmptyString
    order_by: Annotated[list[RecordSort], Field(min_length=1, max_length=16)]
    limit: Annotated[int, Field(ge=1, le=10_000)]


class BindLocalArtifactRequest(BaseModel):
    """Import an allowlisted Worker-local file without crossing the control plane."""

    model_config = ConfigDict(extra="forbid")

    artifact_id: NonEmptyString
    source_path: NonEmptyString
    artifact_type: NonEmptyString
    expected_size_bytes: Annotated[int, Field(ge=0)] | None = None
