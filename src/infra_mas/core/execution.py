"""MAS-level execution models."""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from infra_mas.core.artifact import ArtifactRef

NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
NonNegativeFloat = Annotated[float, Field(ge=0, allow_inf_nan=False)]


class ExecutionRequest(BaseModel):
    """Request semantic work without binding it to a physical executor."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    request_id: NonEmptyString
    agent: NonEmptyString
    capability: NonEmptyString
    task: NonEmptyString
    inputs: list[ArtifactRef]


class ExecutionResult(BaseModel):
    """Record the artifacts and timings produced by physical execution."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    request_id: NonEmptyString
    executor_id: NonEmptyString
    output_artifacts: list[ArtifactRef]
    queue_ms: NonNegativeFloat
    compute_ms: NonNegativeFloat
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


class TransferResult(BaseModel):
    """Measure one explicit data-plane artifact transfer."""

    model_config = ConfigDict(extra="forbid")

    bytes_transferred: Annotated[int, Field(ge=0)]
    transfer_ms: NonNegativeFloat
