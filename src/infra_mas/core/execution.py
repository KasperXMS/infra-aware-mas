"""MAS-level execution models."""

from typing import Annotated, Any

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
