"""Logical model and model-level execution schemas."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class ModelRequest(BaseModel):
    """Describe the worker-local inputs to a model invocation."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    instructions: NonEmptyString
    task: NonEmptyString
    input_paths: list[NonEmptyString]


class ModelSpec(BaseModel):
    """Describe a deployed logical model without physical replica details."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    model_id: NonEmptyString
    description: NonEmptyString
    input_modalities: Annotated[list[NonEmptyString], Field(min_length=1)]
    output_modalities: Annotated[list[NonEmptyString], Field(min_length=1)]
    context_window: Annotated[int, Field(gt=0)]


class ModelResult(BaseModel):
    """Represent the textual result and measured model latency."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    output_text: NonEmptyString
    latency_ms: Annotated[float, Field(ge=0, allow_inf_nan=False)]
