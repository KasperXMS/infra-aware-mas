"""Model-level execution schemas."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class ModelRequest(BaseModel):
    """Describe the worker-local inputs to a model invocation."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    task: NonEmptyString
    input_paths: list[NonEmptyString]


class ModelResult(BaseModel):
    """Represent the textual result and measured model latency."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    output_text: NonEmptyString
    latency_ms: Annotated[float, Field(ge=0, allow_inf_nan=False)]
