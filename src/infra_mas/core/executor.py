"""Physical executor models."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, StringConstraints

NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class ExecutorSpec(BaseModel):
    """Describe a statically configured physical execution target."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    id: NonEmptyString
    capability: NonEmptyString
    worker_id: NonEmptyString
    model: NonEmptyString
    device: NonEmptyString
    site: NonEmptyString
