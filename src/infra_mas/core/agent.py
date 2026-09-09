"""Semantic agent models."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, StringConstraints

NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class AgentSpec(BaseModel):
    """Describe a semantic capability without physical execution details."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    name: NonEmptyString
    capability: NonEmptyString
    instructions: NonEmptyString
    model_id: NonEmptyString | None = None
