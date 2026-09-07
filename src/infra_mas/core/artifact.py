"""Artifact reference models."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class ArtifactRef(BaseModel):
    """Reference an artifact without placing its contents on the control plane."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    id: NonEmptyString
    artifact_type: NonEmptyString
    size_bytes: Annotated[int, Field(ge=0)]
    locations: Annotated[list[NonEmptyString], Field(min_length=1)]

    @field_validator("locations")
    @classmethod
    def locations_must_be_unique(cls, locations: list[str]) -> list[str]:
        """Reject ambiguous duplicate location metadata."""
        if len(locations) != len(set(locations)):
            raise ValueError("artifact locations must be unique")
        return locations
