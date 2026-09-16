"""Infrastructure resource models shared by providers and schedulers."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class ArtifactPlacement(BaseModel):
    """Describe one artifact's size and currently known Worker locations."""

    model_config = ConfigDict(extra="forbid")

    artifact_id: NonEmptyString
    size_bytes: Annotated[int, Field(ge=0)]
    locations: Annotated[list[NonEmptyString], Field(min_length=1)]

    @field_validator("locations")
    @classmethod
    def unique_locations(cls, locations: list[str]) -> list[str]:
        if len(locations) != len(set(locations)):
            raise ValueError("artifact locations must be unique")
        return locations


class ModelReplicaPlacement(BaseModel):
    """Expose a logical model's physical replica placement."""

    model_config = ConfigDict(extra="forbid")

    model_id: NonEmptyString
    executor_id: NonEmptyString
    worker_id: NonEmptyString
    site: NonEmptyString


class ServiceTimeEstimate(BaseModel):
    """Record a measured or static per-invocation service-time estimate."""

    model_config = ConfigDict(extra="forbid")

    executor_id: NonEmptyString
    model_id: NonEmptyString
    service_ms: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    source: Literal["measured", "static"] = "static"


class NetworkLink(BaseModel):
    """Describe an estimated network path between two sites."""

    model_config = ConfigDict(extra="forbid")

    source_site: NonEmptyString
    target_site: NonEmptyString
    bandwidth_mbps: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    rtt_ms: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    bidirectional: bool = True
