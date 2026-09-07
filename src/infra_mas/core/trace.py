"""Trace event models and recording interface."""

from datetime import datetime
from typing import Annotated, Protocol

from pydantic import BaseModel, ConfigDict, StringConstraints

NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class TraceEvent(BaseModel):
    """Represent one extensible, top-level JSONL trace event."""

    model_config = ConfigDict(extra="allow")

    event_type: NonEmptyString
    run_id: NonEmptyString
    timestamp: datetime
    action_id: NonEmptyString | None = None
    parent_action_id: NonEmptyString | None = None


class TraceSink(Protocol):
    """Accept structured events without exposing trace storage details."""

    @property
    def run_id(self) -> str:
        """Return the run identifier attached to recorded events."""
        ...

    async def record(
        self,
        event_type: str,
        *,
        action_id: str | None = None,
        parent_action_id: str | None = None,
        **fields: object,
    ) -> None:
        """Record one structured trace event."""
        ...
