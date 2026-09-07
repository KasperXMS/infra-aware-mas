"""Scheduler interface."""

from typing import Protocol

from infra_mas.core.execution import ExecutionRequest
from infra_mas.core.executor import ExecutorSpec


class Scheduler(Protocol):
    """Select a physical executor for one semantic execution request."""

    async def select(self, request: ExecutionRequest) -> ExecutorSpec:
        """Return one compatible executor without modifying the request."""
        ...
