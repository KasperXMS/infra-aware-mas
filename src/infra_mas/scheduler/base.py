"""Scheduler interface."""

from typing import Protocol

from infra_mas.core.execution import InvocationSpec
from infra_mas.core.executor import ExecutorSpec


class Scheduler(Protocol):
    """Select a physical executor for one semantic execution request."""

    async def select(self, invocation: InvocationSpec) -> ExecutorSpec:
        """Return one compatible executor without modifying the request."""
        ...

    def preset_model_id(self, capability: str) -> str:
        """Resolve a legacy agent capability to one logical model ID."""
        ...
