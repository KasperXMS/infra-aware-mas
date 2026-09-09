"""Round-robin executor scheduler."""

import asyncio

from infra_mas.core.errors import NoExecutorAvailableError
from infra_mas.core.execution import InvocationSpec
from infra_mas.core.executor import ExecutorSpec
from infra_mas.execution.executor_registry import ExecutorRegistry


class RoundRobinScheduler:
    """Rotate deterministically through physical replicas of each logical model."""

    def __init__(self, registry: ExecutorRegistry) -> None:
        self._registry = registry
        self._next_indices: dict[str, int] = {}
        self._lock = asyncio.Lock()

    async def select(self, invocation: InvocationSpec) -> ExecutorSpec:
        """Select the next compatible executor safely under async concurrency."""
        candidates = self._registry.model_candidates(invocation.model_id)
        if not candidates:
            raise NoExecutorAvailableError(
                f"no executor serves model {invocation.model_id!r}"
            )

        async with self._lock:
            index = self._next_indices.get(invocation.model_id, 0)
            selected = candidates[index % len(candidates)]
            self._next_indices[invocation.model_id] = (index + 1) % len(candidates)
        return selected

    def preset_model_id(self, capability: str) -> str:
        """Resolve a legacy capability when it identifies exactly one model."""
        models = {executor.model_id for executor in self._registry.candidates(capability)}
        if len(models) != 1:
            raise NoExecutorAvailableError(
                f"cannot resolve one model for preset capability {capability!r}"
            )
        return models.pop()
