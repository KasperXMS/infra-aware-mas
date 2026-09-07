"""Round-robin executor scheduler."""

import asyncio

from infra_mas.core.errors import NoExecutorAvailableError
from infra_mas.core.execution import ExecutionRequest
from infra_mas.core.executor import ExecutorSpec
from infra_mas.execution.executor_registry import ExecutorRegistry


class RoundRobinScheduler:
    """Rotate deterministically through compatible executors per capability."""

    def __init__(self, registry: ExecutorRegistry) -> None:
        self._registry = registry
        self._next_indices: dict[str, int] = {}
        self._lock = asyncio.Lock()

    async def select(self, request: ExecutionRequest) -> ExecutorSpec:
        """Select the next compatible executor safely under async concurrency."""
        candidates = self._registry.candidates(request.capability)
        if not candidates:
            raise NoExecutorAvailableError(
                f"no executor supports capability {request.capability!r}"
            )

        async with self._lock:
            index = self._next_indices.get(request.capability, 0)
            selected = candidates[index % len(candidates)]
            self._next_indices[request.capability] = (index + 1) % len(candidates)
        return selected
