"""Fixed executor scheduler."""

from collections.abc import Mapping

from infra_mas.core.errors import NoExecutorAvailableError
from infra_mas.core.execution import ExecutionRequest
from infra_mas.core.executor import ExecutorSpec
from infra_mas.execution.executor_registry import ExecutorRegistry


class FixedScheduler:
    """Map each capability to one statically selected executor."""

    def __init__(
        self,
        registry: ExecutorRegistry,
        assignments: Mapping[str, str],
    ) -> None:
        self._registry = registry
        self._assignments = dict(assignments)
        for capability, executor_id in self._assignments.items():
            executor = registry.get(executor_id)
            if executor.capability != capability:
                raise ValueError(
                    f"executor {executor_id!r} does not support mapped capability {capability!r}"
                )

    async def select(self, request: ExecutionRequest) -> ExecutorSpec:
        """Return the executor configured for the request capability."""
        executor_id = self._assignments.get(request.capability)
        if executor_id is None:
            raise NoExecutorAvailableError(
                f"no fixed executor configured for capability {request.capability!r}"
            )
        return self._registry.get(executor_id)
