"""Deterministic locality-aware physical executor scheduler."""

from infra_mas.core.errors import NoExecutorAvailableError
from infra_mas.core.execution import InvocationSpec
from infra_mas.core.executor import ExecutorSpec
from infra_mas.execution.executor_registry import ExecutorRegistry
from infra_mas.resources.provider import ResourceProvider


class ResourceAwareScheduler:
    """Choose the compatible replica requiring the least input transfer.

    The scheduler is deliberately independent of Planner infrastructure visibility and is
    therefore shared by blind and snapshot-visible experiment arms.
    """

    def __init__(
        self,
        registry: ExecutorRegistry,
        resource_provider: ResourceProvider,
    ) -> None:
        self._registry = registry
        self._resource_provider = resource_provider

    async def select(self, invocation: InvocationSpec) -> ExecutorSpec:
        """Minimize estimated required input transfer, then executor ID."""
        candidates = self._registry.model_candidates(invocation.model_id)
        if not candidates:
            raise NoExecutorAvailableError(
                f"no executor serves model {invocation.model_id!r}"
            )
        ranked = sorted(
            (
                self._resource_provider.estimate_input_transfer_ms(
                    invocation.input_artifacts,
                    executor,
                ),
                executor.id,
                executor,
            )
            for executor in candidates
        )
        transfer_ms, _, selected = ranked[0]
        if transfer_ms == float("inf"):
            raise NoExecutorAvailableError(
                f"no reachable replica serves model {invocation.model_id!r}"
            )
        return selected

    def preset_model_id(self, capability: str) -> str:
        """Resolve a legacy capability when it identifies exactly one logical model."""
        models = {executor.model_id for executor in self._registry.candidates(capability)}
        if len(models) != 1:
            raise NoExecutorAvailableError(
                f"cannot resolve one model for preset capability {capability!r}"
            )
        return models.pop()
