"""Fixed executor scheduler."""

from collections.abc import Mapping

from infra_mas.core.errors import NoExecutorAvailableError
from infra_mas.core.execution import InvocationSpec
from infra_mas.core.executor import ExecutorSpec
from infra_mas.execution.executor_registry import ExecutorRegistry


class FixedScheduler:
    """Map each logical model to one statically selected physical replica."""

    def __init__(
        self,
        registry: ExecutorRegistry,
        assignments: Mapping[str, str],
    ) -> None:
        self._registry = registry
        self._assignments: dict[str, str] = {}
        self._legacy_capability_models: dict[str, str] = {}
        for mapping_key, executor_id in assignments.items():
            executor = registry.get(executor_id)
            if mapping_key == executor.model_id:
                self._assignments[mapping_key] = executor_id
            elif mapping_key == executor.capability:
                self._assignments[executor.model_id] = executor_id
                self._legacy_capability_models[mapping_key] = executor.model_id
            else:
                raise ValueError(
                    f"executor {executor_id!r} does not serve mapped model {mapping_key!r}"
                )

    async def select(self, invocation: InvocationSpec) -> ExecutorSpec:
        """Return the executor configured for the invocation's logical model."""
        executor_id = self._assignments.get(invocation.model_id)
        if executor_id is None:
            raise NoExecutorAvailableError(
                f"no fixed executor configured for model {invocation.model_id!r}"
            )
        return self._registry.get(executor_id)

    def preset_model_id(self, capability: str) -> str:
        """Resolve an old capability assignment during preset conversion only."""
        model_id = self._legacy_capability_models.get(capability)
        if model_id is not None:
            return model_id
        models = {
            self._registry.get(executor_id).model_id
            for executor_id in self._assignments.values()
            if self._registry.get(executor_id).capability == capability
        }
        if len(models) != 1:
            raise NoExecutorAvailableError(
                f"cannot resolve one model for preset capability {capability!r}"
            )
        return models.pop()
