"""Deterministic scheduler for repository-local code tools."""

from infra_mas.code_tasks.models import CodeExecutor, RepositoryWorld


class RepositoryLocalityScheduler:
    """Bind every tool to a compatible executor at the repository's current site."""

    def __init__(self, executors: list[CodeExecutor]) -> None:
        if not executors:
            raise ValueError("at least one code executor is required")
        ids = [item.executor_id for item in executors]
        if len(ids) != len(set(ids)):
            raise ValueError("code executor IDs must be unique")
        self._executors = [item.model_copy(deep=True) for item in executors]

    def select(self, world: RepositoryWorld) -> CodeExecutor:
        """Select the repository-local endpoint, breaking ties by executor ID."""
        candidates = sorted(
            (item for item in self._executors if item.site == world.repository_site),
            key=lambda item: item.executor_id,
        )
        if not candidates:
            raise ValueError(
                f"no code executor is available at repository site {world.repository_site!r}"
            )
        return candidates[0].model_copy(deep=True)

    def list(self) -> list[CodeExecutor]:
        return [item.model_copy(deep=True) for item in self._executors]
