"""Deterministic, workflow-agnostic planning history."""

import asyncio
from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict


class InitialInputUsage(BaseModel):
    """Track how many completed invocations directly used one initial input."""

    model_config = ConfigDict(extra="forbid")

    artifact_id: str
    use_count: int


class CompletedInvocation(BaseModel):
    """Describe one completed logical model invocation."""

    model_config = ConfigDict(extra="forbid")

    action_id: str
    model_id: str
    role: str
    task: str
    input_artifact_ids: list[str]
    output_artifact_ids: list[str]


class PlanningState(BaseModel):
    """Serializable deterministic view of the current planning history."""

    model_config = ConfigDict(extra="forbid")

    initial_inputs: list[InitialInputUsage]
    unused_initial_inputs: list[str]
    completed_invocations: list[CompletedInvocation]


class PlanningLedger:
    """Record completed invocations without interpreting workflow intent."""

    def __init__(self, initial_input_artifact_ids: Iterable[str]) -> None:
        initial_ids = list(initial_input_artifact_ids)
        if len(initial_ids) != len(set(initial_ids)):
            raise ValueError("initial input artifact IDs must be unique")
        self._initial_ids = initial_ids
        self._completed: dict[str, CompletedInvocation] = {}
        self._lock = asyncio.Lock()

    async def record_completed(
        self,
        invocation: CompletedInvocation,
    ) -> PlanningState:
        """Record one completion and return the resulting state snapshot."""
        async with self._lock:
            if invocation.action_id in self._completed:
                raise ValueError(
                    f"completed invocation already recorded: {invocation.action_id}"
                )
            self._completed[invocation.action_id] = invocation.model_copy(deep=True)
            return self._snapshot_unlocked()

    async def snapshot(self) -> PlanningState:
        """Return a copy ordered by deterministic action sequence."""
        async with self._lock:
            return self._snapshot_unlocked()

    def _snapshot_unlocked(self) -> PlanningState:
        completed = sorted(
            self._completed.values(),
            key=lambda invocation: self._action_sort_key(invocation.action_id),
        )
        use_counts = dict.fromkeys(self._initial_ids, 0)
        for invocation in completed:
            for artifact_id in set(invocation.input_artifact_ids):
                if artifact_id in use_counts:
                    use_counts[artifact_id] += 1
        initial_inputs = [
            InitialInputUsage(artifact_id=artifact_id, use_count=use_counts[artifact_id])
            for artifact_id in self._initial_ids
        ]
        return PlanningState(
            initial_inputs=initial_inputs,
            unused_initial_inputs=[
                item.artifact_id for item in initial_inputs if item.use_count == 0
            ],
            completed_invocations=[item.model_copy(deep=True) for item in completed],
        )

    @staticmethod
    def _action_sort_key(action_id: str) -> tuple[int, str]:
        suffix = action_id.rpartition("-")[2]
        try:
            return int(suffix), action_id
        except ValueError:
            return 0, action_id
