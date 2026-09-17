"""Per-turn Planner tracing for code tasks."""

from dataclasses import dataclass
from time import perf_counter

from agents import Agent, RunContextWrapper, RunHooks
from agents.items import ModelResponse, TResponseInputItem

from infra_mas.code_tasks.context import CodePlannerContext


@dataclass(frozen=True, slots=True)
class _ActiveCall:
    action_id: str
    turn: int
    started_at: float


class CodePlannerTraceHooks(RunHooks[CodePlannerContext]):
    def __init__(self, context: CodePlannerContext) -> None:
        self._context = context
        self._turn_count = 0
        self._active: _ActiveCall | None = None

    @property
    def turn_count(self) -> int:
        return self._turn_count

    async def on_llm_start(
        self,
        context: RunContextWrapper[CodePlannerContext],
        agent: Agent[CodePlannerContext],
        system_prompt: str | None,
        input_items: list[TResponseInputItem],
    ) -> None:
        del context, agent, system_prompt, input_items
        self._turn_count += 1
        action_id = await self._context.next_action_id("llm")
        self._active = _ActiveCall(action_id, self._turn_count, perf_counter())
        await self._context.trace.record(
            "planner.llm.start",
            action_id=action_id,
            parent_action_id=self._context.coordinator_action_id,
            turn=self._turn_count,
            semantic_operator="invoke_model",
            tool="planner.llm",
            input_artifacts=self._context.consume_observations(),
        )

    async def on_llm_end(
        self,
        context: RunContextWrapper[CodePlannerContext],
        agent: Agent[CodePlannerContext],
        response: ModelResponse,
    ) -> None:
        del context, agent
        active = self._active
        if active is None:
            return
        self._active = None
        await self._context.trace.record(
            "planner.llm.end",
            action_id=active.action_id,
            parent_action_id=self._context.coordinator_action_id,
            turn=active.turn,
            semantic_operator="invoke_model",
            tool="planner.llm",
            output_artifacts=[f"{active.action_id}/reasoning"],
            latency_ms=(perf_counter() - active.started_at) * 1000,
            token_usage={
                "requests": response.usage.requests,
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.total_tokens,
            },
            success=True,
        )
        self._context.record_reasoning_artifact(f"{active.action_id}/reasoning")

    async def finish_pending(self, error: Exception) -> None:
        active = self._active
        if active is None:
            return
        self._active = None
        await self._context.trace.record(
            "planner.llm.end",
            action_id=active.action_id,
            parent_action_id=self._context.coordinator_action_id,
            turn=active.turn,
            semantic_operator="invoke_model",
            tool="planner.llm",
            output_artifacts=[],
            latency_ms=(perf_counter() - active.started_at) * 1000,
            token_usage=None,
            success=False,
            error_type=type(error).__name__,
            error=str(error),
        )
