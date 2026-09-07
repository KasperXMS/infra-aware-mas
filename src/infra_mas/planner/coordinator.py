"""Resource-blind top-level Coordinator."""

from agents import Agent, RunConfig, Runner
from agents.models.interface import Model

from infra_mas.planner.context import PlannerContext
from infra_mas.planner.prompts import build_blind_coordinator_instructions
from infra_mas.planner.tools import delegate, inspect_artifact


class Coordinator:
    """Use the OpenAI Agents SDK only for top-level semantic planning."""

    def __init__(
        self,
        context: PlannerContext,
        *,
        model: str | Model | None = None,
        max_turns: int = 10,
    ) -> None:
        if context.resource_aware:
            raise ValueError("blind Coordinator requires resource_aware=False")
        if max_turns <= 0:
            raise ValueError("max_turns must be positive")

        self._context = context
        self._max_turns = max_turns
        self._agent = Agent[PlannerContext](
            name="coordinator",
            instructions=build_blind_coordinator_instructions(context.agent_registry),
            tools=[delegate, inspect_artifact],
            model=model,
        )

    @property
    def agent(self) -> Agent[PlannerContext]:
        """Expose the configured SDK agent for inspection and testing."""
        return self._agent

    async def run(self, task: str) -> str:
        """Run dynamic semantic delegation and return the SDK final output."""
        if not task.strip():
            raise ValueError("task must not be empty")

        artifact_ids = [artifact.id for artifact in self._context.artifact_catalog.list()]
        available_artifacts = ", ".join(artifact_ids) if artifact_ids else "none"
        planner_input = f"{task}\n\nAvailable input artifact IDs: {available_artifacts}"
        await self._context.trace.start(
            {
                "planner": "openai-agents",
                "resource_aware": False,
                "agents": [agent.name for agent in self._context.agent_registry.list()],
                "input_artifacts": artifact_ids,
            }
        )

        try:
            run_result = await Runner.run(
                self._agent,
                planner_input,
                context=self._context,
                max_turns=self._max_turns,
                run_config=RunConfig(
                    tracing_disabled=True,
                    workflow_name="Infra-Aware MAS Blind Coordinator",
                ),
            )
            final_output: object = run_result.final_output
            if not isinstance(final_output, str) or not final_output.strip():
                raise ValueError("Coordinator returned an empty or non-text final output")
        except Exception as error:
            action_id = await self._context.next_action_id("finish")
            await self._context.trace.record(
                "planner.finish",
                action_id=action_id,
                parent_action_id=self._context.coordinator_action_id,
                success=False,
                error_type=type(error).__name__,
                error=str(error),
            )
            await self._context.trace.end(
                {
                    "success": False,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            raise

        action_id = await self._context.next_action_id("finish")
        await self._context.trace.record(
            "planner.finish",
            action_id=action_id,
            parent_action_id=self._context.coordinator_action_id,
            success=True,
        )
        await self._context.trace.end({"success": True, "answer": final_output})
        return final_output
