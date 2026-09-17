"""Open-ended code-task Coordinator using the existing Planner model interface."""

from agents import Agent, RunConfig, Runner, Tool
from agents.model_settings import ModelSettings
from agents.models.interface import Model

from infra_mas.code_tasks.context import CodePlannerContext
from infra_mas.code_tasks.tools import (
    apply_patch,
    edit_file,
    read_file,
    run_full_test,
    run_targeted_test,
    search_code,
    submit_patch,
)
from infra_mas.code_tasks.trace_hooks import CodePlannerTraceHooks


def build_code_planner_instructions() -> str:
    """Return world-invariant instructions shared by static and snapshot arms."""
    return """You are an open-ended software engineering Planner. Solve the reported issue in the
checked-out repository using only the available generic tools. Search and read code as needed,
make a minimal production-quality edit with edit_file or apply_patch, run useful tests when
feasible, and call submit_patch
exactly once after the repository contains your final solution. Do not merely describe a patch.

There is no predefined workflow, file list, patch, test sequence, or required call count. Plan
adaptively from the issue and tool results. Tool execution has resource and context-transfer cost:
avoid substantially redundant searches, reads, and tests, but do not sacrifice correctness.
Treat concrete issue evidence such as stack frames, failing symbols, and reproduction snippets as
high-priority evidence. Once a minimal causal hypothesis is supported, edit and validate it instead
of continuing a broad audit without a concrete unresolved question.
Physical executor selection is exclusively the scheduler's responsibility. Infrastructure facts,
when present, are observations rather than workflow recommendations. Never ask for benchmark
answers, evaluator configuration, candidate workflows, or oracle results."""


class CodeTaskCoordinator:
    def __init__(
        self,
        context: CodePlannerContext,
        *,
        model: str | Model | None = None,
        max_turns: int = 20,
        max_output_tokens: int = 4096,
    ) -> None:
        self._context = context
        self._max_turns = max_turns
        self._hooks = CodePlannerTraceHooks(context)
        tools: list[Tool] = [
            search_code,
            read_file,
            edit_file,
            apply_patch,
            run_targeted_test,
            run_full_test,
            submit_patch,
        ]
        self._agent = Agent[CodePlannerContext](
            name="code-task-coordinator",
            instructions=build_code_planner_instructions(),
            tools=tools,
            model=model,
            model_settings=ModelSettings(max_tokens=max_output_tokens),
        )

    @property
    def agent(self) -> Agent[CodePlannerContext]:
        return self._agent

    async def run(self, task: str) -> str:
        planner_input = f"Software issue:\n{task}\n\n{self._context.render_static_context()}"
        dynamic_snapshot: dict[str, object] | None = None
        if self._context.visibility == "snapshot":
            planner_input += f"\n\n{self._context.render_dynamic_context()}"
            dynamic_snapshot = self._context.world.model_dump(mode="json")
        await self._context.trace.record(
            "planner.infrastructure_context",
            infrastructure_visibility=self._context.visibility,
            static_context=self._context.render_static_context(),
            dynamic_snapshot=dynamic_snapshot,
        )
        try:
            result = await Runner.run(
                self._agent,
                planner_input,
                context=self._context,
                max_turns=self._max_turns,
                hooks=self._hooks,
                run_config=RunConfig(
                    tracing_disabled=True,
                    workflow_name="Infra-Aware MAS SWE-bench Coordinator",
                ),
            )
            output: object = result.final_output
            if not isinstance(output, str) or not output.strip():
                raise ValueError("Code-task Coordinator returned an empty final output")
        except Exception as error:
            await self._hooks.finish_pending(error)
            raise
        return output
