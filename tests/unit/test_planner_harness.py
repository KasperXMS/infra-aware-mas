"""Tests for minimal, stateful, and efficient Planner harnesses."""

import json
from pathlib import Path
from typing import cast

from infra_mas.core.agent import AgentSpec
from infra_mas.core.artifact import ArtifactRef
from infra_mas.core.execution import ExecutionResult, InvocationSpec
from infra_mas.core.model import ModelSpec
from infra_mas.planner.context import ArtifactCatalog, PlannerContext, PlannerHarness
from infra_mas.planner.prompts import build_blind_coordinator_instructions
from infra_mas.planner.tools import execute_delegation, execute_dynamic_invocation
from infra_mas.runtime.agent_registry import AgentRegistry
from infra_mas.runtime.model_registry import ModelRegistry
from infra_mas.runtime.runtime import AgentRuntime
from infra_mas.tracing.recorder import TraceRecorder


class StubRuntime:
    """Return deterministic artifacts while capturing unified invocations."""

    def __init__(self) -> None:
        self.invocations: list[InvocationSpec] = []

    def create_preset_invocation(
        self,
        agent_name: str,
        task: str,
        inputs: list[ArtifactRef],
    ) -> InvocationSpec:
        return InvocationSpec(
            model_id="test-model",
            role=agent_name,
            instructions="Use the preset instructions.",
            task=task,
            input_artifacts=inputs,
        )

    async def invoke(
        self,
        invocation: InvocationSpec,
        *,
        parent_action_id: str | None = None,
    ) -> ExecutionResult:
        del parent_action_id
        self.invocations.append(invocation)
        sequence = len(self.invocations)
        return ExecutionResult(
            request_id=f"request-{sequence}",
            executor_id="executor",
            output_artifacts=[
                ArtifactRef(
                    id=f"run/output-{sequence}",
                    artifact_type="application/octet-stream",
                    size_bytes=1,
                    locations=["worker"],
                )
            ],
            queue_ms=0,
            service_ms=1,
        )


def model_registry() -> ModelRegistry:
    return ModelRegistry(
        [
            ModelSpec(
                model_id="test-model",
                description="General test model.",
                input_modalities=["text"],
                output_modalities=["text"],
                context_window=8192,
            )
        ]
    )


def build_context(
    tmp_path: Path,
    harness: PlannerHarness,
    *,
    static_agents: bool = False,
) -> tuple[PlannerContext, StubRuntime]:
    trace = TraceRecorder(tmp_path / harness, "run")
    catalog = ArtifactCatalog({}, tmp_path / harness / "inspection", trace)
    catalog.register_many(
        [
            ArtifactRef(
                id="run/input-a",
                artifact_type="text/plain",
                size_bytes=1,
                locations=["worker"],
            ),
            ArtifactRef(
                id="run/input-b",
                artifact_type="text/plain",
                size_bytes=1,
                locations=["worker"],
            ),
        ]
    )
    runtime = StubRuntime()
    agents = (
        AgentRegistry(
            [
                AgentSpec(
                    name="preset",
                    capability="reasoning",
                    instructions="Use the preset instructions.",
                    model_id="test-model",
                )
            ]
        )
        if static_agents
        else None
    )
    context = PlannerContext(
        cast(AgentRuntime, runtime),
        agents,
        catalog,
        trace,
        model_registry=model_registry(),
        planner_mode="static_agents" if static_agents else "dynamic_models",
        planner_harness=harness,
    )
    return context, runtime


async def test_minimal_tool_result_preserves_existing_shape(tmp_path: Path) -> None:
    context, _ = build_context(tmp_path, "minimal")

    result = await execute_dynamic_invocation(
        context,
        "test-model",
        "novel_role",
        "Do focused work.",
        "Analyze the input.",
        ["run/input-a"],
    )

    payload = json.loads(result.model_dump_json(exclude_none=True))
    assert payload == {
        "agent": "novel_role",
        "output_artifact_ids": ["run/output-1"],
        "output_text": "Artifact run/output-1 is available for further delegation.",
    }


async def test_stateful_result_tracks_input_usage_and_history(tmp_path: Path) -> None:
    context, _ = build_context(tmp_path, "stateful")

    first = await execute_dynamic_invocation(
        context,
        "test-model",
        "novel_role",
        "Do focused work.",
        "Analyze the first input.",
        ["run/input-a"],
    )
    second = await execute_dynamic_invocation(
        context,
        "test-model",
        "another_novel_role",
        "Check the available evidence.",
        "Review prior work with the same input.",
        ["run/input-a", "run/output-1"],
    )

    assert first.planning_state is not None
    assert first.planning_state.unused_initial_inputs == ["run/input-b"]
    assert second.planning_state is not None
    assert [item.use_count for item in second.planning_state.initial_inputs] == [2, 0]
    assert second.planning_state.unused_initial_inputs == ["run/input-b"]
    assert [
        item.role for item in second.planning_state.completed_invocations
    ] == ["novel_role", "another_novel_role"]
    events = [
        json.loads(line)
        for line in context.trace.path.read_text(encoding="utf-8").splitlines()
    ]
    ledger_events = [
        event for event in events if event["event_type"] == "planner.ledger.updated"
    ]
    assert len(ledger_events) == 2
    assert ledger_events[-1]["planning_state"] == second.planning_state.model_dump(
        mode="json"
    )


async def test_stateful_delegate_also_returns_planning_state(tmp_path: Path) -> None:
    context, _ = build_context(tmp_path, "stateful", static_agents=True)

    result = await execute_delegation(
        context,
        "preset",
        "Analyze the input.",
        ["run/input-b"],
    )

    assert result.planning_state is not None
    assert [item.use_count for item in result.planning_state.initial_inputs] == [0, 1]
    completed = result.planning_state.completed_invocations[0]
    assert completed.model_id == "test-model"
    assert completed.role == "preset"


async def test_efficient_only_changes_instructions_not_execution_semantics(
    tmp_path: Path,
) -> None:
    models = model_registry()
    minimal_instructions = build_blind_coordinator_instructions(
        models, None, "dynamic_models", "minimal"
    )
    stateful_instructions = build_blind_coordinator_instructions(
        models, None, "dynamic_models", "stateful"
    )
    efficient_instructions = build_blind_coordinator_instructions(
        models, None, "dynamic_models", "efficient"
    )

    assert minimal_instructions == stateful_instructions
    assert efficient_instructions.startswith(stateful_instructions)
    assert "A model invocation consumes resources." in efficient_instructions
    assert "Avoid substantially redundant work" in efficient_instructions
    assert "existing artifacts are sufficient" in efficient_instructions
    assert "Stop once the user task can be adequately answered." in efficient_instructions

    stateful_context, stateful_runtime = build_context(tmp_path, "stateful")
    efficient_context, efficient_runtime = build_context(tmp_path, "efficient")
    arguments = (
        "test-model",
        "novel_role",
        "Do focused work.",
        "Analyze the input.",
        ["run/input-a"],
    )
    stateful_result = await execute_dynamic_invocation(stateful_context, *arguments)
    efficient_result = await execute_dynamic_invocation(efficient_context, *arguments)

    assert stateful_runtime.invocations == efficient_runtime.invocations
    assert stateful_result == efficient_result
