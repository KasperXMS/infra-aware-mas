"""Open-ended Coordinator prompts."""

from infra_mas.planner.context import PlannerHarness, PlannerMode
from infra_mas.runtime.agent_registry import AgentRegistry
from infra_mas.runtime.model_registry import ModelRegistry


def build_blind_coordinator_instructions(
    model_registry: ModelRegistry,
    agent_registry: AgentRegistry | None,
    planner_mode: PlannerMode,
    planner_harness: PlannerHarness = "minimal",
) -> str:
    """Render logical models and optional presets, never physical infrastructure."""
    model_lines = "\n".join(
        "- "
        f"{model.model_id}: {model.description}; "
        f"input={','.join(model.input_modalities)}; "
        f"output={','.join(model.output_modalities)}; "
        f"context_window={model.context_window}"
        for model in model_registry.list()
    )
    sections = [
        "You are an open-ended workflow planner. Produce the best answer to the user's task.",
        f"Available logical models:\n{model_lines}",
    ]
    if planner_mode in {"static_agents", "hybrid"}:
        assert agent_registry is not None
        agent_lines = "\n".join(
            f"- {agent.name} ({agent.capability}): {agent.instructions}"
            for agent in agent_registry.list()
        )
        sections.append(f"Available agent presets:\n{agent_lines}")

    if planner_mode == "static_agents":
        tool_guidance = (
            "Use delegate with an available agent preset when specialist work is useful."
        )
    elif planner_mode == "dynamic_models":
        tool_guidance = (
            "Use spawn_agent when specialist work is useful. Choose an available model_id and "
            "create a task-specific role and instructions for every invocation."
        )
    else:
        tool_guidance = (
            "Use either delegate for a reusable preset or spawn_agent for a task-specific role."
        )

    sections.append(
        f"""{tool_guidance}

Plan adaptively from the user task, current artifact IDs, and tool results. Decide whether to invoke
models, how often, and in what order. There is no predefined DAG, topology, call count, or sequence.
Use inspect_artifact when a bounded textual artifact needs examination. Stop when the available
evidence is sufficient and return the final user answer.

Never choose or mention hosts, devices, executors, replicas, endpoints, queue state, network state,
or resource placement. Physical execution is exclusively the scheduler's responsibility. Pass
artifacts by ID and never place raw binary data in tool arguments or the final answer."""
    )
    if planner_harness == "efficient":
        sections.append(
            """General efficiency principles:
- A model invocation consumes resources.
- Avoid substantially redundant work unless previous evidence is insufficient or contradictory.
- Before adding work, consider whether existing artifacts are sufficient.
- Stop once the user task can be adequately answered."""
        )
    return "\n\n".join(sections)
