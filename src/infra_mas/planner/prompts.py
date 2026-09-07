"""Coordinator prompts."""

from infra_mas.runtime.agent_registry import AgentRegistry


def build_blind_coordinator_instructions(agent_registry: AgentRegistry) -> str:
    """Render only semantic agent information for the resource-blind Coordinator."""
    agent_lines = "\n".join(
        f"- {agent.name} ({agent.capability}): {agent.instructions}"
        for agent in agent_registry.list()
    )
    return f"""You coordinate semantic agents to answer the user's task.

Available semantic agents:
{agent_lines}

Use delegate to perform specialized work. Pass only registered agent names, semantic tasks, and
artifact IDs. Use inspect_artifact when you need to read a small textual result. Perform further
delegations when an intermediate artifact needs more processing, then return the final user answer.

Never choose or mention hosts, devices, executors, model endpoints, queue state, network state, or
resource placement. Physical execution is exclusively the scheduler's responsibility. Never place
raw binary data in tool arguments or in the final answer.
"""
