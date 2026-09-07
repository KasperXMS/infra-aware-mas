"""Resource-blind Coordinator tools."""

from agents import RunContextWrapper, function_tool
from pydantic import BaseModel, ConfigDict

from infra_mas.planner.context import PlannerContext

_MAX_DELEGATE_SUMMARY_CHARS = 4000


class DelegationToolResult(BaseModel):
    """Return semantic output to the Coordinator without infrastructure details."""

    model_config = ConfigDict(extra="forbid")

    agent: str
    output_artifact_ids: list[str]
    output_text: str


async def execute_delegation(
    context: PlannerContext,
    agent: str,
    task: str,
    input_artifact_ids: list[str],
) -> DelegationToolResult:
    """Execute one semantic delegation and register its output artifacts."""
    action_id = await context.next_action_id("delegate")
    await context.trace.record(
        "planner.delegate",
        action_id=action_id,
        parent_action_id=context.coordinator_action_id,
        agent=agent,
        task=task,
        input_artifacts=input_artifact_ids,
    )
    inputs = [context.artifact_catalog.get(artifact_id) for artifact_id in input_artifact_ids]
    result = await context.runtime.execute(
        agent,
        task,
        inputs,
        parent_action_id=action_id,
    )
    context.artifact_catalog.register_many(result.output_artifacts)

    summaries: list[str] = []
    for artifact in result.output_artifacts:
        if context.artifact_catalog.is_inspectable(artifact):
            text = await context.artifact_catalog.inspect_text(artifact.id)
            if len(text) > _MAX_DELEGATE_SUMMARY_CHARS:
                text = f"{text[:_MAX_DELEGATE_SUMMARY_CHARS]}\n[truncated]"
            summaries.append(text)
        else:
            summaries.append(f"Artifact {artifact.id} is available for further delegation.")
    return DelegationToolResult(
        agent=agent,
        output_artifact_ids=[artifact.id for artifact in result.output_artifacts],
        output_text="\n".join(summaries),
    )


async def inspect_artifact_text(context: PlannerContext, artifact_id: str) -> str:
    """Inspect a registered textual artifact and trace the semantic read."""
    action_id = await context.next_action_id("inspect")
    await context.trace.record(
        "planner.inspect_artifact",
        action_id=action_id,
        parent_action_id=context.coordinator_action_id,
        artifact_id=artifact_id,
    )
    return await context.artifact_catalog.inspect_text(artifact_id)


@function_tool
async def delegate(
    ctx: RunContextWrapper[PlannerContext],
    agent: str,
    task: str,
    input_artifact_ids: list[str],
) -> str:
    """Delegate semantic work to a registered agent.

    Args:
        agent: Registered semantic agent name, never a machine or executor ID.
        task: Self-contained description of the semantic work to perform.
        input_artifact_ids: IDs of artifacts the delegated agent should consume.
    """
    result = await execute_delegation(ctx.context, agent, task, input_artifact_ids)
    return result.model_dump_json()


@function_tool
async def inspect_artifact(
    ctx: RunContextWrapper[PlannerContext],
    artifact_id: str,
) -> str:
    """Read a small textual or structured artifact.

    Args:
        artifact_id: Registered artifact ID returned by a delegation.
    """
    return await inspect_artifact_text(ctx.context, artifact_id)
