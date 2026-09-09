"""Resource-blind Coordinator tools."""

from agents import RunContextWrapper, function_tool
from pydantic import BaseModel, ConfigDict

from infra_mas.core.artifact import ArtifactRef
from infra_mas.core.execution import InvocationSpec
from infra_mas.planner.context import PlannerContext

_MAX_DELEGATE_SUMMARY_CHARS = 4000


class DelegationToolResult(BaseModel):
    """Return semantic output to the Coordinator without infrastructure details."""

    model_config = ConfigDict(extra="forbid")

    agent: str
    output_artifact_ids: list[str]
    output_text: str


async def _summarize_outputs(
    context: PlannerContext,
    output_artifacts: list[ArtifactRef],
    action_id: str,
) -> str:
    summaries: list[str] = []
    for artifact in output_artifacts:
        if context.artifact_catalog.is_inspectable(artifact):
            artifact_text = await context.artifact_catalog.inspect_text(
                artifact.id,
                parent_action_id=action_id,
            )
            if len(artifact_text) > _MAX_DELEGATE_SUMMARY_CHARS:
                artifact_text = (
                    f"{artifact_text[:_MAX_DELEGATE_SUMMARY_CHARS]}\n[truncated]"
                )
            summaries.append(artifact_text)
        else:
            summaries.append(f"Artifact {artifact.id} is available for further delegation.")
    return "\n".join(summaries)


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

    return DelegationToolResult(
        agent=agent,
        output_artifact_ids=[artifact.id for artifact in result.output_artifacts],
        output_text=await _summarize_outputs(context, result.output_artifacts, action_id),
    )


async def execute_dynamic_invocation(
    context: PlannerContext,
    model_id: str,
    role: str,
    instructions: str,
    task: str,
    input_artifact_ids: list[str],
) -> DelegationToolResult:
    """Execute a planner-created role through the unified invocation path."""
    assert context.model_registry is not None
    context.model_registry.get(model_id)
    action_id = await context.next_action_id("spawn")
    await context.trace.record(
        "planner.spawn_agent",
        action_id=action_id,
        parent_action_id=context.coordinator_action_id,
        model_id=model_id,
        role=role,
        instructions=instructions,
        task=task,
        input_artifacts=input_artifact_ids,
    )
    invocation = InvocationSpec(
        model_id=model_id,
        role=role,
        instructions=instructions,
        task=task,
        input_artifacts=[
            context.artifact_catalog.get(artifact_id)
            for artifact_id in input_artifact_ids
        ],
    )
    result = await context.runtime.invoke(invocation, parent_action_id=action_id)
    context.artifact_catalog.register_many(result.output_artifacts)
    return DelegationToolResult(
        agent=role,
        output_artifact_ids=[artifact.id for artifact in result.output_artifacts],
        output_text=await _summarize_outputs(context, result.output_artifacts, action_id),
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
    return await context.artifact_catalog.inspect_text(
        artifact_id,
        parent_action_id=action_id,
    )


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
async def spawn_agent(
    ctx: RunContextWrapper[PlannerContext],
    model_id: str,
    role: str,
    instructions: str,
    task: str,
    input_artifact_ids: list[str],
) -> str:
    """Create and invoke a task-specific agent role on a logical model.

    Args:
        model_id: Available logical model ID, never an executor or machine ID.
        role: Concise task-specific role name generated for this invocation.
        instructions: System instructions defining the role's behavior and constraints.
        task: Self-contained work for this invocation.
        input_artifact_ids: IDs of artifacts this invocation should consume.
    """
    result = await execute_dynamic_invocation(
        ctx.context,
        model_id,
        role,
        instructions,
        task,
        input_artifact_ids,
    )
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
