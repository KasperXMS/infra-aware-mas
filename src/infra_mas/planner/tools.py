"""Resource-blind Coordinator tools."""

from agents import RunContextWrapper, function_tool
from pydantic import BaseModel, ConfigDict

from infra_mas.core.artifact import ArtifactRef
from infra_mas.core.execution import InvocationSpec
from infra_mas.planner.context import PlannerContext
from infra_mas.planner.ledger import CompletedInvocation, PlanningState

_MAX_DELEGATE_SUMMARY_CHARS = 4000


class DelegationToolResult(BaseModel):
    """Return semantic output to the Coordinator without infrastructure details."""

    model_config = ConfigDict(extra="forbid")

    agent: str
    output_artifact_ids: list[str]
    output_text: str
    planning_state: PlanningState | None = None


class MediaToolResult(BaseModel):
    """Return logical media-operation outputs without physical placement details."""

    model_config = ConfigDict(extra="forbid")

    operator: str
    output_artifact_ids: list[str]
    output_text: str


async def _record_completed_invocation(
    context: PlannerContext,
    action_id: str,
    invocation: InvocationSpec,
    output_artifact_ids: list[str],
) -> PlanningState | None:
    assert context.planning_ledger is not None
    state = await context.planning_ledger.record_completed(
        CompletedInvocation(
            action_id=action_id,
            model_id=invocation.model_id,
            role=invocation.role,
            task=invocation.task,
            input_artifact_ids=[artifact.id for artifact in invocation.input_artifacts],
            output_artifact_ids=output_artifact_ids,
        )
    )
    await context.trace.record(
        "planner.ledger.updated",
        action_id=action_id,
        parent_action_id=context.coordinator_action_id,
        planner_harness=context.planner_harness,
        planning_state=state.model_dump(mode="json"),
    )
    return state if context.planner_harness != "minimal" else None


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
        semantic_operator="invoke_model",
        input_artifacts=input_artifact_ids,
    )
    inputs = [context.artifact_catalog.get(artifact_id) for artifact_id in input_artifact_ids]
    invocation = context.runtime.create_preset_invocation(agent, task, inputs)
    result = await context.runtime.invoke(invocation, parent_action_id=action_id)
    context.artifact_catalog.register_many(result.output_artifacts)
    output_artifact_ids = [artifact.id for artifact in result.output_artifacts]
    planning_state = await _record_completed_invocation(
        context,
        action_id,
        invocation,
        output_artifact_ids,
    )

    return DelegationToolResult(
        agent=agent,
        output_artifact_ids=output_artifact_ids,
        output_text=await _summarize_outputs(context, result.output_artifacts, action_id),
        planning_state=planning_state,
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
        semantic_operator="invoke_model",
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
    output_artifact_ids = [artifact.id for artifact in result.output_artifacts]
    planning_state = await _record_completed_invocation(
        context,
        action_id,
        invocation,
        output_artifact_ids,
    )
    return DelegationToolResult(
        agent=role,
        output_artifact_ids=output_artifact_ids,
        output_text=await _summarize_outputs(context, result.output_artifacts, action_id),
        planning_state=planning_state,
    )


async def inspect_artifact_text(context: PlannerContext, artifact_id: str) -> str:
    """Inspect a registered textual artifact and trace the semantic read."""
    action_id = await context.next_action_id("inspect")
    await context.trace.record(
        "planner.inspect_artifact",
        action_id=action_id,
        parent_action_id=context.coordinator_action_id,
        artifact_id=artifact_id,
        semantic_operator="read_artifact",
    )
    return await context.artifact_catalog.inspect_text(
        artifact_id,
        parent_action_id=action_id,
    )


async def _finish_media_operation(
    context: PlannerContext,
    operator: str,
    action_id: str,
    output_artifacts: list[ArtifactRef],
) -> MediaToolResult:
    context.artifact_catalog.register_many(output_artifacts)
    return MediaToolResult(
        operator=operator,
        output_artifact_ids=[artifact.id for artifact in output_artifacts],
        output_text=await _summarize_outputs(context, output_artifacts, action_id),
    )


async def _start_media_operation(
    context: PlannerContext,
    operator: str,
    input_artifact_ids: list[str],
) -> str:
    action_id = await context.next_action_id("media")
    await context.trace.record(
        "planner.media_operator",
        action_id=action_id,
        parent_action_id=context.coordinator_action_id,
        semantic_operator=operator,
        input_artifacts=input_artifact_ids,
    )
    return action_id


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
    return result.model_dump_json(exclude_none=True)


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
    return result.model_dump_json(exclude_none=True)


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


@function_tool
async def sample_frames(
    ctx: RunContextWrapper[PlannerContext],
    artifact_id: str,
    duration_s: float | None = None,
    sample_count: int = 20,
    columns: int = 5,
    frame_width: int = 448,
) -> str:
    """Uniformly sample a video into a chronological contact-sheet artifact.

    Args:
        artifact_id: Logical ID of the input video artifact.
        duration_s: Optional duration override; the Worker probes it when omitted.
        sample_count: Number of fixed-rate samples across the complete duration.
        columns: Contact-sheet column count.
        frame_width: Width in pixels of each sampled frame.
    """
    context = ctx.context
    action_id = await _start_media_operation(context, "sample_frames", [artifact_id])
    artifact = context.artifact_catalog.get(artifact_id)
    result = await context.runtime.sample_frames(
        artifact,
        duration_s=duration_s,
        sample_count=sample_count,
        columns=columns,
        frame_width=frame_width,
        parent_action_id=action_id,
    )
    return (
        await _finish_media_operation(
            context, "sample_frames", action_id, result.output_artifacts
        )
    ).model_dump_json()


@function_tool
async def make_contact_sheet(
    ctx: RunContextWrapper[PlannerContext],
    input_artifact_ids: list[str],
    columns: int = 5,
    duration_s: float | None = None,
) -> str:
    """Compose logical image artifacts into a chronological contact sheet.

    Args:
        input_artifact_ids: Logical IDs of image artifacts in chronological order.
        columns: Contact-sheet column count.
        duration_s: Optional covered duration used only for timestamp labels.
    """
    context = ctx.context
    action_id = await _start_media_operation(
        context, "make_contact_sheet", input_artifact_ids
    )
    result = await context.runtime.make_contact_sheet(
        [context.artifact_catalog.get(item) for item in input_artifact_ids],
        columns=columns,
        duration_s=duration_s,
        parent_action_id=action_id,
    )
    return (
        await _finish_media_operation(
            context, "make_contact_sheet", action_id, result.output_artifacts
        )
    ).model_dump_json()


@function_tool
async def extract_clip(
    ctx: RunContextWrapper[PlannerContext],
    artifact_id: str,
    start_s: float,
    end_s: float,
) -> str:
    """Extract a fixed time interval from a logical video artifact.

    Args:
        artifact_id: Logical ID of the input video artifact.
        start_s: Inclusive clip start in seconds.
        end_s: Exclusive clip end in seconds.
    """
    context = ctx.context
    action_id = await _start_media_operation(context, "extract_clip", [artifact_id])
    result = await context.runtime.extract_clip(
        context.artifact_catalog.get(artifact_id),
        start_s=start_s,
        end_s=end_s,
        parent_action_id=action_id,
    )
    return (
        await _finish_media_operation(
            context, "extract_clip", action_id, result.output_artifacts
        )
    ).model_dump_json()


@function_tool
async def aggregate_artifacts(
    ctx: RunContextWrapper[PlannerContext],
    input_artifact_ids: list[str],
) -> str:
    """Aggregate logical textual evidence artifacts into one JSON artifact.

    Args:
        input_artifact_ids: Logical IDs of UTF-8 textual evidence artifacts.
    """
    context = ctx.context
    action_id = await _start_media_operation(
        context, "aggregate_artifacts", input_artifact_ids
    )
    result = await context.runtime.aggregate_artifacts(
        [context.artifact_catalog.get(item) for item in input_artifact_ids],
        parent_action_id=action_id,
    )
    return (
        await _finish_media_operation(
            context, "aggregate_artifacts", action_id, result.output_artifacts
        )
    ).model_dump_json()


@function_tool
async def process_local_artifact(
    ctx: RunContextWrapper[PlannerContext],
    model_id: str,
    role: str,
    instructions: str,
    task: str,
    input_artifact_ids: list[str],
) -> str:
    """Process logical artifacts with a logical model selected by the runtime.

    Args:
        model_id: Logical model ID, never an executor or machine ID.
        role: Concise semantic role for this processing step.
        instructions: General processing instructions.
        task: Self-contained semantic processing task.
        input_artifact_ids: Logical IDs of artifacts to process.
    """
    context = ctx.context
    action_id = await _start_media_operation(
        context, "process_local_artifact", input_artifact_ids
    )
    inputs = [context.artifact_catalog.get(item) for item in input_artifact_ids]
    invocation = InvocationSpec(
        model_id=model_id,
        role=role,
        instructions=instructions,
        task=task,
        input_artifacts=inputs,
        semantic_operator="process_local_artifact",
    )
    result = await context.runtime.invoke(invocation, parent_action_id=action_id)
    context.artifact_catalog.register_many(result.output_artifacts)
    output_ids = [artifact.id for artifact in result.output_artifacts]
    await _record_completed_invocation(context, action_id, invocation, output_ids)
    return MediaToolResult(
        operator="process_local_artifact",
        output_artifact_ids=output_ids,
        output_text=await _summarize_outputs(context, result.output_artifacts, action_id),
    ).model_dump_json()
