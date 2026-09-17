"""Planner context for repository code tools."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field

from infra_mas.code_tasks.client import CodeWorkerClient
from infra_mas.code_tasks.models import CodeToolName, CodeToolResult, RepositoryWorld
from infra_mas.code_tasks.scheduler import RepositoryLocalityScheduler
from infra_mas.planner.context import InfrastructureVisibility
from infra_mas.tracing.recorder import TraceRecorder


@dataclass(slots=True)
class CodePlannerContext:
    """Hold world state, tool transport, and trace state without semantic advice."""

    trace: TraceRecorder
    scheduler: RepositoryLocalityScheduler
    world: RepositoryWorld
    visibility: InfrastructureVisibility
    planner_site: str
    timeout_seconds: float = 900
    max_exploration_calls: int = 16
    submitted_patch: str | None = None
    _exploration_calls: int = field(default=0, init=False, repr=False)
    _action_counter: int = field(default=0, init=False, repr=False)
    _action_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)

    @property
    def coordinator_action_id(self) -> str:
        return f"{self.trace.run_id}/coordinator"

    async def next_action_id(self, prefix: str) -> str:
        async with self._action_lock:
            self._action_counter += 1
            sequence = self._action_counter
        return f"{self.trace.run_id}/{prefix}-{sequence:04d}"

    async def invoke_tool(
        self, tool: CodeToolName, payload: dict[str, object]
    ) -> str:
        action_id = await self.next_action_id("code-tool")
        executor = self.scheduler.select(self.world)
        sanitized = {
            key: (f"<{len(str(value).encode('utf-8'))} bytes>" if key == "patch" else value)
            for key, value in payload.items()
        }
        await self.trace.record(
            "planner.code_tool",
            action_id=action_id,
            parent_action_id=self.coordinator_action_id,
            tool=tool,
            arguments=sanitized,
        )
        await self.trace.record(
            "code_tool.start",
            action_id=action_id,
            parent_action_id=self.coordinator_action_id,
            tool=tool,
            executor_id=executor.executor_id,
            site=executor.site,
        )
        if tool in {"search_code", "read_file"}:
            self._exploration_calls += 1
            if self._exploration_calls > self.max_exploration_calls:
                result = CodeToolResult(
                    tool=tool,
                    success=False,
                    output=(
                        "The fixed read/search exploration budget is exhausted. Use the "
                        "evidence already gathered to edit, test, and submit the solution."
                    ),
                    service_ms=0,
                )
                planner_text = result.model_dump_json(exclude={"patch"})
                await self.trace.record(
                    "code_tool.end",
                    action_id=action_id,
                    parent_action_id=self.coordinator_action_id,
                    tool=tool,
                    executor_id=executor.executor_id,
                    site=executor.site,
                    success=False,
                    budget_exhausted=True,
                    service_ms=0,
                    elapsed_ms=0,
                    request_bytes=0,
                    response_bytes=0,
                    planner_context_bytes=len(planner_text.encode("utf-8")),
                    cross_site=False,
                    cross_site_transfer_bytes=0,
                    cross_site_transfer_ms=0,
                    truncated=False,
                )
                return planner_text
        client = CodeWorkerClient(executor.endpoint, timeout_seconds=self.timeout_seconds)
        try:
            result, request_bytes, response_bytes, elapsed_ms = await client.invoke(tool, payload)
        except Exception as error:
            await self.trace.record(
                "code_tool.end",
                action_id=action_id,
                parent_action_id=self.coordinator_action_id,
                tool=tool,
                executor_id=executor.executor_id,
                site=executor.site,
                success=False,
                error_type=type(error).__name__,
                error=str(error),
            )
            raise
        finally:
            await client.aclose()

        if tool == "submit_patch" and result.success:
            self.submitted_patch = result.patch
        planner_payload = result.model_dump(exclude={"patch"}, mode="json")
        planner_text = json.dumps(planner_payload, ensure_ascii=False)
        cross_site = executor.site != self.planner_site
        transfer_bytes = request_bytes + response_bytes if cross_site else 0
        transfer_ms = max(0.0, elapsed_ms - result.service_ms) if cross_site else 0.0
        await self.trace.record(
            "code_tool.end",
            action_id=action_id,
            parent_action_id=self.coordinator_action_id,
            tool=tool,
            executor_id=executor.executor_id,
            site=executor.site,
            success=result.success,
            service_ms=result.service_ms,
            elapsed_ms=elapsed_ms,
            request_bytes=request_bytes,
            response_bytes=response_bytes,
            planner_context_bytes=len(planner_text.encode("utf-8")),
            cross_site=cross_site,
            cross_site_transfer_bytes=transfer_bytes,
            cross_site_transfer_ms=transfer_ms,
            truncated=result.truncated,
        )
        return planner_text

    def render_static_context(self) -> str:
        executor_sites = ",".join(sorted({item.site for item in self.scheduler.list()}))
        return "\n".join(
            [
                "Static execution context:",
                f"- Planner reasoning service site: {self.planner_site}.",
                f"- Compatible code-tool executor sites: {executor_sites}.",
                "- All code tools execute against the same mutable repository workspace.",
                "- Tool execution is deterministically bound to the executor holding "
                "that workspace.",
                "- Tool output becomes context for the next Planner reasoning turn.",
                "- Cross-site tool requests and outputs consume network transfer.",
                f"- Combined search_code/read_file budget: {self.max_exploration_calls} calls.",
            ]
        )

    def render_dynamic_context(self) -> str:
        return "\n".join(
            [
                "Dynamic infrastructure snapshot (raw facts):",
                "Repository artifact:",
                f"- artifact_id={self.world.repository_artifact_id}; "
                f"size_bytes={self.world.repository_size_bytes}; "
                f"location={self.world.repository_site}",
                "Network link:",
                f"- source_site=edge; target_site=cloud; "
                f"bandwidth_mbps={self.world.bandwidth_mbps:g}; "
                f"rtt_ms={self.world.rtt_ms:g}; bidirectional=true",
            ]
        )
