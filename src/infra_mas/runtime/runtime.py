"""Semantic-to-physical agent runtime boundary."""

from collections.abc import Callable
from uuid import uuid4

from infra_mas.core.artifact import ArtifactRef
from infra_mas.core.execution import ExecutionRequest, ExecutionResult
from infra_mas.core.trace import TraceSink
from infra_mas.execution.manager import ExecutionManager
from infra_mas.runtime.agent_registry import AgentRegistry
from infra_mas.scheduler.base import Scheduler

RequestIdFactory = Callable[[], str]


class AgentRuntime:
    """Translate semantic agent delegation into scheduled physical execution."""

    def __init__(
        self,
        agent_registry: AgentRegistry,
        scheduler: Scheduler,
        execution_manager: ExecutionManager,
        trace: TraceSink,
        request_id_factory: RequestIdFactory | None = None,
    ) -> None:
        self._agent_registry = agent_registry
        self._scheduler = scheduler
        self._execution_manager = execution_manager
        self._trace = trace
        self._request_id_factory = request_id_factory or self._default_request_id

    async def execute(
        self,
        agent_name: str,
        task: str,
        inputs: list[ArtifactRef],
        *,
        parent_action_id: str | None = None,
    ) -> ExecutionResult:
        """Resolve an agent, schedule its request, and execute the selected binding."""
        agent = self._agent_registry.get(agent_name)
        request = ExecutionRequest(
            request_id=self._request_id_factory(),
            agent=agent.name,
            capability=agent.capability,
            task=task,
            inputs=inputs,
        )
        await self._trace.record(
            "execution.request",
            action_id=request.request_id,
            parent_action_id=parent_action_id,
            request_id=request.request_id,
            agent=agent.name,
            capability=agent.capability,
            task=task,
            input_artifacts=[artifact.id for artifact in inputs],
        )
        executor = await self._scheduler.select(request)
        await self._trace.record(
            "executor.selected",
            action_id=request.request_id,
            parent_action_id=parent_action_id,
            request_id=request.request_id,
            agent=agent.name,
            executor=executor.id,
            worker_id=executor.worker_id,
            site=executor.site,
        )
        return await self._execution_manager.execute(request, executor)

    @staticmethod
    def _default_request_id() -> str:
        return f"request-{uuid4().hex}"
