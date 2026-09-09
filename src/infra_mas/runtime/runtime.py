"""Semantic-to-physical agent runtime boundary."""

from collections.abc import Callable
from uuid import uuid4

from infra_mas.core.artifact import ArtifactRef
from infra_mas.core.execution import ExecutionRequest, ExecutionResult, InvocationSpec
from infra_mas.core.trace import TraceSink
from infra_mas.execution.manager import ExecutionManager
from infra_mas.runtime.agent_registry import AgentRegistry
from infra_mas.runtime.model_registry import ModelRegistry
from infra_mas.scheduler.base import Scheduler

RequestIdFactory = Callable[[], str]


class AgentRuntime:
    """Translate semantic agent delegation into scheduled physical execution."""

    def __init__(
        self,
        agent_registry: AgentRegistry | None,
        scheduler: Scheduler,
        execution_manager: ExecutionManager,
        trace: TraceSink,
        request_id_factory: RequestIdFactory | None = None,
        *,
        model_registry: ModelRegistry | None = None,
    ) -> None:
        self._agent_registry = agent_registry
        self._scheduler = scheduler
        self._execution_manager = execution_manager
        self._trace = trace
        self._model_registry = model_registry
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
        if self._agent_registry is None:
            raise ValueError("static agent execution requires an AgentRegistry")
        agent = self._agent_registry.get(agent_name)
        model_id = agent.model_id or self._scheduler.preset_model_id(agent.capability)
        invocation = InvocationSpec(
            model_id=model_id,
            role=agent.name,
            instructions=agent.instructions,
            task=task,
            input_artifacts=inputs,
        )
        return await self.invoke(invocation, parent_action_id=parent_action_id)

    async def invoke(
        self,
        invocation: InvocationSpec,
        *,
        parent_action_id: str | None = None,
    ) -> ExecutionResult:
        """Schedule and execute one unified logical model invocation."""
        if self._model_registry is not None:
            self._model_registry.get(invocation.model_id)
        executor = await self._scheduler.select(invocation)
        request = ExecutionRequest(
            request_id=self._request_id_factory(),
            agent=invocation.role,
            model_id=invocation.model_id,
            capability=executor.capability,
            instructions=invocation.instructions,
            task=invocation.task,
            inputs=invocation.input_artifacts,
        )
        await self._trace.record(
            "execution.request",
            action_id=request.request_id,
            parent_action_id=parent_action_id,
            request_id=request.request_id,
            agent=invocation.role,
            model_id=invocation.model_id,
            capability=executor.capability,
            task=invocation.task,
            input_artifacts=[artifact.id for artifact in invocation.input_artifacts],
        )
        await self._trace.record(
            "executor.selected",
            action_id=request.request_id,
            parent_action_id=parent_action_id,
            request_id=request.request_id,
            agent=invocation.role,
            model_id=invocation.model_id,
            executor=executor.id,
            worker_id=executor.worker_id,
            site=executor.site,
        )
        return await self._execution_manager.execute(request, executor)

    @staticmethod
    def _default_request_id() -> str:
        return f"request-{uuid4().hex}"
