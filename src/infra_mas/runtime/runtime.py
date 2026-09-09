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

_TEXT_APPLICATION_TYPES = frozenset(
    {
        "application/json",
        "application/xml",
        "application/yaml",
        "application/x-yaml",
    }
)


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
        model_registry: ModelRegistry,
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
        invocation = self.create_preset_invocation(agent_name, task, inputs)
        return await self.invoke(invocation, parent_action_id=parent_action_id)

    def create_preset_invocation(
        self,
        agent_name: str,
        task: str,
        inputs: list[ArtifactRef],
    ) -> InvocationSpec:
        """Convert one backward-compatible agent preset to the unified request."""
        if self._agent_registry is None:
            raise ValueError("static agent execution requires an AgentRegistry")
        agent = self._agent_registry.get(agent_name)
        model_id = agent.model_id or self._scheduler.preset_model_id(agent.capability)
        return InvocationSpec(
            model_id=model_id,
            role=agent.name,
            instructions=agent.instructions,
            task=task,
            input_artifacts=inputs,
        )

    async def invoke(
        self,
        invocation: InvocationSpec,
        *,
        parent_action_id: str | None = None,
    ) -> ExecutionResult:
        """Schedule and execute one unified logical model invocation."""
        model = self._model_registry.get(invocation.model_id)
        accepted_modalities = {modality.lower() for modality in model.input_modalities}
        for artifact in invocation.input_artifacts:
            modality = self._artifact_modality(artifact.artifact_type)
            if modality is None or modality not in accepted_modalities:
                inferred = modality or "unsupported"
                raise ValueError(
                    f"model {model.model_id!r} does not accept artifact {artifact.id!r} "
                    f"with MIME type {artifact.artifact_type!r} "
                    f"(inferred modality: {inferred!r}); accepted input modalities: "
                    f"{sorted(accepted_modalities)}"
                )
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

    @staticmethod
    def _artifact_modality(artifact_type: str) -> str | None:
        media_type = artifact_type.partition(";")[0].strip().lower()
        if media_type.startswith("image/"):
            return "image"
        if (
            media_type.startswith("text/")
            or media_type in _TEXT_APPLICATION_TYPES
            or media_type.endswith("+json")
            or media_type.endswith("+xml")
            or media_type.endswith("+yaml")
        ):
            return "text"
        return None
