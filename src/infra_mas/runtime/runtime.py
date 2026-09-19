"""Semantic-to-physical agent runtime boundary."""

from collections.abc import Callable
from uuid import uuid4

from infra_mas.core.artifact import ArtifactRef
from infra_mas.core.execution import (
    AggregateArtifactsRequest,
    ExecutionRequest,
    ExecutionResult,
    ExtractClipRequest,
    InvocationSpec,
    MakeContactSheetRequest,
    SampleFramesRequest,
)
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
            semantic_operator=invocation.semantic_operator,
            tool=invocation.semantic_operator,
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

    async def sample_frames(
        self,
        artifact: ArtifactRef,
        *,
        duration_s: float | None = None,
        sample_count: int = 20,
        columns: int = 5,
        frame_width: int = 448,
        parent_action_id: str | None = None,
    ) -> ExecutionResult:
        """Uniformly sample a video; runtime chooses a local capable Worker."""
        return await self._sample_frames(
            artifact,
            target_worker_id=None,
            duration_s=duration_s,
            sample_count=sample_count,
            columns=columns,
            frame_width=frame_width,
            parent_action_id=parent_action_id,
        )

    async def sample_frames_on_worker(
        self,
        artifact: ArtifactRef,
        target_worker_id: str,
        *,
        duration_s: float | None = None,
        sample_count: int = 20,
        columns: int = 5,
        frame_width: int = 448,
        parent_action_id: str | None = None,
    ) -> ExecutionResult:
        """Internal fixed-site adapter used by reference measurement workflows."""
        return await self._sample_frames(
            artifact,
            target_worker_id=target_worker_id,
            duration_s=duration_s,
            sample_count=sample_count,
            columns=columns,
            frame_width=frame_width,
            parent_action_id=parent_action_id,
        )

    async def _sample_frames(
        self,
        artifact: ArtifactRef,
        *,
        target_worker_id: str | None,
        duration_s: float | None,
        sample_count: int,
        columns: int,
        frame_width: int,
        parent_action_id: str | None,
    ) -> ExecutionResult:
        request_id = self._request_id_factory()
        await self._trace.record(
            "execution.request",
            action_id=request_id,
            parent_action_id=parent_action_id,
            request_id=request_id,
            agent="sample_frames",
            capability="video_preprocessing",
            semantic_operator="sample_frames",
            tool="sample_frames",
            task="Uniformly sample frames into a contact sheet.",
            input_artifacts=[artifact.id],
        )
        return await self._execution_manager.sample_frames(
            SampleFramesRequest(
                request_id=request_id,
                input_artifact=artifact,
                output_artifact_id=f"{request_id.rsplit('/', 1)[0]}/frames-{uuid4().hex}.jpg",
                duration_s=duration_s,
                sample_count=sample_count,
                columns=columns,
                frame_width=frame_width,
            ),
            target_worker_id,
        )

    async def make_contact_sheet(
        self,
        artifacts: list[ArtifactRef],
        *,
        columns: int = 5,
        duration_s: float | None = None,
        parent_action_id: str | None = None,
    ) -> ExecutionResult:
        """Compose images on a locality-selected Worker."""
        request_id = self._request_id_factory()
        await self._record_operator_request(
            request_id,
            "make_contact_sheet",
            "Compose chronological image artifacts into a contact sheet.",
            artifacts,
            parent_action_id,
        )
        return await self._execution_manager.make_contact_sheet(
            MakeContactSheetRequest(
                request_id=request_id,
                input_artifacts=artifacts,
                output_artifact_id=f"{request_id.rsplit('/', 1)[0]}/sheet-{uuid4().hex}.jpg",
                columns=columns,
                duration_s=duration_s,
            )
        )

    async def extract_clip(
        self,
        artifact: ArtifactRef,
        *,
        start_s: float,
        end_s: float,
        parent_action_id: str | None = None,
    ) -> ExecutionResult:
        """Extract a fixed interval on a locality-selected Worker."""
        request_id = self._request_id_factory()
        await self._record_operator_request(
            request_id,
            "extract_clip",
            f"Extract video interval [{start_s}, {end_s}) seconds.",
            [artifact],
            parent_action_id,
        )
        return await self._execution_manager.extract_clip(
            ExtractClipRequest(
                request_id=request_id,
                input_artifact=artifact,
                output_artifact_id=f"{request_id.rsplit('/', 1)[0]}/clip-{uuid4().hex}.mp4",
                start_s=start_s,
                end_s=end_s,
            )
        )

    async def aggregate_artifacts(
        self,
        artifacts: list[ArtifactRef],
        *,
        parent_action_id: str | None = None,
    ) -> ExecutionResult:
        """Aggregate textual evidence on a locality-selected Worker."""
        request_id = self._request_id_factory()
        await self._record_operator_request(
            request_id,
            "aggregate_artifacts",
            "Aggregate textual semantic evidence into one structured artifact.",
            artifacts,
            parent_action_id,
        )
        return await self._execution_manager.aggregate_artifacts(
            AggregateArtifactsRequest(
                request_id=request_id,
                input_artifacts=artifacts,
                output_artifact_id=f"{request_id.rsplit('/', 1)[0]}/evidence-{uuid4().hex}.json",
            )
        )

    async def process_local_artifact(
        self,
        *,
        model_id: str,
        role: str,
        instructions: str,
        task: str,
        artifacts: list[ArtifactRef],
        parent_action_id: str | None = None,
    ) -> ExecutionResult:
        """Run model processing through normal capability/locality scheduling."""
        return await self.invoke(
            InvocationSpec(
                model_id=model_id,
                role=role,
                instructions=instructions,
                task=task,
                input_artifacts=artifacts,
                semantic_operator="process_local_artifact",
            ),
            parent_action_id=parent_action_id,
        )

    async def _record_operator_request(
        self,
        request_id: str,
        operator: str,
        task: str,
        artifacts: list[ArtifactRef],
        parent_action_id: str | None,
    ) -> None:
        await self._trace.record(
            "execution.request",
            action_id=request_id,
            parent_action_id=parent_action_id,
            request_id=request_id,
            agent=operator,
            capability="media_processing",
            semantic_operator=operator,
            tool=operator,
            task=task,
            input_artifacts=[artifact.id for artifact in artifacts],
        )

    @staticmethod
    def _default_request_id() -> str:
        return f"request-{uuid4().hex}"

    @staticmethod
    def _artifact_modality(artifact_type: str) -> str | None:
        media_type = artifact_type.partition(";")[0].strip().lower()
        if media_type.startswith("image/"):
            return "image"
        if media_type.startswith("video/"):
            return "video"
        if (
            media_type.startswith("text/")
            or media_type in _TEXT_APPLICATION_TYPES
            or media_type.endswith("+json")
            or media_type.endswith("+xml")
            or media_type.endswith("+yaml")
        ):
            return "text"
        return None
