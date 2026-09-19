"""Execution manager."""

import asyncio
from collections.abc import Awaitable, Callable, Mapping

from infra_mas.core.artifact import ArtifactRef
from infra_mas.core.errors import ExecutionFailedError, WorkerUnavailableError
from infra_mas.core.execution import (
    AggregateArtifactsRequest,
    ExecutionRequest,
    ExecutionResult,
    ExtractClipRequest,
    MakeContactSheetRequest,
    SampleFramesRequest,
)
from infra_mas.core.executor import ExecutorSpec
from infra_mas.core.trace import TraceSink
from infra_mas.execution.transfer import TransferManager
from infra_mas.execution.worker_client import WorkerClient


class ExecutionManager:
    """Execute an already scheduled request on its assigned physical executor."""

    def __init__(
        self,
        clients: Mapping[str, WorkerClient],
        transfer_manager: TransferManager,
        trace: TraceSink,
    ) -> None:
        self._clients = dict(clients)
        self._transfer_manager = transfer_manager
        self._trace = trace

    async def execute(
        self,
        request: ExecutionRequest,
        executor: ExecutorSpec,
    ) -> ExecutionResult:
        """Localize inputs, invoke the assigned worker, and combine timing metadata."""
        client = self._clients.get(executor.worker_id)
        if client is None:
            raise WorkerUnavailableError(f"no client configured for worker {executor.worker_id!r}")

        transfer_ms = 0.0
        for artifact in request.inputs:
            transfer = await self._transfer_manager.ensure_local(
                artifact,
                executor.worker_id,
                action_id=request.request_id,
            )
            transfer_ms += transfer.transfer_ms

        await self._trace.record(
            "worker.execution.start",
            action_id=request.request_id,
            request_id=request.request_id,
            agent=request.agent,
            executor=executor.id,
            worker_id=executor.worker_id,
        )
        try:
            result = await client.execute(request, executor_id=executor.id)
            if result.executor_id != executor.id:
                raise ExecutionFailedError(
                    f"worker executed {result.executor_id!r}, expected {executor.id!r}"
                )
            if any(
                executor.worker_id not in artifact.locations for artifact in result.output_artifacts
            ):
                raise ExecutionFailedError(
                    f"worker {executor.worker_id!r} returned output without a local location"
                )
            result.transfer_ms += transfer_ms
        except Exception as error:
            await self._trace.record(
                "worker.execution.end",
                action_id=request.request_id,
                request_id=request.request_id,
                agent=request.agent,
                executor=executor.id,
                worker_id=executor.worker_id,
                success=False,
                error_type=type(error).__name__,
                error=str(error),
            )
            raise
        await self._trace.record(
            "worker.execution.end",
            action_id=request.request_id,
            request_id=request.request_id,
            agent=request.agent,
            executor=executor.id,
            worker_id=executor.worker_id,
            queue_ms=result.queue_ms,
            service_ms=result.service_ms,
            transfer_ms=result.transfer_ms,
            input_tokens=result.metadata.get("input_tokens", 0),
            output_tokens=result.metadata.get("output_tokens", 0),
            api_cost_usd=result.metadata.get("api_cost_usd", 0.0),
            output_artifacts=[artifact.id for artifact in result.output_artifacts],
            success=True,
        )
        for artifact in result.output_artifacts:
            await self._trace.record(
                "artifact.created",
                action_id=request.request_id,
                artifact_id=artifact.id,
                artifact_type=artifact.artifact_type,
                size_bytes=artifact.size_bytes,
                locations=artifact.locations,
                executor=executor.id,
            )
        return result

    async def sample_frames(
        self,
        request: SampleFramesRequest,
        target_worker_id: str | None = None,
    ) -> ExecutionResult:
        """Localize a video and execute the registered generic sampling operator."""
        target_worker_id = target_worker_id or await self.select_media_worker(
            [request.input_artifact], "sample_frames"
        )
        return await self._execute_media_operator(
            request_id=request.request_id,
            operator="sample_frames",
            inputs=[request.input_artifact],
            target_worker_id=target_worker_id,
            call=lambda client: client.sample_frames(request),
        )

    async def make_contact_sheet(
        self,
        request: MakeContactSheetRequest,
        target_worker_id: str | None = None,
    ) -> ExecutionResult:
        """Localize images and compose a contact sheet on the best local Worker."""
        target_worker_id = target_worker_id or await self.select_media_worker(
            request.input_artifacts, "make_contact_sheet"
        )
        return await self._execute_media_operator(
            request_id=request.request_id,
            operator="make_contact_sheet",
            inputs=request.input_artifacts,
            target_worker_id=target_worker_id,
            call=lambda client: client.make_contact_sheet(request),
        )

    async def extract_clip(
        self,
        request: ExtractClipRequest,
        target_worker_id: str | None = None,
    ) -> ExecutionResult:
        """Localize a video and extract a clip on the best local Worker."""
        target_worker_id = target_worker_id or await self.select_media_worker(
            [request.input_artifact], "extract_clip"
        )
        return await self._execute_media_operator(
            request_id=request.request_id,
            operator="extract_clip",
            inputs=[request.input_artifact],
            target_worker_id=target_worker_id,
            call=lambda client: client.extract_clip(request),
        )

    async def aggregate_artifacts(
        self,
        request: AggregateArtifactsRequest,
        target_worker_id: str | None = None,
    ) -> ExecutionResult:
        """Localize textual evidence and aggregate it on the best local Worker."""
        target_worker_id = target_worker_id or await self.select_media_worker(
            request.input_artifacts, "aggregate_artifacts"
        )
        return await self._execute_media_operator(
            request_id=request.request_id,
            operator="aggregate_artifacts",
            inputs=request.input_artifacts,
            target_worker_id=target_worker_id,
            call=lambda client: client.aggregate_artifacts(request),
        )

    async def select_media_worker(
        self, inputs: list[ArtifactRef], operator: str
    ) -> str:
        """Choose an operator-capable Worker by maximum already-local input bytes."""
        if not self._clients:
            raise WorkerUnavailableError(f"no Worker supports media operator {operator!r}")
        statuses = await asyncio.gather(
            *(client.status() for client in self._clients.values()),
            return_exceptions=True,
        )
        candidates = [
            worker_id
            for worker_id, status in zip(self._clients, statuses, strict=True)
            if not isinstance(status, BaseException) and operator in status.operators
        ]
        if not candidates:
            raise WorkerUnavailableError(f"no Worker supports media operator {operator!r}")
        return max(
            sorted(candidates),
            key=lambda worker_id: sum(
                artifact.size_bytes
                for artifact in inputs
                if worker_id in artifact.locations
            ),
        )

    async def _execute_media_operator(
        self,
        *,
        request_id: str,
        operator: str,
        inputs: list[ArtifactRef],
        target_worker_id: str,
        call: Callable[[WorkerClient], Awaitable[ExecutionResult]],
    ) -> ExecutionResult:
        """Run one worker-native operator with shared transfer and lineage tracing."""
        client = self._clients.get(target_worker_id)
        if client is None:
            raise WorkerUnavailableError(f"no client configured for worker {target_worker_id!r}")
        transfer_ms = 0.0
        for artifact in inputs:
            transfer = await self._transfer_manager.ensure_local(
                artifact,
                target_worker_id,
                action_id=request_id,
            )
            transfer_ms += transfer.transfer_ms
        executor_id = f"{target_worker_id}:{operator}"
        await self._trace.record(
            "executor.selected",
            action_id=request_id,
            request_id=request_id,
            agent=operator,
            executor=executor_id,
            worker_id=target_worker_id,
            semantic_operator=operator,
        )
        await self._trace.record(
            "worker.execution.start",
            action_id=request_id,
            request_id=request_id,
            agent=operator,
            executor=executor_id,
            worker_id=target_worker_id,
            semantic_operator=operator,
        )
        try:
            result = await call(client)
            if result.executor_id != executor_id:
                raise ExecutionFailedError(
                    f"worker executed {result.executor_id!r}, expected {executor_id!r}"
                )
            if any(
                target_worker_id not in artifact.locations
                for artifact in result.output_artifacts
            ):
                raise ExecutionFailedError(
                    f"worker {target_worker_id!r} returned output without a local location"
                )
            result.transfer_ms += transfer_ms
        except Exception as error:
            await self._trace.record(
                "worker.execution.end",
                action_id=request_id,
                request_id=request_id,
                agent=operator,
                executor=executor_id,
                worker_id=target_worker_id,
                semantic_operator=operator,
                success=False,
                error_type=type(error).__name__,
                error=str(error),
            )
            raise
        await self._trace.record(
            "worker.execution.end",
            action_id=request_id,
            request_id=request_id,
            agent=operator,
            executor=result.executor_id,
            worker_id=target_worker_id,
            queue_ms=result.queue_ms,
            service_ms=result.service_ms,
            transfer_ms=result.transfer_ms,
            input_tokens=0,
            output_tokens=0,
            api_cost_usd=0.0,
            semantic_operator=operator,
            output_artifacts=[artifact.id for artifact in result.output_artifacts],
            success=True,
        )
        for artifact in result.output_artifacts:
            await self._trace.record(
                "artifact.created",
                action_id=request_id,
                artifact_id=artifact.id,
                artifact_type=artifact.artifact_type,
                size_bytes=artifact.size_bytes,
                locations=artifact.locations,
                executor=result.executor_id,
            )
        return result
