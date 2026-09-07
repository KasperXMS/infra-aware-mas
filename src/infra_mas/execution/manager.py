"""Execution manager."""

from collections.abc import Mapping

from infra_mas.core.errors import ExecutionFailedError, WorkerUnavailableError
from infra_mas.core.execution import ExecutionRequest, ExecutionResult
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
            compute_ms=result.compute_ms,
            transfer_ms=result.transfer_ms,
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
