"""Execution manager."""

from collections.abc import Mapping

from infra_mas.core.errors import ExecutionFailedError, WorkerUnavailableError
from infra_mas.core.execution import ExecutionRequest, ExecutionResult
from infra_mas.core.executor import ExecutorSpec
from infra_mas.execution.transfer import TransferManager
from infra_mas.execution.worker_client import WorkerClient


class ExecutionManager:
    """Execute an already scheduled request on its assigned physical executor."""

    def __init__(
        self,
        clients: Mapping[str, WorkerClient],
        transfer_manager: TransferManager,
    ) -> None:
        self._clients = dict(clients)
        self._transfer_manager = transfer_manager

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
        return result
