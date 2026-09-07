"""Worker execution service."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable
from uuid import uuid4

from pydantic import ValidationError

from infra_mas.core.errors import ExecutionFailedError, InvalidModelResponseError
from infra_mas.core.execution import ExecutionRequest, ExecutionResult, WorkerStatus
from infra_mas.core.model import ModelRequest, ModelResult
from infra_mas.worker.artifact_store import ArtifactStore
from infra_mas.worker.backends.base import ModelBackend


@dataclass(frozen=True, slots=True)
class WorkerExecutor:
    """Bind one local executor identity to a capability and backend."""

    id: str
    capability: str
    backend: ModelBackend


ArtifactIdFactory = Callable[[ExecutionRequest], str]


@runtime_checkable
class _AsyncClosable(Protocol):
    async def aclose(self) -> None: ...


@runtime_checkable
class _AsyncCheckable(Protocol):
    async def check(self) -> None: ...


class WorkerService:
    """Resolve local inputs, invoke a backend, and persist its output."""

    def __init__(
        self,
        worker_id: str,
        artifact_store: ArtifactStore,
        executors: Iterable[WorkerExecutor],
        artifact_id_factory: ArtifactIdFactory | None = None,
    ) -> None:
        if not worker_id.strip():
            raise ValueError("worker_id must not be empty")
        if artifact_store.worker_id != worker_id:
            raise ValueError("artifact store worker_id does not match worker service")

        executor_list = list(executors)
        executor_ids = [executor.id for executor in executor_list]
        if any(not executor_id.strip() for executor_id in executor_ids):
            raise ValueError("executor IDs must not be empty")
        if any(not executor.capability.strip() for executor in executor_list):
            raise ValueError("executor capabilities must not be empty")
        if len(executor_ids) != len(set(executor_ids)):
            raise ValueError("executor IDs must be unique within a worker")

        self._worker_id = worker_id
        self._artifact_store = artifact_store
        self._executors = {executor.id: executor for executor in executor_list}
        self._artifact_id_factory = artifact_id_factory or self._default_artifact_id

    @property
    def artifact_store(self) -> ArtifactStore:
        """Return the worker-local artifact store for HTTP data-plane endpoints."""
        return self._artifact_store

    def status(self) -> WorkerStatus:
        """Return static worker and executor identity information."""
        return WorkerStatus(
            worker_id=self._worker_id,
            executors=[executor.id for executor in self._executors.values()],
        )

    async def aclose(self) -> None:
        """Close executor backends that own asynchronous resources."""
        for executor in self._executors.values():
            if isinstance(executor.backend, _AsyncClosable):
                await executor.backend.aclose()

    async def check_backends(self) -> None:
        """Check all backends that provide an explicit readiness probe."""
        for executor in self._executors.values():
            if isinstance(executor.backend, _AsyncCheckable):
                await executor.backend.check()

    async def execute(
        self,
        request: ExecutionRequest,
        executor_id: str | None = None,
    ) -> ExecutionResult:
        """Execute a semantic request using a selected or uniquely compatible backend."""
        executor = self._resolve_executor(request.capability, executor_id)

        input_paths = [str(await self._artifact_store.get_path(item.id)) for item in request.inputs]
        model_request = ModelRequest(task=request.task, input_paths=input_paths)

        try:
            backend_result = await executor.backend.infer(model_request)
            model_result = ModelResult.model_validate(backend_result)
        except ExecutionFailedError:
            raise
        except ValidationError as error:
            raise InvalidModelResponseError("model backend returned an invalid result") from error
        except Exception as error:
            raise ExecutionFailedError(f"model execution failed: {error}") from error

        output = await self._artifact_store.put_text(
            artifact_id=self._artifact_id_factory(request),
            text=model_result.output_text,
        )
        return ExecutionResult(
            request_id=request.request_id,
            executor_id=executor.id,
            output_artifacts=[output],
            queue_ms=0.0,
            compute_ms=model_result.latency_ms,
        )

    def _resolve_executor(self, capability: str, executor_id: str | None) -> WorkerExecutor:
        if executor_id is not None:
            executor = self._executors.get(executor_id)
            if executor is None:
                raise ExecutionFailedError(
                    f"executor {executor_id!r} is not hosted by worker {self._worker_id!r}"
                )
            if executor.capability != capability:
                raise ExecutionFailedError(
                    f"executor {executor_id!r} does not support capability {capability!r}"
                )
            return executor

        candidates = [
            executor for executor in self._executors.values() if executor.capability == capability
        ]
        if not candidates:
            raise ExecutionFailedError(
                f"worker {self._worker_id!r} has no executor for {capability!r}"
            )
        if len(candidates) > 1:
            raise ExecutionFailedError(
                f"worker {self._worker_id!r} requires an executor selection for {capability!r}"
            )
        return candidates[0]

    @staticmethod
    def _default_artifact_id(request: ExecutionRequest) -> str:
        run_id = request.request_id.rsplit("/", maxsplit=1)[0]
        return f"{run_id}/output-{uuid4().hex}"
