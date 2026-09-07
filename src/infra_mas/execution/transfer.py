"""Explicit direct Worker-to-Worker artifact transfer management."""

import asyncio
from collections.abc import Mapping
from time import perf_counter

from infra_mas.core.artifact import ArtifactRef
from infra_mas.core.errors import ArtifactTransferError
from infra_mas.core.execution import ArtifactPullRequest, TransferResult
from infra_mas.core.trace import TraceSink
from infra_mas.execution.worker_client import WorkerClient


class TransferManager:
    """Instruct target Workers to pull artifacts without relaying bytes through Controller."""

    def __init__(
        self,
        clients: Mapping[str, WorkerClient],
        trace: TraceSink,
    ) -> None:
        self._clients = dict(clients)
        self._trace = trace
        self._locks: dict[tuple[str, str], asyncio.Lock] = {}
        self._completed: set[tuple[str, str]] = set()
        self._locks_guard = asyncio.Lock()

    async def ensure_local(
        self,
        artifact: ArtifactRef,
        target_worker_id: str,
        *,
        action_id: str | None = None,
        parent_action_id: str | None = None,
    ) -> TransferResult:
        """Have the target pull an artifact once when it is not already local."""
        if target_worker_id in artifact.locations:
            return TransferResult(bytes_transferred=0, transfer_ms=0.0)

        key = (artifact.id, target_worker_id)
        lock = await self._lock_for(*key)
        async with lock:
            if key in self._completed:
                if target_worker_id not in artifact.locations:
                    artifact.locations = [*artifact.locations, target_worker_id]
                return TransferResult(bytes_transferred=0, transfer_ms=0.0)
            if target_worker_id in artifact.locations:
                self._completed.add(key)
                return TransferResult(bytes_transferred=0, transfer_ms=0.0)
            result = await self._transfer(
                artifact,
                target_worker_id,
                action_id=action_id,
                parent_action_id=parent_action_id,
            )
            self._completed.add(key)
            return result

    async def _transfer(
        self,
        artifact: ArtifactRef,
        target_worker_id: str,
        *,
        action_id: str | None,
        parent_action_id: str | None,
    ) -> TransferResult:
        target = self._clients.get(target_worker_id)
        if target is None:
            raise ArtifactTransferError(f"unknown target worker: {target_worker_id}")

        source_worker_id = next(
            (location for location in artifact.locations if location in self._clients),
            None,
        )
        if source_worker_id is None:
            raise ArtifactTransferError(
                f"no reachable source for artifact {artifact.id!r}: {artifact.locations}"
            )
        source = self._clients[source_worker_id]

        await self._trace.record(
            "artifact.transfer.start",
            action_id=action_id,
            parent_action_id=parent_action_id,
            artifact_id=artifact.id,
            source_worker_id=source_worker_id,
            target_worker_id=target_worker_id,
            expected_bytes=artifact.size_bytes,
            path="worker_to_worker",
        )
        started_at = perf_counter()
        try:
            result = await target.pull_artifact(
                ArtifactPullRequest(
                    artifact=artifact,
                    source_worker_id=source_worker_id,
                    source_endpoint=source.base_url,
                )
            )
            if result.bytes_transferred not in {0, artifact.size_bytes}:
                raise ArtifactTransferError(
                    f"target pulled {result.bytes_transferred} bytes for {artifact.id!r}; "
                    f"expected {artifact.size_bytes}"
                )
            artifact.locations = [*artifact.locations, target_worker_id]
            await self._trace.record(
                "artifact.transfer.end",
                action_id=action_id,
                parent_action_id=parent_action_id,
                artifact_id=artifact.id,
                source_worker_id=source_worker_id,
                target_worker_id=target_worker_id,
                bytes_transferred=result.bytes_transferred,
                transfer_ms=result.transfer_ms,
                path="worker_to_worker",
                success=True,
            )
            return result
        except Exception as error:
            await self._trace.record(
                "artifact.transfer.end",
                action_id=action_id,
                parent_action_id=parent_action_id,
                artifact_id=artifact.id,
                source_worker_id=source_worker_id,
                target_worker_id=target_worker_id,
                bytes_transferred=0,
                transfer_ms=(perf_counter() - started_at) * 1000,
                path="worker_to_worker",
                success=False,
                error_type=type(error).__name__,
                error=str(error),
            )
            if isinstance(error, ArtifactTransferError):
                raise
            raise ArtifactTransferError(
                f"failed direct transfer of {artifact.id!r} from {source_worker_id!r} "
                f"to {target_worker_id!r}: {error}"
            ) from error

    async def _lock_for(self, artifact_id: str, target_worker_id: str) -> asyncio.Lock:
        key = (artifact_id, target_worker_id)
        async with self._locks_guard:
            lock = self._locks.get(key)
            if lock is None:
                lock = asyncio.Lock()
                self._locks[key] = lock
            return lock
