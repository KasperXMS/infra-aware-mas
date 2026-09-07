"""Explicit artifact transfer management."""

import asyncio
from collections.abc import Mapping
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from infra_mas.core.artifact import ArtifactRef
from infra_mas.core.errors import ArtifactTransferError
from infra_mas.core.execution import TransferResult
from infra_mas.core.trace import TraceSink
from infra_mas.execution.worker_client import WorkerClient


class TransferManager:
    """Ensure referenced artifacts are physically available on target workers."""

    def __init__(
        self,
        clients: Mapping[str, WorkerClient],
        temporary_directory: Path,
        trace: TraceSink,
    ) -> None:
        self._clients = dict(clients)
        self._temporary_directory = temporary_directory.resolve()
        self._temporary_directory.mkdir(parents=True, exist_ok=True)
        self._trace = trace

    async def ensure_local(
        self,
        artifact: ArtifactRef,
        target_worker_id: str,
        *,
        action_id: str | None = None,
        parent_action_id: str | None = None,
    ) -> TransferResult:
        """Transfer an artifact when its location metadata does not include the target."""
        if target_worker_id in artifact.locations:
            return TransferResult(bytes_transferred=0, transfer_ms=0.0)

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
        temporary = self._temporary_directory / f"transfer-{uuid4().hex}.artifact"

        await self._trace.record(
            "artifact.transfer.start",
            action_id=action_id,
            parent_action_id=parent_action_id,
            artifact_id=artifact.id,
            source_worker_id=source_worker_id,
            target_worker_id=target_worker_id,
            expected_bytes=artifact.size_bytes,
        )
        started_at = perf_counter()
        bytes_transferred = 0

        try:
            bytes_transferred = await source.download_artifact(artifact.id, temporary)
            if bytes_transferred != artifact.size_bytes:
                raise ArtifactTransferError(
                    f"downloaded {bytes_transferred} bytes for {artifact.id!r}; "
                    f"expected {artifact.size_bytes}"
                )

            uploaded = await target.upload_artifact(artifact, temporary)
            if uploaded.id != artifact.id or uploaded.artifact_type != artifact.artifact_type:
                raise ArtifactTransferError(
                    f"target worker returned mismatched metadata for {artifact.id!r}"
                )
            if uploaded.size_bytes != artifact.size_bytes:
                raise ArtifactTransferError(
                    f"uploaded {uploaded.size_bytes} bytes for {artifact.id!r}; "
                    f"expected {artifact.size_bytes}"
                )

            transfer_ms = (perf_counter() - started_at) * 1000
            artifact.locations = [*artifact.locations, target_worker_id]
            result = TransferResult(
                bytes_transferred=bytes_transferred,
                transfer_ms=transfer_ms,
            )
            await self._trace.record(
                "artifact.transfer.end",
                action_id=action_id,
                parent_action_id=parent_action_id,
                artifact_id=artifact.id,
                source_worker_id=source_worker_id,
                target_worker_id=target_worker_id,
                bytes_transferred=result.bytes_transferred,
                transfer_ms=result.transfer_ms,
                success=True,
            )
            return result
        except Exception as error:
            transfer_ms = (perf_counter() - started_at) * 1000
            await self._trace.record(
                "artifact.transfer.end",
                action_id=action_id,
                parent_action_id=parent_action_id,
                artifact_id=artifact.id,
                source_worker_id=source_worker_id,
                target_worker_id=target_worker_id,
                bytes_transferred=bytes_transferred,
                transfer_ms=transfer_ms,
                success=False,
                error_type=type(error).__name__,
                error=str(error),
            )
            if isinstance(error, ArtifactTransferError):
                raise
            raise ArtifactTransferError(
                f"failed to transfer {artifact.id!r} from {source_worker_id!r} "
                f"to {target_worker_id!r}: {error}"
            ) from error
        finally:
            if await asyncio.to_thread(temporary.is_file):
                await asyncio.to_thread(temporary.unlink)
