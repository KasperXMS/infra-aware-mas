"""Worker-local artifact storage."""

import asyncio
import os
from collections.abc import AsyncIterable
from pathlib import Path, PurePosixPath
from uuid import uuid4

from infra_mas.core.artifact import ArtifactRef
from infra_mas.core.errors import ArtifactNotFoundError


class ArtifactStore:
    """Persist artifacts beneath one worker-owned local directory."""

    def __init__(self, root: Path, worker_id: str) -> None:
        if not worker_id.strip():
            raise ValueError("worker_id must not be empty")

        self._root = root.resolve()
        self._root.mkdir(parents=True, exist_ok=True)
        self._worker_id = worker_id.strip()

    @property
    def worker_id(self) -> str:
        """Return the physical worker that owns this store."""
        return self._worker_id

    async def put_bytes(
        self,
        artifact_id: str,
        data: bytes,
        artifact_type: str = "application/octet-stream",
    ) -> ArtifactRef:
        """Atomically persist an in-memory artifact."""
        target = self._path_for(artifact_id)
        await asyncio.to_thread(target.parent.mkdir, parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")

        try:
            await asyncio.to_thread(temporary.write_bytes, data)
            await asyncio.to_thread(os.replace, temporary, target)
        finally:
            if temporary.exists():
                await asyncio.to_thread(temporary.unlink)

        return self._ref(artifact_id, artifact_type, len(data))

    async def put_text(
        self,
        artifact_id: str,
        text: str,
        artifact_type: str = "text/plain",
    ) -> ArtifactRef:
        """UTF-8 encode and persist a textual artifact."""
        return await self.put_bytes(artifact_id, text.encode("utf-8"), artifact_type)

    async def put_stream(
        self,
        artifact_id: str,
        chunks: AsyncIterable[bytes],
        artifact_type: str = "application/octet-stream",
    ) -> ArtifactRef:
        """Atomically persist an artifact without buffering the complete body."""
        target = self._path_for(artifact_id)
        await asyncio.to_thread(target.parent.mkdir, parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
        handle = await asyncio.to_thread(temporary.open, "wb")
        size_bytes = 0

        try:
            async for chunk in chunks:
                if chunk:
                    await asyncio.to_thread(handle.write, chunk)
                    size_bytes += len(chunk)
            await asyncio.to_thread(handle.close)
            await asyncio.to_thread(os.replace, temporary, target)
        except BaseException:
            await asyncio.to_thread(handle.close)
            raise
        finally:
            if temporary.exists():
                await asyncio.to_thread(temporary.unlink)

        return self._ref(artifact_id, artifact_type, size_bytes)

    async def get_path(self, artifact_id: str) -> Path:
        """Resolve an artifact ID to an existing worker-local path."""
        path = self._path_for(artifact_id)
        if not await asyncio.to_thread(path.is_file):
            raise ArtifactNotFoundError(f"artifact not found: {artifact_id}")
        return path

    async def exists(self, artifact_id: str) -> bool:
        """Return whether an artifact is locally available."""
        return await asyncio.to_thread(self._path_for(artifact_id).is_file)

    async def delete(self, artifact_id: str) -> None:
        """Remove one local artifact when a transfer fails integrity validation."""
        path = self._path_for(artifact_id)
        if await asyncio.to_thread(path.is_file):
            await asyncio.to_thread(path.unlink)

    def _path_for(self, artifact_id: str) -> Path:
        normalized = PurePosixPath(artifact_id)
        raw_parts = artifact_id.split("/")
        if (
            not artifact_id
            or "\\" in artifact_id
            or normalized.is_absolute()
            or any(part in {"", ".", ".."} or ":" in part for part in raw_parts)
        ):
            raise ValueError(f"invalid artifact ID: {artifact_id!r}")

        path = self._root.joinpath(*normalized.parts).resolve()
        if not path.is_relative_to(self._root):
            raise ValueError(f"artifact ID escapes store root: {artifact_id!r}")
        return path

    def _ref(self, artifact_id: str, artifact_type: str, size_bytes: int) -> ArtifactRef:
        return ArtifactRef(
            id=artifact_id,
            artifact_type=artifact_type,
            size_bytes=size_bytes,
            locations=[self._worker_id],
        )
