"""Asynchronous worker HTTP client."""

import asyncio
import os
from collections.abc import AsyncIterable, AsyncIterator, Mapping
from pathlib import Path
from types import TracebackType
from typing import cast
from urllib.parse import quote
from uuid import uuid4

import httpx

from infra_mas.core.artifact import ArtifactRef
from infra_mas.core.errors import (
    ArtifactNotFoundError,
    ExecutionFailedError,
    WorkerUnavailableError,
)
from infra_mas.core.execution import (
    ExecutionRequest,
    ExecutionResult,
    HealthResponse,
    WorkerStatus,
)

STREAM_CHUNK_SIZE = 64 * 1024


class WorkerClient:
    """Keep worker networking behind one reusable asynchronous HTTP client."""

    def __init__(
        self,
        base_url: str | None = None,
        *,
        client: httpx.AsyncClient | None = None,
        timeout: float = 30.0,
    ) -> None:
        if client is None:
            if base_url is None:
                raise ValueError("base_url is required when client is not provided")
            self._client = httpx.AsyncClient(base_url=base_url, timeout=timeout)
            self._owns_client = True
        else:
            self._client = client
            self._owns_client = False

    async def __aenter__(self) -> "WorkerClient":
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback
        await self.aclose()

    async def aclose(self) -> None:
        """Close the internally owned HTTP connection pool."""
        if self._owns_client:
            await self._client.aclose()

    async def health(self) -> HealthResponse:
        """Fetch and validate worker health."""
        response = await self._request("GET", "/health")
        return HealthResponse.model_validate(response.json())

    async def ready(self) -> HealthResponse:
        """Verify Worker readiness, including configured model endpoints."""
        response = await self._request("GET", "/ready")
        return HealthResponse.model_validate(response.json())

    async def status(self) -> WorkerStatus:
        """Fetch and validate worker status."""
        response = await self._request("GET", "/status")
        return WorkerStatus.model_validate(response.json())

    async def execute(
        self,
        request: ExecutionRequest,
        executor_id: str | None = None,
    ) -> ExecutionResult:
        """Submit a request with an optional scheduler-selected executor binding."""
        headers = {"X-Executor-ID": executor_id} if executor_id is not None else None
        response = await self._request(
            "POST",
            "/execute",
            json_payload=request.model_dump(mode="json"),
            headers=headers,
            not_found_is_artifact=True,
        )
        return ExecutionResult.model_validate(response.json())

    async def download_artifact(self, artifact_id: str, destination: Path) -> int:
        """Stream an artifact into a local file and return transferred bytes."""
        path = f"/artifacts/{quote(artifact_id, safe='/')}"
        temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
        await asyncio.to_thread(destination.parent.mkdir, parents=True, exist_ok=True)
        size_bytes = 0

        try:
            async with self._client.stream("GET", path) as response:
                if not response.is_success:
                    await response.aread()
                self._raise_for_status(
                    response,
                    artifact_id=artifact_id,
                    not_found_is_artifact=True,
                )
                handle = await asyncio.to_thread(temporary.open, "wb")
                try:
                    async for chunk in response.aiter_bytes(STREAM_CHUNK_SIZE):
                        await asyncio.to_thread(handle.write, chunk)
                        size_bytes += len(chunk)
                finally:
                    await asyncio.to_thread(handle.close)
            await asyncio.to_thread(os.replace, temporary, destination)
        except httpx.RequestError as error:
            raise WorkerUnavailableError(str(error)) from error
        finally:
            if temporary.exists():
                await asyncio.to_thread(temporary.unlink)

        return size_bytes

    async def upload_artifact(self, artifact: ArtifactRef, source: Path) -> ArtifactRef:
        """Stream one local file into the worker artifact store."""
        if not await asyncio.to_thread(source.is_file):
            raise ArtifactNotFoundError(f"artifact source not found: {source}")

        response = await self._request(
            "POST",
            "/artifacts",
            params={"artifact_id": artifact.id, "artifact_type": artifact.artifact_type},
            content=self._file_chunks(source),
        )
        return ArtifactRef.model_validate(response.json())

    async def _request(
        self,
        method: str,
        url: str,
        *,
        json_payload: object | None = None,
        params: Mapping[str, str] | None = None,
        headers: Mapping[str, str] | None = None,
        content: bytes | AsyncIterable[bytes] | None = None,
        not_found_is_artifact: bool = False,
    ) -> httpx.Response:
        try:
            response = await self._client.request(
                method,
                url,
                json=json_payload,
                params=params,
                headers=headers,
                content=content,
            )
        except httpx.RequestError as error:
            raise WorkerUnavailableError(str(error)) from error
        self._raise_for_status(response, not_found_is_artifact=not_found_is_artifact)
        return response

    @staticmethod
    def _raise_for_status(
        response: httpx.Response,
        artifact_id: str | None = None,
        *,
        not_found_is_artifact: bool = False,
    ) -> None:
        if response.is_success:
            return

        detail = response.text
        try:
            payload: object = response.json()
            if isinstance(payload, dict):
                payload_dict = cast(dict[str, object], payload)
                detail_value = payload_dict.get("detail")
                if isinstance(detail_value, str):
                    detail = detail_value
        except ValueError:
            pass

        if response.status_code == 404 and not_found_is_artifact:
            if artifact_id is not None:
                detail = f"artifact not found: {artifact_id}"
            raise ArtifactNotFoundError(detail)
        raise ExecutionFailedError(
            f"worker request failed with HTTP {response.status_code}: {detail}"
        )

    @staticmethod
    async def _file_chunks(source: Path) -> AsyncIterator[bytes]:
        handle = await asyncio.to_thread(source.open, "rb")
        try:
            while chunk := await asyncio.to_thread(handle.read, STREAM_CHUNK_SIZE):
                yield chunk
        finally:
            await asyncio.to_thread(handle.close)
