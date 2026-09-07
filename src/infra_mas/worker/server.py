"""Worker HTTP server."""

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from infra_mas.core.artifact import ArtifactRef
from infra_mas.core.errors import ArtifactNotFoundError, ExecutionFailedError
from infra_mas.core.execution import ExecutionRequest, ExecutionResult, HealthResponse, WorkerStatus
from infra_mas.worker.service import WorkerService

STREAM_CHUNK_SIZE = 64 * 1024


def create_app(service: WorkerService) -> FastAPI:
    """Create a worker API around an injected execution service."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
        del app
        yield
        await service.aclose()

    app = FastAPI(title="Infra-Aware MAS Worker", lifespan=lifespan)

    async def health() -> HealthResponse:
        return HealthResponse()

    async def status() -> WorkerStatus:
        return service.status()

    async def ready() -> HealthResponse:
        try:
            await service.check_backends()
        except ExecutionFailedError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        return HealthResponse()

    async def execute(
        execution_request: ExecutionRequest,
        executor_id: Annotated[str | None, Header(alias="X-Executor-ID")] = None,
    ) -> ExecutionResult:
        try:
            return await service.execute(execution_request, executor_id=executor_id)
        except ArtifactNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ExecutionFailedError as error:
            raise HTTPException(status_code=502, detail=str(error)) from error

    async def download_artifact(artifact_id: str) -> StreamingResponse:
        try:
            path = await service.artifact_store.get_path(artifact_id)
        except ArtifactNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

        return StreamingResponse(
            _read_chunks(path),
            media_type="application/octet-stream",
            headers={"Content-Length": str(path.stat().st_size)},
        )

    async def upload_artifact(
        request: Request,
        artifact_id: Annotated[str, Query(min_length=1)],
        artifact_type: Annotated[str, Query(min_length=1)] = "application/octet-stream",
    ) -> ArtifactRef:
        try:
            return await service.artifact_store.put_stream(
                artifact_id=artifact_id,
                chunks=request.stream(),
                artifact_type=artifact_type,
            )
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    app.add_api_route("/health", health, methods=["GET"], response_model=HealthResponse)
    app.add_api_route("/ready", ready, methods=["GET"], response_model=HealthResponse)
    app.add_api_route("/status", status, methods=["GET"], response_model=WorkerStatus)
    app.add_api_route("/execute", execute, methods=["POST"], response_model=ExecutionResult)
    app.add_api_route("/artifacts/{artifact_id:path}", download_artifact, methods=["GET"])
    app.add_api_route("/artifacts", upload_artifact, methods=["POST"], response_model=ArtifactRef)

    return app


async def _read_chunks(path: Path) -> AsyncIterator[bytes]:
    handle = await asyncio.to_thread(path.open, "rb")
    try:
        while chunk := await asyncio.to_thread(handle.read, STREAM_CHUNK_SIZE):
            yield chunk
    finally:
        await asyncio.to_thread(handle.close)
