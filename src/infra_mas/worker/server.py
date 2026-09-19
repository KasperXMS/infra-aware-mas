"""Worker HTTP server."""

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from infra_mas.core.artifact import ArtifactRef
from infra_mas.core.errors import ArtifactNotFoundError, ArtifactTransferError, ExecutionFailedError
from infra_mas.core.execution import (
    AggregateArtifactsRequest,
    AggregateRecordsRequest,
    ArtifactPullRequest,
    BindLocalArtifactRequest,
    BM25RetrieveRequest,
    DeriveFieldsRequest,
    ExecutionRequest,
    ExecutionResult,
    ExtractClipRequest,
    FilterRecordsRequest,
    HealthResponse,
    MakeContactSheetRequest,
    SampleFramesRequest,
    SelectFieldsRequest,
    TopKRecordsRequest,
    TransferResult,
    WorkerStatus,
)
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

    async def pull_artifact(pull_request: ArtifactPullRequest) -> TransferResult:
        try:
            return await service.pull_artifact(pull_request)
        except ArtifactTransferError as error:
            raise HTTPException(status_code=502, detail=str(error)) from error

    async def sample_frames(request: SampleFramesRequest) -> ExecutionResult:
        try:
            return await service.sample_frames(request)
        except ArtifactNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ExecutionFailedError as error:
            raise HTTPException(status_code=502, detail=str(error)) from error

    async def make_contact_sheet(request: MakeContactSheetRequest) -> ExecutionResult:
        try:
            return await service.make_contact_sheet(request)
        except ArtifactNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ExecutionFailedError as error:
            raise HTTPException(status_code=502, detail=str(error)) from error

    async def extract_clip(request: ExtractClipRequest) -> ExecutionResult:
        try:
            return await service.extract_clip(request)
        except ArtifactNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ExecutionFailedError as error:
            raise HTTPException(status_code=502, detail=str(error)) from error

    async def aggregate_artifacts(request: AggregateArtifactsRequest) -> ExecutionResult:
        try:
            return await service.aggregate_artifacts(request)
        except ArtifactNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ExecutionFailedError as error:
            raise HTTPException(status_code=502, detail=str(error)) from error

    async def bm25_retrieve(request: BM25RetrieveRequest) -> ExecutionResult:
        try:
            return await service.bm25_retrieve(request)
        except ArtifactNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ExecutionFailedError as error:
            raise HTTPException(status_code=502, detail=str(error)) from error

    async def filter_records(request: FilterRecordsRequest) -> ExecutionResult:
        try:
            return await service.filter_records(request)
        except ArtifactNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ExecutionFailedError as error:
            raise HTTPException(status_code=502, detail=str(error)) from error

    async def select_fields(request: SelectFieldsRequest) -> ExecutionResult:
        try:
            return await service.select_fields(request)
        except ArtifactNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ExecutionFailedError as error:
            raise HTTPException(status_code=502, detail=str(error)) from error

    async def aggregate_records(request: AggregateRecordsRequest) -> ExecutionResult:
        try:
            return await service.aggregate_records(request)
        except ArtifactNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ExecutionFailedError as error:
            raise HTTPException(status_code=502, detail=str(error)) from error

    async def derive_fields(request: DeriveFieldsRequest) -> ExecutionResult:
        try:
            return await service.derive_fields(request)
        except ArtifactNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ExecutionFailedError as error:
            raise HTTPException(status_code=502, detail=str(error)) from error

    async def top_k_records(request: TopKRecordsRequest) -> ExecutionResult:
        try:
            return await service.top_k_records(request)
        except ArtifactNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ExecutionFailedError as error:
            raise HTTPException(status_code=502, detail=str(error)) from error

    async def bind_local_artifact(request: BindLocalArtifactRequest) -> ArtifactRef:
        try:
            return await service.bind_local_artifact(request)
        except ArtifactNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ExecutionFailedError as error:
            raise HTTPException(status_code=403, detail=str(error)) from error

    app.add_api_route("/health", health, methods=["GET"], response_model=HealthResponse)
    app.add_api_route("/ready", ready, methods=["GET"], response_model=HealthResponse)
    app.add_api_route("/status", status, methods=["GET"], response_model=WorkerStatus)
    app.add_api_route("/execute", execute, methods=["POST"], response_model=ExecutionResult)
    app.add_api_route(
        "/operators/sample-frames",
        sample_frames,
        methods=["POST"],
        response_model=ExecutionResult,
    )
    app.add_api_route(
        "/operators/make-contact-sheet",
        make_contact_sheet,
        methods=["POST"],
        response_model=ExecutionResult,
    )
    app.add_api_route(
        "/operators/extract-clip",
        extract_clip,
        methods=["POST"],
        response_model=ExecutionResult,
    )
    app.add_api_route(
        "/operators/aggregate-artifacts",
        aggregate_artifacts,
        methods=["POST"],
        response_model=ExecutionResult,
    )
    app.add_api_route(
        "/operators/bm25-retrieve",
        bm25_retrieve,
        methods=["POST"],
        response_model=ExecutionResult,
    )
    app.add_api_route(
        "/operators/filter-records",
        filter_records,
        methods=["POST"],
        response_model=ExecutionResult,
    )
    app.add_api_route(
        "/operators/select-fields",
        select_fields,
        methods=["POST"],
        response_model=ExecutionResult,
    )
    app.add_api_route(
        "/operators/aggregate-records",
        aggregate_records,
        methods=["POST"],
        response_model=ExecutionResult,
    )
    app.add_api_route(
        "/operators/derive-fields",
        derive_fields,
        methods=["POST"],
        response_model=ExecutionResult,
    )
    app.add_api_route(
        "/operators/top-k-records",
        top_k_records,
        methods=["POST"],
        response_model=ExecutionResult,
    )
    app.add_api_route(
        "/artifacts/bind-local",
        bind_local_artifact,
        methods=["POST"],
        response_model=ArtifactRef,
    )
    app.add_api_route("/artifacts/{artifact_id:path}", download_artifact, methods=["GET"])
    app.add_api_route("/artifacts", upload_artifact, methods=["POST"], response_model=ArtifactRef)
    app.add_api_route(
        "/transfers/pull",
        pull_artifact,
        methods=["POST"],
        response_model=TransferResult,
    )

    return app


async def _read_chunks(path: Path) -> AsyncIterator[bytes]:
    handle = await asyncio.to_thread(path.open, "rb")
    try:
        while chunk := await asyncio.to_thread(handle.read, STREAM_CHUNK_SIZE):
            yield chunk
    finally:
        await asyncio.to_thread(handle.close)
