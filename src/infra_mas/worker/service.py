"""Worker execution service."""

import asyncio
import io
import math
import shutil
import tempfile
from collections.abc import AsyncIterator, Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Literal, Protocol, runtime_checkable
from urllib.parse import quote
from uuid import uuid4

import httpx
from PIL import Image, ImageDraw
from pydantic import ValidationError

from infra_mas.core.artifact import ArtifactRef
from infra_mas.core.errors import (
    ArtifactTransferError,
    ExecutionFailedError,
    InvalidModelResponseError,
)
from infra_mas.core.execution import (
    ArtifactPullRequest,
    ExecutionRequest,
    ExecutionResult,
    SampleFramesRequest,
    TransferResult,
    WorkerStatus,
)
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


def _compose_contact_sheet(
    frame_paths: list[Path],
    columns: int,
    duration_s: float,
) -> bytes:
    """Pack fixed chronological samples into one labelled generic JPEG artifact."""
    loaded: list[Image.Image] = []
    try:
        for path in frame_paths:
            with Image.open(path) as image:
                loaded.append(image.convert("RGB"))
        if not loaded:
            raise ExecutionFailedError("sample_frames cannot compose an empty contact sheet")
        tile_width = max(image.width for image in loaded)
        tile_height = max(image.height for image in loaded)
        active_columns = min(columns, len(loaded))
        rows = math.ceil(len(loaded) / active_columns)
        label_height = 24
        sheet = Image.new(
            "RGB",
            (active_columns * tile_width, rows * (tile_height + label_height)),
            color="black",
        )
        draw = ImageDraw.Draw(sheet)
        for index, image in enumerate(loaded):
            column = index % active_columns
            row = index // active_columns
            x = column * tile_width
            y = row * (tile_height + label_height)
            sheet.paste(image, (x, y))
            timestamp_s = duration_s * (index + 0.5) / len(loaded)
            draw.text(
                (x + 6, y + tile_height + 4),
                f"sample {index + 1:02d} @ {timestamp_s:.0f} sec",
                fill="white",
            )
        buffer = io.BytesIO()
        sheet.save(buffer, format="JPEG", quality=90, optimize=True)
        sheet.close()
        return buffer.getvalue()
    finally:
        for image in loaded:
            image.close()


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
        transfer_client: httpx.AsyncClient | None = None,
        ffmpeg_path: str = "ffmpeg",
        gstreamer_path: str = "gst-launch-1.0",
        frame_sampler: Literal["auto", "ffmpeg", "gstreamer"] = "auto",
        gstreamer_converter: Literal["auto", "nvvidconv", "videoconvert"] = "auto",
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
        self._transfer_client = transfer_client
        self._ffmpeg_path = ffmpeg_path
        self._gstreamer_path = gstreamer_path
        self._frame_sampler = frame_sampler
        self._gstreamer_converter = gstreamer_converter

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
        model_request = ModelRequest(
            instructions=request.instructions,
            task=request.task,
            input_paths=input_paths,
        )

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
            service_ms=model_result.latency_ms,
            metadata={
                "input_tokens": model_result.input_tokens,
                "output_tokens": model_result.output_tokens,
                "api_cost_usd": model_result.api_cost_usd,
            },
        )

    async def pull_artifact(self, request: ArtifactPullRequest) -> TransferResult:
        """Pull an artifact directly from its source Worker into this Worker."""
        artifact = request.artifact
        if request.source_worker_id not in artifact.locations:
            raise ArtifactTransferError(
                f"artifact {artifact.id!r} is not located on source {request.source_worker_id!r}"
            )
        if await self._artifact_store.exists(artifact.id):
            local_path = await self._artifact_store.get_path(artifact.id)
            if local_path.stat().st_size != artifact.size_bytes:
                raise ArtifactTransferError(f"local artifact {artifact.id!r} has unexpected size")
            return TransferResult(bytes_transferred=0, transfer_ms=0.0)

        url = f"{request.source_endpoint.rstrip('/')}/artifacts/{quote(artifact.id, safe='/')}"
        started_at = perf_counter()
        try:
            if request.rtt_ms:
                await asyncio.sleep(request.rtt_ms / 1000.0)
            if self._transfer_client is None:
                async with httpx.AsyncClient(timeout=300.0) as client:
                    uploaded = await self._pull_with_client(client, url, request)
            else:
                uploaded = await self._pull_with_client(self._transfer_client, url, request)
        except ArtifactTransferError:
            raise
        except httpx.RequestError as error:
            raise ArtifactTransferError(
                f"source Worker {request.source_worker_id!r} is unavailable: {error}"
            ) from error
        except Exception as error:
            raise ArtifactTransferError(
                f"failed to pull artifact {artifact.id!r}: {error}"
            ) from error

        if uploaded.size_bytes != artifact.size_bytes:
            await self._artifact_store.delete(artifact.id)
            raise ArtifactTransferError(
                f"pulled {uploaded.size_bytes} bytes for {artifact.id!r}; "
                f"expected {artifact.size_bytes}"
            )
        return TransferResult(
            bytes_transferred=uploaded.size_bytes,
            transfer_ms=(perf_counter() - started_at) * 1000,
        )

    async def sample_frames(self, request: SampleFramesRequest) -> ExecutionResult:
        """Run fixed uniform sampling locally and return JPEG frame artifacts."""
        source = await self._artifact_store.get_path(request.input_artifact.id)
        if not request.input_artifact.artifact_type.startswith("video/"):
            raise ExecutionFailedError("sample_frames requires a video artifact")
        started_at = perf_counter()
        sampler = self._frame_sampler
        if sampler == "auto":
            if shutil.which(self._ffmpeg_path):
                sampler = "ffmpeg"
            elif shutil.which(self._gstreamer_path):
                sampler = "gstreamer"
            else:
                raise ExecutionFailedError("neither ffmpeg nor gst-launch-1.0 is available")
        with tempfile.TemporaryDirectory(prefix="infra-mas-frames-") as directory:
            frame_pattern = str(Path(directory) / "frame-%03d.jpg")
            fps = request.sample_count / request.duration_s
            if sampler == "ffmpeg":
                process = await asyncio.create_subprocess_exec(
                    self._ffmpeg_path,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    str(source),
                    "-vf",
                    f"fps={fps:.12f},scale={request.frame_width}:-2",
                    "-frames:v",
                    str(request.sample_count),
                    "-q:v",
                    "2",
                    frame_pattern,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
            else:
                # uridecodebin autoplugs NVIDIA decode on Jetson when its plugins are
                # available. videorate then applies the same duration-derived fixed rate.
                from fractions import Fraction

                rate = Fraction(fps).limit_denominator(100_000)
                converter = self._gstreamer_converter
                if converter == "auto":
                    inspect = shutil.which("gst-inspect-1.0")
                    if inspect is not None:
                        probe = await asyncio.create_subprocess_exec(
                            inspect,
                            "nvvidconv",
                            stdout=asyncio.subprocess.DEVNULL,
                            stderr=asyncio.subprocess.DEVNULL,
                        )
                        converter = (
                            "nvvidconv" if await probe.wait() == 0 else "videoconvert"
                        )
                    else:
                        converter = "videoconvert"
                process = await asyncio.create_subprocess_exec(
                    self._gstreamer_path,
                    "-q",
                    "uridecodebin",
                    f"uri={source.resolve().as_uri()}",
                    "!",
                    converter,
                    "!",
                    f"video/x-raw,width={request.frame_width}",
                    "!",
                    "videorate",
                    "!",
                    f"video/x-raw,framerate={rate.numerator}/{rate.denominator}",
                    "!",
                    "jpegenc",
                    "!",
                    "multifilesink",
                    f"location={frame_pattern}",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
            _, stderr = await process.communicate()
            frames = sorted(Path(directory).glob("frame-*.jpg"))
            if process.returncode != 0 or len(frames) < request.sample_count:
                detail = stderr.decode("utf-8", errors="replace").strip()
                raise ExecutionFailedError(
                    f"{sampler} sample_frames produced {len(frames)}/"
                    f"{request.sample_count} frames: {detail}"
                )
            contact_sheet = await asyncio.to_thread(
                _compose_contact_sheet,
                frames[: request.sample_count],
                request.columns,
                request.duration_s,
            )
            output = await self._artifact_store.put_bytes(
                request.output_artifact_id,
                contact_sheet,
                "image/jpeg",
            )
        return ExecutionResult(
            request_id=request.request_id,
            executor_id=f"{self._worker_id}:sample_frames",
            output_artifacts=[output],
            queue_ms=0.0,
            service_ms=(perf_counter() - started_at) * 1000,
            metadata={
                "semantic_operator": "sample_frames",
                "sampler": sampler,
                "sample_count": request.sample_count,
                "columns": request.columns,
                "layout": "chronological_contact_sheet",
            },
        )

    async def _pull_with_client(
        self,
        client: httpx.AsyncClient,
        url: str,
        request: ArtifactPullRequest,
    ) -> ArtifactRef:
        async with client.stream("GET", url) as response:
            if not response.is_success:
                await response.aread()
                raise ArtifactTransferError(
                    f"source Worker returned HTTP {response.status_code}: {response.text}"
                )
            chunks = response.aiter_bytes()
            if request.bandwidth_mbps is not None:
                chunks = self._throttled_chunks(chunks, request.bandwidth_mbps)
            return await self._artifact_store.put_stream(
                request.artifact.id,
                chunks,
                request.artifact.artifact_type,
            )

    @staticmethod
    async def _throttled_chunks(
        chunks: AsyncIterator[bytes], bandwidth_mbps: float
    ) -> AsyncIterator[bytes]:
        bytes_per_second = bandwidth_mbps * 1_000_000 / 8.0
        async for chunk in chunks:
            yield chunk
            await asyncio.sleep(len(chunk) / bytes_per_second)

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
