"""Worker execution service."""

import asyncio
import io
import json
import math
import re
import shutil
import tempfile
from collections import Counter
from collections.abc import AsyncIterator, Callable, Iterable
from dataclasses import dataclass
from functools import cmp_to_key
from pathlib import Path
from time import perf_counter
from typing import Any, Literal, Protocol, cast, runtime_checkable
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
    MakeContactSheetRequest,
    RecordAggregation,
    RecordDerivation,
    RecordPredicate,
    RecordSort,
    SampleFramesRequest,
    SelectFieldsRequest,
    TopKRecordsRequest,
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


def _bm25_tokens(text: str) -> list[str]:
    """Use the fixed lowercase ASCII-alphanumeric tokenizer from candidate retrieval."""
    return re.findall(r"[a-z0-9]+", text.lower())


_MISSING = object()
_JSON_RECORD_MEDIA_TYPES = frozenset(
    {
        "application/json",
        "application/jsonl",
        "application/x-jsonlines",
        "application/x-ndjson",
        "text/json",
        "text/jsonl",
        "text/x-jsonlines",
    }
)


def _load_json_records(inputs: list[tuple[ArtifactRef, Path]]) -> list[dict[str, Any]]:
    """Load JSON arrays, record envelopes, objects, or JSON Lines deterministically."""
    records: list[dict[str, Any]] = []
    for artifact, path in inputs:
        media_type = artifact.artifact_type.partition(";")[0].strip().lower()
        if media_type not in _JSON_RECORD_MEDIA_TYPES and not media_type.endswith("+json"):
            raise ExecutionFailedError(
                "structured record operators require JSON or JSON Lines inputs, "
                f"got {artifact.artifact_type!r}"
            )
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as error:
            raise ExecutionFailedError(
                f"artifact {artifact.id!r} is not valid UTF-8"
            ) from error
        try:
            payload: object = json.loads(content)
        except json.JSONDecodeError:
            try:
                payload = [
                    json.loads(line)
                    for line in content.splitlines()
                    if line.strip()
                ]
            except json.JSONDecodeError as error:
                raise ExecutionFailedError(
                    f"artifact {artifact.id!r} is neither valid JSON nor JSON Lines"
                ) from error
        if isinstance(payload, dict):
            payload_mapping = cast(dict[str, object], payload)
            if "records" in payload_mapping:
                payload = payload_mapping["records"]
            else:
                payload = [payload_mapping]
        if not isinstance(payload, list):
            raise ExecutionFailedError(
                f"artifact {artifact.id!r} must contain a JSON record or record array"
            )
        for index, record in enumerate(cast(list[Any], payload)):
            if not isinstance(record, dict):
                raise ExecutionFailedError(
                    f"artifact {artifact.id!r} record {index} must be a JSON object"
                )
            record_mapping = cast(dict[object, object], record)
            if not all(isinstance(key, str) for key in record_mapping):
                raise ExecutionFailedError(
                    f"artifact {artifact.id!r} record {index} must use string keys"
                )
            records.append(cast(dict[str, Any], record_mapping))
    return records


def _field_value(record: dict[str, Any], field: str) -> Any:
    """Resolve exact keys first, then dotted paths through nested objects."""
    if field in record:
        return record[field]
    value: Any = record
    for component in field.split("."):
        if not isinstance(value, dict) or component not in value:
            return _MISSING
        value = cast(dict[str, Any], value)[component]
    return value


def _matches_predicate(record: dict[str, Any], predicate: RecordPredicate) -> bool:
    actual = _field_value(record, predicate.field)
    operator = predicate.operator
    expected = predicate.value
    if operator == "is_null":
        return actual is _MISSING or actual is None
    if operator == "not_null":
        return actual is not _MISSING and actual is not None
    if actual is _MISSING:
        return False
    if operator == "eq":
        return bool(actual == expected)
    if operator == "ne":
        return bool(actual != expected)
    if operator in {"in", "not_in"}:
        assert isinstance(expected, list)
        contained = actual in expected
        return contained if operator == "in" else not contained
    if operator in {"contains", "not_contains"}:
        try:
            contained = expected in actual
        except (TypeError, AttributeError):
            contained = False
        return contained if operator == "contains" else not contained
    try:
        if operator == "lt":
            return bool(actual < expected)
        if operator == "lte":
            return bool(actual <= expected)
        if operator == "gt":
            return bool(actual > expected)
        if operator == "gte":
            return bool(actual >= expected)
    except TypeError:
        return False
    raise AssertionError(f"unhandled predicate operator: {operator}")


def _aggregate_values(
    records: list[dict[str, Any]], aggregation: RecordAggregation
) -> int | float | str | bool | None:
    if aggregation.operation == "count" and aggregation.field is None:
        return len(records)
    assert aggregation.field is not None
    values = [
        value
        for record in records
        if (value := _field_value(record, aggregation.field)) is not _MISSING
        and value is not None
    ]
    if aggregation.operation == "count":
        return len(values)
    if aggregation.operation == "count_distinct":
        return len(
            {
                json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                for value in values
            }
        )
    if not values:
        return None
    if aggregation.operation in {"sum", "mean"}:
        if any(not isinstance(value, (int, float)) or isinstance(value, bool) for value in values):
            raise ExecutionFailedError(
                f"aggregate_records {aggregation.operation} requires numeric values in "
                f"field {aggregation.field!r}"
            )
        total = sum(values)
        return total if aggregation.operation == "sum" else total / len(values)
    try:
        if aggregation.operation == "min":
            return min(values)
        if aggregation.operation == "max":
            return max(values)
    except TypeError as error:
        raise ExecutionFailedError(
            f"aggregate_records {aggregation.operation} found incomparable values in "
            f"field {aggregation.field!r}"
        ) from error
    raise AssertionError(f"unhandled aggregation operation: {aggregation.operation}")


def _filter_json_records(
    records: list[dict[str, Any]],
    predicates: list[RecordPredicate],
    match: Literal["all", "any"],
) -> list[dict[str, Any]]:
    combiner = all if match == "all" else any
    return [
        record
        for record in records
        if combiner(_matches_predicate(record, predicate) for predicate in predicates)
    ]


def _select_json_fields(
    records: list[dict[str, Any]], fields: list[str]
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for record in records:
        projected: dict[str, Any] = {}
        for field in fields:
            value = _field_value(record, field)
            if value is not _MISSING:
                projected[field] = value
        selected.append(projected)
    return selected


def _aggregate_json_records(
    records: list[dict[str, Any]],
    aggregations: list[RecordAggregation],
    group_by: list[str],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], tuple[list[Any], list[dict[str, Any]]]] = {}
    for record in records:
        group_values = [
            None if (value := _field_value(record, field)) is _MISSING else value
            for field in group_by
        ]
        key = tuple(
            json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            for value in group_values
        )
        if key not in grouped:
            grouped[key] = (group_values, [])
        grouped[key][1].append(record)
    if not group_by and not grouped:
        grouped[()] = ([], [])

    output: list[dict[str, Any]] = []
    for key in sorted(grouped):
        group_values, group_records = grouped[key]
        item = dict(zip(group_by, group_values, strict=True))
        item.update(
            {
                aggregation.output_field: _aggregate_values(
                    group_records, aggregation
                )
                for aggregation in aggregations
            }
        )
        output.append(item)
    return output


def _derive_json_fields(
    records: list[dict[str, Any]], derivations: list[RecordDerivation]
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        derived = dict(record)
        for derivation in derivations:
            left = _field_value(derived, derivation.left_field)
            right = _field_value(derived, derivation.right_field)
            if (
                left is _MISSING
                or right is _MISSING
                or not isinstance(left, (int, float))
                or isinstance(left, bool)
                or not isinstance(right, (int, float))
                or isinstance(right, bool)
            ):
                raise ExecutionFailedError(
                    f"derive_fields record {index} requires numeric fields "
                    f"{derivation.left_field!r} and {derivation.right_field!r}"
                )
            if derivation.operation == "add":
                value = left + right
            elif derivation.operation == "subtract":
                value = left - right
            elif derivation.operation == "multiply":
                value = left * right
            elif derivation.operation == "divide":
                if right == 0:
                    raise ExecutionFailedError(
                        f"derive_fields record {index} divides by zero in field "
                        f"{derivation.right_field!r}"
                    )
                value = left / right
            else:
                raise AssertionError(
                    f"unhandled derivation operation: {derivation.operation}"
                )
            derived[derivation.output_field] = value
        output.append(derived)
    return output


def _compare_record_values(left: Any, right: Any, sort: RecordSort) -> int:
    left_null = left is _MISSING or left is None
    right_null = right is _MISSING or right is None
    if left_null or right_null:
        if left_null and right_null:
            return 0
        null_first = sort.nulls == "first"
        return -1 if left_null == null_first else 1
    if (
        isinstance(left, (int, float))
        and not isinstance(left, bool)
        and isinstance(right, (int, float))
        and not isinstance(right, bool)
    ):
        comparison = (left > right) - (left < right)
    elif isinstance(left, str) and isinstance(right, str):
        comparison = (left > right) - (left < right)
    elif isinstance(left, bool) and isinstance(right, bool):
        comparison = (left > right) - (left < right)
    else:
        left_encoded = json.dumps(
            left, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        right_encoded = json.dumps(
            right, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        comparison = (left_encoded > right_encoded) - (left_encoded < right_encoded)
    return comparison if sort.direction == "ascending" else -comparison


def _top_k_json_records(
    records: list[dict[str, Any]], order_by: list[RecordSort], limit: int
) -> list[dict[str, Any]]:
    def compare(left: dict[str, Any], right: dict[str, Any]) -> int:
        for sort in order_by:
            result = _compare_record_values(
                _field_value(left, sort.field), _field_value(right, sort.field), sort
            )
            if result:
                return result
        left_encoded = json.dumps(
            left, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        right_encoded = json.dumps(
            right, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        return (left_encoded > right_encoded) - (left_encoded < right_encoded)

    return sorted(records, key=cmp_to_key(compare))[:limit]


def _compose_contact_sheet(
    frame_paths: list[Path],
    columns: int,
    duration_s: float | None,
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
            label = f"sample {index + 1:02d}"
            if duration_s is not None:
                timestamp_s = duration_s * (index + 0.5) / len(loaded)
                label += f" @ {timestamp_s:.0f} sec"
            draw.text(
                (x + 6, y + tile_height + 4),
                label,
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
        local_source_roots: Iterable[Path] = (),
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
        self._local_source_roots = tuple(path.resolve() for path in local_source_roots)

    @property
    def artifact_store(self) -> ArtifactStore:
        """Return the worker-local artifact store for HTTP data-plane endpoints."""
        return self._artifact_store

    def status(self) -> WorkerStatus:
        """Return static worker and executor identity information."""
        operators = [
            "make_contact_sheet",
            "aggregate_artifacts",
            "bm25_retrieve",
            "filter_records",
            "select_fields",
            "aggregate_records",
            "derive_fields",
            "top_k_records",
        ]
        ffmpeg_available = shutil.which(self._ffmpeg_path) is not None
        gstreamer_available = shutil.which(self._gstreamer_path) is not None
        if ffmpeg_available:
            operators.append("extract_clip")
        has_duration_probe = shutil.which("ffprobe") is not None or (
            shutil.which("gst-discoverer-1.0") is not None
        )
        sampler_available = (
            self._frame_sampler == "ffmpeg" and ffmpeg_available
        ) or (
            self._frame_sampler == "gstreamer" and gstreamer_available
        ) or (
            self._frame_sampler == "auto"
            and (ffmpeg_available or gstreamer_available)
        )
        if sampler_available and has_duration_probe:
            operators.append("sample_frames")
        return WorkerStatus(
            worker_id=self._worker_id,
            executors=[executor.id for executor in self._executors.values()],
            operators=operators,
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
        duration_s = request.duration_s or await self._probe_video_duration(source)
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
            fps = request.sample_count / duration_s
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
                duration_s,
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
                "duration_s": duration_s,
                "layout": "chronological_contact_sheet",
            },
        )

    async def _probe_video_duration(self, source: Path) -> float:
        """Probe video duration locally without sending media to the coordinator."""
        ffprobe = shutil.which("ffprobe")
        if ffprobe is not None:
            process = await asyncio.create_subprocess_exec(
                ffprobe,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(source),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await process.communicate()
            if process.returncode == 0:
                try:
                    duration = float(stdout.decode().strip())
                except ValueError:
                    duration = 0.0
                if duration > 0:
                    return duration

        discoverer = shutil.which("gst-discoverer-1.0")
        if discoverer is not None:
            process = await asyncio.create_subprocess_exec(
                discoverer,
                source.resolve().as_uri(),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await process.communicate()
            if process.returncode == 0:
                match = re.search(
                    rb"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", stdout
                )
                if match is not None:
                    hours, minutes, seconds = match.groups()
                    duration = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
                    if duration > 0:
                        return duration
        raise ExecutionFailedError(
            "sample_frames could not probe duration; pass duration_s or install "
            "ffprobe/gst-discoverer-1.0"
        )

    async def make_contact_sheet(self, request: MakeContactSheetRequest) -> ExecutionResult:
        """Compose already-local chronological images into one JPEG contact sheet."""
        if any(not item.artifact_type.startswith("image/") for item in request.input_artifacts):
            raise ExecutionFailedError("make_contact_sheet requires only image artifacts")
        started_at = perf_counter()
        paths = [await self._artifact_store.get_path(item.id) for item in request.input_artifacts]
        encoded = await asyncio.to_thread(
            _compose_contact_sheet,
            paths,
            request.columns,
            request.duration_s,
        )
        output = await self._artifact_store.put_bytes(
            request.output_artifact_id,
            encoded,
            "image/jpeg",
        )
        return ExecutionResult(
            request_id=request.request_id,
            executor_id=f"{self._worker_id}:make_contact_sheet",
            output_artifacts=[output],
            queue_ms=0.0,
            service_ms=(perf_counter() - started_at) * 1000,
            metadata={
                "semantic_operator": "make_contact_sheet",
                "input_count": len(paths),
                "columns": request.columns,
            },
        )

    async def extract_clip(self, request: ExtractClipRequest) -> ExecutionResult:
        """Extract one fixed video interval locally with ffmpeg stream copying."""
        if not request.input_artifact.artifact_type.startswith("video/"):
            raise ExecutionFailedError("extract_clip requires a video artifact")
        source = await self._artifact_store.get_path(request.input_artifact.id)
        if shutil.which(self._ffmpeg_path) is None:
            raise ExecutionFailedError("extract_clip requires ffmpeg on the selected Worker")
        started_at = perf_counter()
        with tempfile.TemporaryDirectory(prefix="infra-mas-clip-") as directory:
            target = Path(directory) / "clip.mp4"
            process = await asyncio.create_subprocess_exec(
                self._ffmpeg_path,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-ss",
                str(request.start_s),
                "-i",
                str(source),
                "-t",
                str(request.duration_s),
                "-map",
                "0",
                "-c",
                "copy",
                str(target),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await process.communicate()
            if process.returncode != 0 or not target.is_file():
                detail = stderr.decode("utf-8", errors="replace").strip()
                raise ExecutionFailedError(f"ffmpeg extract_clip failed: {detail}")
            output = await self._artifact_store.import_file(
                request.output_artifact_id,
                target,
                "video/mp4",
            )
        return ExecutionResult(
            request_id=request.request_id,
            executor_id=f"{self._worker_id}:extract_clip",
            output_artifacts=[output],
            queue_ms=0.0,
            service_ms=(perf_counter() - started_at) * 1000,
            metadata={
                "semantic_operator": "extract_clip",
                "start_s": request.start_s,
                "end_s": request.end_s,
            },
        )

    async def aggregate_artifacts(
        self, request: AggregateArtifactsRequest
    ) -> ExecutionResult:
        """Aggregate UTF-8 evidence into a stable JSON envelope locally."""
        started_at = perf_counter()
        records: list[dict[str, str]] = []
        for artifact in request.input_artifacts:
            media_type = artifact.artifact_type.partition(";")[0].lower()
            if not (
                media_type.startswith("text/")
                or media_type.endswith("+json")
                or media_type in {"application/json", "application/xml", "application/yaml"}
            ):
                raise ExecutionFailedError(
                    f"aggregate_artifacts requires textual inputs, got {artifact.artifact_type!r}"
                )
            path = await self._artifact_store.get_path(artifact.id)
            try:
                content = await asyncio.to_thread(path.read_text, encoding="utf-8")
            except UnicodeDecodeError as error:
                raise ExecutionFailedError(
                    f"artifact {artifact.id!r} is not valid UTF-8"
                ) from error
            records.append({"artifact_id": artifact.id, "content": content})
        payload = json.dumps({"artifacts": records}, ensure_ascii=False, separators=(",", ":"))
        output = await self._artifact_store.put_text(
            request.output_artifact_id,
            payload,
            "application/json",
        )
        return ExecutionResult(
            request_id=request.request_id,
            executor_id=f"{self._worker_id}:aggregate_artifacts",
            output_artifacts=[output],
            queue_ms=0.0,
            service_ms=(perf_counter() - started_at) * 1000,
            metadata={
                "semantic_operator": "aggregate_artifacts",
                "input_count": len(records),
            },
        )

    async def bm25_retrieve(
        self, request: BM25RetrieveRequest
    ) -> ExecutionResult:
        """Rank independent UTF-8 documents with deterministic, dataset-neutral BM25."""
        started_at = perf_counter()
        documents: list[tuple[ArtifactRef, str, list[str]]] = []
        for artifact in request.input_artifacts:
            media_type = artifact.artifact_type.partition(";")[0].lower()
            if not (
                media_type.startswith("text/")
                or media_type.endswith("+json")
                or media_type
                in {"application/json", "application/xml", "application/yaml"}
            ):
                raise ExecutionFailedError(
                    f"bm25_retrieve requires textual inputs, got {artifact.artifact_type!r}"
                )
            path = await self._artifact_store.get_path(artifact.id)
            try:
                content = await asyncio.to_thread(path.read_text, encoding="utf-8")
            except UnicodeDecodeError as error:
                raise ExecutionFailedError(
                    f"artifact {artifact.id!r} is not valid UTF-8"
                ) from error
            documents.append((artifact, content, _bm25_tokens(content)))

        query_terms = _bm25_tokens(request.query)
        if not query_terms:
            raise ExecutionFailedError("bm25_retrieve query has no searchable terms")
        document_count = len(documents)
        average_length = sum(len(tokens) for _, _, tokens in documents) / document_count
        document_frequency = {
            term: sum(term in set(tokens) for _, _, tokens in documents)
            for term in set(query_terms)
        }
        ranked: list[tuple[float, str, str, int]] = []
        for artifact, content, tokens in documents:
            frequencies = Counter(tokens)
            score = 0.0
            for term in set(query_terms):
                frequency = frequencies[term]
                if frequency == 0:
                    continue
                frequency_in_corpus = document_frequency[term]
                inverse_document_frequency = math.log(
                    1.0
                    + (document_count - frequency_in_corpus + 0.5)
                    / (frequency_in_corpus + 0.5)
                )
                length_normalizer = 1.0 - request.b
                if average_length > 0:
                    length_normalizer += request.b * len(tokens) / average_length
                score += inverse_document_frequency * (
                    frequency * (request.k1 + 1.0)
                    / (frequency + request.k1 * length_normalizer)
                )
            ranked.append((score, artifact.id, content, len(tokens)))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        selected = ranked[: min(request.top_k, len(ranked))]
        payload = json.dumps(
            {
                "schema_version": "generic-bm25-evidence-v1",
                "algorithm": "bm25",
                "parameters": {"top_k": request.top_k, "k1": request.k1, "b": request.b},
                "query": request.query,
                "candidate_document_count": document_count,
                "documents": [
                    {
                        "artifact_id": artifact_id,
                        "score": score,
                        "content": content,
                    }
                    for score, artifact_id, content, _ in selected
                ],
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        output = await self._artifact_store.put_text(
            request.output_artifact_id,
            payload,
            "application/json",
        )
        return ExecutionResult(
            request_id=request.request_id,
            executor_id=f"{self._worker_id}:bm25_retrieve",
            output_artifacts=[output],
            queue_ms=0.0,
            service_ms=(perf_counter() - started_at) * 1000,
            metadata={
                "semantic_operator": "bm25_retrieve",
                "algorithm": "bm25",
                "top_k": request.top_k,
                "k1": request.k1,
                "b": request.b,
                "candidate_document_count": document_count,
                "candidate_tokens": sum(len(tokens) for _, _, tokens in documents),
                "retrieved_document_count": len(selected),
                "retrieved_tokens": sum(token_count for *_, token_count in selected),
                "retrieved_artifact_ids": [artifact_id for _, artifact_id, _, _ in selected],
            },
        )

    async def filter_records(
        self, request: FilterRecordsRequest
    ) -> ExecutionResult:
        """Apply deterministic predicates to generic JSON records locally."""
        started_at = perf_counter()
        inputs = [
            (artifact, await self._artifact_store.get_path(artifact.id))
            for artifact in request.input_artifacts
        ]
        records = await asyncio.to_thread(_load_json_records, inputs)
        filtered = await asyncio.to_thread(
            _filter_json_records, records, request.predicates, request.match
        )
        payload = json.dumps(
            {
                "schema_version": "generic-record-set-v1",
                "operator": "filter_records",
                "records": filtered,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        output_tokens = len(_bm25_tokens(payload))
        output = await self._artifact_store.put_text(
            request.output_artifact_id, payload, "application/json"
        )
        return ExecutionResult(
            request_id=request.request_id,
            executor_id=f"{self._worker_id}:filter_records",
            output_artifacts=[output],
            queue_ms=0.0,
            service_ms=(perf_counter() - started_at) * 1000,
            metadata={
                "semantic_operator": "filter_records",
                "input_record_count": len(records),
                "output_record_count": len(filtered),
                "predicate_count": len(request.predicates),
                "match": request.match,
                "output_tokens": output_tokens,
                "output_tokens_lexical": output_tokens,
            },
        )

    async def select_fields(
        self, request: SelectFieldsRequest
    ) -> ExecutionResult:
        """Project generic JSON records onto explicit field paths locally."""
        started_at = perf_counter()
        inputs = [
            (artifact, await self._artifact_store.get_path(artifact.id))
            for artifact in request.input_artifacts
        ]
        records = await asyncio.to_thread(_load_json_records, inputs)
        selected = await asyncio.to_thread(
            _select_json_fields, records, request.fields
        )
        payload = json.dumps(
            {
                "schema_version": "generic-record-set-v1",
                "operator": "select_fields",
                "records": selected,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        output_tokens = len(_bm25_tokens(payload))
        output = await self._artifact_store.put_text(
            request.output_artifact_id, payload, "application/json"
        )
        return ExecutionResult(
            request_id=request.request_id,
            executor_id=f"{self._worker_id}:select_fields",
            output_artifacts=[output],
            queue_ms=0.0,
            service_ms=(perf_counter() - started_at) * 1000,
            metadata={
                "semantic_operator": "select_fields",
                "input_record_count": len(records),
                "output_record_count": len(selected),
                "selected_fields": request.fields,
                "output_tokens": output_tokens,
                "output_tokens_lexical": output_tokens,
            },
        )

    async def aggregate_records(
        self, request: AggregateRecordsRequest
    ) -> ExecutionResult:
        """Group and aggregate generic JSON records locally."""
        started_at = perf_counter()
        inputs = [
            (artifact, await self._artifact_store.get_path(artifact.id))
            for artifact in request.input_artifacts
        ]
        records = await asyncio.to_thread(_load_json_records, inputs)
        groups = await asyncio.to_thread(
            _aggregate_json_records,
            records,
            request.aggregations,
            request.group_by,
        )
        payload = json.dumps(
            {
                "schema_version": "generic-record-aggregation-v1",
                "operator": "aggregate_records",
                "group_by": request.group_by,
                "aggregations": [
                    aggregation.model_dump(mode="json")
                    for aggregation in request.aggregations
                ],
                "groups": groups,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        output_tokens = len(_bm25_tokens(payload))
        output = await self._artifact_store.put_text(
            request.output_artifact_id, payload, "application/json"
        )
        return ExecutionResult(
            request_id=request.request_id,
            executor_id=f"{self._worker_id}:aggregate_records",
            output_artifacts=[output],
            queue_ms=0.0,
            service_ms=(perf_counter() - started_at) * 1000,
            metadata={
                "semantic_operator": "aggregate_records",
                "input_record_count": len(records),
                "output_group_count": len(groups),
                "group_by": request.group_by,
                "aggregation_count": len(request.aggregations),
                "output_tokens": output_tokens,
                "output_tokens_lexical": output_tokens,
            },
        )

    async def derive_fields(
        self, request: DeriveFieldsRequest
    ) -> ExecutionResult:
        """Add safe declarative arithmetic fields to generic JSON records locally."""
        started_at = perf_counter()
        inputs = [
            (artifact, await self._artifact_store.get_path(artifact.id))
            for artifact in request.input_artifacts
        ]
        records = await asyncio.to_thread(_load_json_records, inputs)
        derived = await asyncio.to_thread(
            _derive_json_fields, records, request.derivations
        )
        payload = json.dumps(
            {
                "schema_version": "generic-record-set-v1",
                "operator": "derive_fields",
                "records": derived,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        output_tokens = len(_bm25_tokens(payload))
        output = await self._artifact_store.put_text(
            request.output_artifact_id, payload, "application/json"
        )
        return ExecutionResult(
            request_id=request.request_id,
            executor_id=f"{self._worker_id}:derive_fields",
            output_artifacts=[output],
            queue_ms=0.0,
            service_ms=(perf_counter() - started_at) * 1000,
            metadata={
                "semantic_operator": "derive_fields",
                "input_record_count": len(records),
                "output_record_count": len(derived),
                "derived_fields": [item.output_field for item in request.derivations],
                "output_tokens": output_tokens,
                "output_tokens_lexical": output_tokens,
            },
        )

    async def top_k_records(
        self, request: TopKRecordsRequest
    ) -> ExecutionResult:
        """Order generic JSON records deterministically and retain a prefix locally."""
        started_at = perf_counter()
        inputs = [
            (artifact, await self._artifact_store.get_path(artifact.id))
            for artifact in request.input_artifacts
        ]
        records = await asyncio.to_thread(_load_json_records, inputs)
        selected = await asyncio.to_thread(
            _top_k_json_records, records, request.order_by, request.limit
        )
        payload = json.dumps(
            {
                "schema_version": "generic-record-set-v1",
                "operator": "top_k_records",
                "records": selected,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        output_tokens = len(_bm25_tokens(payload))
        output = await self._artifact_store.put_text(
            request.output_artifact_id, payload, "application/json"
        )
        return ExecutionResult(
            request_id=request.request_id,
            executor_id=f"{self._worker_id}:top_k_records",
            output_artifacts=[output],
            queue_ms=0.0,
            service_ms=(perf_counter() - started_at) * 1000,
            metadata={
                "semantic_operator": "top_k_records",
                "input_record_count": len(records),
                "output_record_count": len(selected),
                "limit": request.limit,
                "order_by": [item.model_dump(mode="json") for item in request.order_by],
                "output_tokens": output_tokens,
                "output_tokens_lexical": output_tokens,
            },
        )

    async def bind_local_artifact(self, request: BindLocalArtifactRequest) -> ArtifactRef:
        """Import an allowlisted local file; no artifact bytes traverse HTTP."""
        source = Path(request.source_path).resolve()
        if not self._local_source_roots:
            raise ExecutionFailedError("Worker has no allowlisted local source roots")
        if not any(source.is_relative_to(root) for root in self._local_source_roots):
            raise ExecutionFailedError("local artifact source is outside allowlisted roots")
        output = await self._artifact_store.import_file(
            request.artifact_id,
            source,
            request.artifact_type,
        )
        if (
            request.expected_size_bytes is not None
            and output.size_bytes != request.expected_size_bytes
        ):
            await self._artifact_store.delete(output.id)
            raise ExecutionFailedError(
                f"local artifact size is {output.size_bytes}, expected "
                f"{request.expected_size_bytes}"
            )
        return output

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
