"""Worker process configuration and service construction."""

import os
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from infra_mas.worker.artifact_store import ArtifactStore
from infra_mas.worker.backends.mock import MockBackend
from infra_mas.worker.backends.openai_compatible import OpenAICompatibleBackend
from infra_mas.worker.service import WorkerExecutor, WorkerService

NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class OpenAICompatibleBackendConfig(BaseModel):
    """Configure one OpenAI-compatible worker-local model endpoint."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["openai_compatible"]
    base_url: NonEmptyString
    model: NonEmptyString
    api_key_env: NonEmptyString | None = None
    timeout_seconds: Annotated[float, Field(gt=0)] = 120.0
    max_tokens: Annotated[int, Field(gt=0)] = 1024
    temperature: Annotated[float, Field(ge=0, le=2)] = 0.0
    verify_model: bool = True
    video_transport: Literal["data_url", "file_url"] = "data_url"
    input_cost_per_million_tokens_usd: Annotated[float, Field(ge=0)] = 0.0
    output_cost_per_million_tokens_usd: Annotated[float, Field(ge=0)] = 0.0
    reasoning_effort: Literal["none", "low", "medium", "high"] | None = None
    output_format: Literal["text", "json_object"] = "text"


class MockBackendConfig(BaseModel):
    """Configure a deterministic backend for deployment smoke tests."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["mock"]
    latency_ms: Annotated[float, Field(ge=0)] = 0.0
    output: NonEmptyString = "mock result"
    fail: bool = False
    real_delay: bool = False


BackendConfig = Annotated[
    OpenAICompatibleBackendConfig | MockBackendConfig,
    Field(discriminator="type"),
]


class WorkerExecutorConfig(BaseModel):
    """Bind a local executor identity to a configured backend."""

    model_config = ConfigDict(extra="forbid")

    id: NonEmptyString
    capability: NonEmptyString
    backend: BackendConfig


class WorkerConfig(BaseModel):
    """Describe one independently deployable Worker process."""

    model_config = ConfigDict(extra="forbid")

    worker_id: NonEmptyString
    host: NonEmptyString = "0.0.0.0"
    port: Annotated[int, Field(ge=1, le=65535)] = 9001
    artifact_root: Path = Path("data/artifacts")
    ffmpeg_path: NonEmptyString = "ffmpeg"
    gstreamer_path: NonEmptyString = "gst-launch-1.0"
    frame_sampler: Literal["auto", "ffmpeg", "gstreamer"] = "auto"
    gstreamer_converter: Literal["auto", "nvvidconv", "videoconvert"] = "auto"
    local_source_roots: Annotated[list[Path], Field(default_factory=list)]
    executors: Annotated[list[WorkerExecutorConfig], Field(min_length=1)]

    @classmethod
    def from_yaml(cls, path: Path) -> "WorkerConfig":
        """Load and validate one Worker YAML file."""
        raw: object = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls.model_validate(raw)


def build_worker_service(config: WorkerConfig, config_directory: Path) -> WorkerService:
    """Construct a Worker service from validated configuration."""
    artifact_root = config.artifact_root
    if not artifact_root.is_absolute():
        artifact_root = config_directory / artifact_root
    local_source_roots = [
        path if path.is_absolute() else config_directory / path
        for path in config.local_source_roots
    ]

    executors: list[WorkerExecutor] = []
    for entry in config.executors:
        backend_config = entry.backend
        if isinstance(backend_config, OpenAICompatibleBackendConfig):
            api_key = "not-required"
            if backend_config.api_key_env is not None:
                api_key = os.getenv(backend_config.api_key_env, "")
                if not api_key:
                    raise ValueError(
                        f"required environment variable {backend_config.api_key_env!r} is not set"
                    )
            backend = OpenAICompatibleBackend(
                base_url=backend_config.base_url,
                model=backend_config.model,
                api_key=api_key,
                timeout_seconds=backend_config.timeout_seconds,
                max_tokens=backend_config.max_tokens,
                temperature=backend_config.temperature,
                verify_model=backend_config.verify_model,
                video_transport=backend_config.video_transport,
                input_cost_per_million_tokens_usd=(
                    backend_config.input_cost_per_million_tokens_usd
                ),
                output_cost_per_million_tokens_usd=(
                    backend_config.output_cost_per_million_tokens_usd
                ),
                reasoning_effort=backend_config.reasoning_effort,
                output_format=backend_config.output_format,
            )
        else:
            backend = MockBackend(
                latency_ms=backend_config.latency_ms,
                output=backend_config.output,
                fail=backend_config.fail,
                real_delay=backend_config.real_delay,
            )
        executors.append(WorkerExecutor(entry.id, entry.capability, backend))

    return WorkerService(
        worker_id=config.worker_id,
        artifact_store=ArtifactStore(artifact_root, config.worker_id),
        executors=executors,
        ffmpeg_path=config.ffmpeg_path,
        gstreamer_path=config.gstreamer_path,
        frame_sampler=config.frame_sampler,
        gstreamer_converter=config.gstreamer_converter,
        local_source_roots=local_source_roots,
    )
