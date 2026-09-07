"""Planner dependencies and small-artifact catalog."""

from __future__ import annotations

import asyncio
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

from infra_mas.core.artifact import ArtifactRef
from infra_mas.core.errors import ArtifactNotFoundError
from infra_mas.execution.worker_client import WorkerClient
from infra_mas.runtime.agent_registry import AgentRegistry
from infra_mas.runtime.runtime import AgentRuntime
from infra_mas.tracing.recorder import TraceRecorder

_TEXT_ARTIFACT_TYPES = frozenset(
    {
        "application/json",
        "application/toml",
        "application/xml",
        "application/x-yaml",
        "application/yaml",
    }
)


class ArtifactCatalog:
    """Resolve artifact IDs and inspect only bounded textual artifacts."""

    def __init__(
        self,
        clients: Mapping[str, WorkerClient],
        temporary_directory: Path,
        *,
        max_inspect_bytes: int = 64 * 1024,
    ) -> None:
        if max_inspect_bytes <= 0:
            raise ValueError("max_inspect_bytes must be positive")
        self._clients = dict(clients)
        self._temporary_directory = temporary_directory.resolve()
        self._temporary_directory.mkdir(parents=True, exist_ok=True)
        self._max_inspect_bytes = max_inspect_bytes
        self._artifacts: dict[str, ArtifactRef] = {}

    def register(self, artifact: ArtifactRef) -> ArtifactRef:
        """Register an artifact or merge newly observed locations."""
        existing = self._artifacts.get(artifact.id)
        if existing is None:
            self._artifacts[artifact.id] = artifact
            return artifact
        if (
            existing.artifact_type != artifact.artifact_type
            or existing.size_bytes != artifact.size_bytes
        ):
            raise ValueError(f"conflicting metadata for artifact {artifact.id!r}")
        existing.locations = list(dict.fromkeys([*existing.locations, *artifact.locations]))
        return existing

    def register_many(self, artifacts: Iterable[ArtifactRef]) -> None:
        """Register several artifact references."""
        for artifact in artifacts:
            self.register(artifact)

    def get(self, artifact_id: str) -> ArtifactRef:
        """Resolve an artifact ID without loading its contents."""
        try:
            return self._artifacts[artifact_id]
        except KeyError as error:
            raise ArtifactNotFoundError(f"unknown artifact: {artifact_id}") from error

    def list(self) -> list[ArtifactRef]:
        """Return registered artifacts in insertion order."""
        return list(self._artifacts.values())

    def is_inspectable(self, artifact: ArtifactRef) -> bool:
        """Return whether an artifact is a bounded text or structured payload."""
        is_text = artifact.artifact_type.startswith("text/") or (
            artifact.artifact_type in _TEXT_ARTIFACT_TYPES
        )
        return is_text and artifact.size_bytes <= self._max_inspect_bytes

    async def inspect_text(self, artifact_id: str) -> str:
        """Download and decode one bounded textual artifact for planner inspection."""
        artifact = self.get(artifact_id)
        if not self.is_inspectable(artifact):
            raise ValueError(
                f"artifact {artifact_id!r} is binary or exceeds the inspection size limit"
            )
        source = next(
            (
                self._clients[location]
                for location in artifact.locations
                if location in self._clients
            ),
            None,
        )
        if source is None:
            raise ArtifactNotFoundError(f"no reachable location for artifact {artifact_id!r}")

        temporary = self._temporary_directory / f"inspect-{uuid4().hex}.artifact"
        try:
            downloaded = await source.download_artifact(artifact.id, temporary)
            if downloaded != artifact.size_bytes:
                raise ValueError(
                    f"artifact {artifact_id!r} size changed from {artifact.size_bytes} "
                    f"to {downloaded} bytes"
                )
            try:
                return await asyncio.to_thread(temporary.read_text, encoding="utf-8")
            except UnicodeDecodeError as error:
                raise ValueError(f"artifact {artifact_id!r} is not valid UTF-8 text") from error
        finally:
            if await asyncio.to_thread(temporary.is_file):
                await asyncio.to_thread(temporary.unlink)


@dataclass(slots=True)
class PlannerContext:
    """Provide local dependencies to Coordinator function tools."""

    runtime: AgentRuntime
    agent_registry: AgentRegistry
    artifact_catalog: ArtifactCatalog
    trace: TraceRecorder
    resource_provider: object | None = None
    resource_aware: bool = False
    _action_counter: int = field(default=0, init=False, repr=False)
    _action_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)

    @property
    def coordinator_action_id(self) -> str:
        """Return the stable parent action for planner decisions in this run."""
        return f"{self.trace.run_id}/coordinator"

    async def next_action_id(self, prefix: str) -> str:
        """Create a deterministic, concurrency-safe action ID within the run."""
        if not prefix.strip():
            raise ValueError("action prefix must not be empty")
        async with self._action_lock:
            self._action_counter += 1
            sequence = self._action_counter
        return f"{self.trace.run_id}/{prefix}-{sequence:04d}"
