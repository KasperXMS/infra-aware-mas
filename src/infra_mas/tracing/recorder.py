"""Asynchronous JSONL trace recorder."""

import asyncio
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Final
from uuid import uuid4

import yaml
from pydantic import BaseModel

from infra_mas.core.trace import TraceEvent

_RESERVED_FIELDS: Final = frozenset(
    {"event_type", "run_id", "timestamp", "action_id", "parent_action_id"}
)
_ACTION_EVENT_TYPES: Final = frozenset(
    {
        "planner.delegate",
        "planner.finish",
        "planner.inspect_artifact",
        "planner.llm.start",
        "planner.llm.end",
        "execution.request",
        "executor.selected",
        "artifact.transfer.start",
        "artifact.transfer.end",
        "artifact.inspect.start",
        "artifact.inspect.end",
        "worker.execution.start",
        "worker.execution.end",
        "artifact.created",
    }
)


class TraceRecorder:
    """Append structured run events safely across concurrent async calls."""

    def __init__(self, runs_root: Path, run_id: str) -> None:
        if not run_id.strip():
            raise ValueError("run_id must not be empty")
        if "/" in run_id or "\\" in run_id or ":" in run_id or run_id in {".", ".."}:
            raise ValueError("run_id must be one safe path segment")

        self._run_id = run_id
        self._run_directory = runs_root.resolve() / run_id
        self._run_directory.mkdir(parents=True, exist_ok=True)
        self._path = self._run_directory / "trace.jsonl"
        self._config_path = self._run_directory / "config.yaml"
        self._result_path = self._run_directory / "result.json"
        self._lock = asyncio.Lock()

    @property
    def run_id(self) -> str:
        """Return the run identifier added to every event."""
        return self._run_id

    @property
    def path(self) -> Path:
        """Return the JSONL destination path."""
        return self._path

    @property
    def config_path(self) -> Path:
        """Return the run configuration destination."""
        return self._config_path

    @property
    def result_path(self) -> Path:
        """Return the final run result destination."""
        return self._result_path

    async def start(self, config: dict[str, object] | None = None) -> None:
        """Persist run configuration and append the run start event."""
        await self.save_config(config or {})
        await self.record("run.start")

    async def end(self, result: BaseModel | dict[str, object] | None = None) -> None:
        """Persist the final result and append the run end event."""
        await self.save_result(result or {})
        await self.record("run.end")

    async def save_config(self, config: dict[str, object]) -> None:
        """Atomically persist the effective run configuration as YAML."""
        content = yaml.safe_dump(config, sort_keys=False, allow_unicode=True)
        await self._write_atomic(self._config_path, content)

    async def save_result(self, result: BaseModel | dict[str, object]) -> None:
        """Atomically persist the final run result as formatted JSON."""
        payload: object = (
            result.model_dump(mode="json") if isinstance(result, BaseModel) else result
        )
        content = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        await self._write_atomic(self._result_path, content)

    async def record(
        self,
        event_type: str,
        *,
        action_id: str | None = None,
        parent_action_id: str | None = None,
        **fields: object,
    ) -> None:
        """Serialize and append one event without blocking the event loop on file I/O."""
        if not event_type.strip():
            raise ValueError("event_type must not be empty")
        if event_type in _ACTION_EVENT_TYPES and action_id is None:
            raise ValueError(f"event {event_type!r} requires an action_id")
        reserved = _RESERVED_FIELDS.intersection(fields)
        if reserved:
            raise ValueError(f"trace fields may not override reserved fields: {sorted(reserved)}")

        async with self._lock:
            event = TraceEvent(
                event_type=event_type,
                run_id=self._run_id,
                timestamp=datetime.now(UTC),
                action_id=action_id,
                parent_action_id=parent_action_id,
                **fields,
            )
            payload = event.model_dump(mode="json", exclude_none=True)
            if action_id is not None:
                payload["parent_action_id"] = parent_action_id
            line = (
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )
            await asyncio.to_thread(self._append_line, line)

    def _append_line(self, line: str) -> None:
        with self._path.open("a", encoding="utf-8", newline="") as trace_file:
            trace_file.write(line)

    async def _write_atomic(self, destination: Path, content: str) -> None:
        temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
        try:
            await asyncio.to_thread(temporary.write_text, content, encoding="utf-8", newline="")
            await asyncio.to_thread(os.replace, temporary, destination)
        finally:
            if await asyncio.to_thread(temporary.is_file):
                await asyncio.to_thread(temporary.unlink)
