"""Asynchronous JSONL trace recorder."""

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

_RESERVED_FIELDS: Final = frozenset({"event_type", "run_id", "timestamp"})


class TraceRecorder:
    """Append structured run events safely across concurrent async calls."""

    def __init__(self, runs_root: Path, run_id: str) -> None:
        if not run_id.strip():
            raise ValueError("run_id must not be empty")
        if "/" in run_id or "\\" in run_id or run_id in {".", ".."}:
            raise ValueError("run_id must be one safe path segment")

        self._run_id = run_id
        self._run_directory = runs_root.resolve() / run_id
        self._run_directory.mkdir(parents=True, exist_ok=True)
        self._path = self._run_directory / "trace.jsonl"
        self._lock = asyncio.Lock()

    @property
    def run_id(self) -> str:
        """Return the run identifier added to every event."""
        return self._run_id

    @property
    def path(self) -> Path:
        """Return the JSONL destination path."""
        return self._path

    async def record(self, event_type: str, **fields: object) -> None:
        """Serialize and append one event without blocking the event loop on file I/O."""
        if not event_type.strip():
            raise ValueError("event_type must not be empty")
        reserved = _RESERVED_FIELDS.intersection(fields)
        if reserved:
            raise ValueError(f"trace fields may not override reserved fields: {sorted(reserved)}")

        event: dict[str, object] = {
            "event_type": event_type,
            "run_id": self._run_id,
            "timestamp": datetime.now(UTC).isoformat(),
            **fields,
        }
        line = json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n"
        async with self._lock:
            await asyncio.to_thread(self._append_line, line)

    def _append_line(self, line: str) -> None:
        with self._path.open("a", encoding="utf-8", newline="") as trace_file:
            trace_file.write(line)
