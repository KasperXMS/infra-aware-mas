"""Unit tests for concurrent JSONL research tracing."""

import asyncio
import json
from pathlib import Path

import pytest
import yaml

from infra_mas.core.trace import TraceEvent
from infra_mas.tracing.recorder import TraceRecorder


async def test_recorder_persists_run_files_and_concurrent_events(tmp_path: Path) -> None:
    recorder = TraceRecorder(tmp_path / "runs", "run-001")
    await recorder.start({"scheduler": "fixed", "resource_aware": False})

    await asyncio.gather(
        *(
            recorder.record(
                "execution.request",
                action_id=f"action-{index}",
                agent="reasoner",
            )
            for index in range(20)
        )
    )
    await recorder.end({"answer": "done"})

    events = [
        TraceEvent.model_validate_json(line)
        for line in recorder.path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(events) == 22
    assert events[0].event_type == "run.start"
    assert events[-1].event_type == "run.end"
    assert {event.run_id for event in events} == {"run-001"}
    assert {event.action_id for event in events[1:-1]} == {f"action-{index}" for index in range(20)}
    raw_action_event: object = json.loads(recorder.path.read_text(encoding="utf-8").splitlines()[1])
    assert isinstance(raw_action_event, dict)
    assert "parent_action_id" in raw_action_event
    assert yaml.safe_load(recorder.config_path.read_text(encoding="utf-8")) == {
        "scheduler": "fixed",
        "resource_aware": False,
    }
    assert json.loads(recorder.result_path.read_text(encoding="utf-8")) == {"answer": "done"}


async def test_action_event_requires_action_id(tmp_path: Path) -> None:
    recorder = TraceRecorder(tmp_path / "runs", "run-001")

    with pytest.raises(ValueError, match="requires an action_id"):
        await recorder.record("worker.execution.start")


async def test_reserved_trace_fields_cannot_be_overridden(tmp_path: Path) -> None:
    recorder = TraceRecorder(tmp_path / "runs", "run-001")

    with pytest.raises(ValueError, match="reserved fields"):
        await recorder.record("custom.event", run_id="other-run")


def test_exclusive_recorder_rejects_non_empty_run_directory(tmp_path: Path) -> None:
    run_directory = tmp_path / "runs" / "run-001"
    run_directory.mkdir(parents=True)
    existing_trace = run_directory / "trace.jsonl"
    existing_trace.write_text("existing event\n", encoding="utf-8")

    with pytest.raises(ValueError, match="already exists and is non-empty"):
        TraceRecorder(tmp_path / "runs", "run-001", exclusive=True)

    assert existing_trace.read_text(encoding="utf-8") == "existing event\n"


def test_exclusive_recorder_atomically_reserves_empty_run_directory(tmp_path: Path) -> None:
    run_directory = tmp_path / "runs" / "run-001"
    run_directory.mkdir(parents=True)

    recorder = TraceRecorder(tmp_path / "runs", "run-001", exclusive=True)

    assert recorder.path.is_file()
    with pytest.raises(ValueError, match="already exists and is non-empty"):
        TraceRecorder(tmp_path / "runs", "run-001", exclusive=True)
