from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from infra_mas.calibration_v0 import (
    CalibrationV0SweepConfig,
    _attempt_schedule,  # pyright: ignore[reportPrivateUsage]
    _completed_keys,  # pyright: ignore[reportPrivateUsage]
    _execution_rows,  # pyright: ignore[reportPrivateUsage]
    _realized_trace,  # pyright: ignore[reportPrivateUsage]
    _transfer_rows,  # pyright: ignore[reportPrivateUsage]
    build_video_task_interaction,
    execute_video_reference_workflow,
)
from infra_mas.core.artifact import ArtifactRef
from infra_mas.core.execution import ExecutionResult, InvocationSpec
from infra_mas.execution.executor_registry import ExecutorRegistry


def _chunks() -> list[ArtifactRef]:
    return [
        ArtifactRef(
            id=f"run/chunk-{index}.mp4",
            artifact_type="video/mp4",
            size_bytes=index * 1_000,
            locations=[worker],
        )
        for index, worker in enumerate(("a4", "a5", "a28"), start=1)
    ]


class FakeRuntime:
    def __init__(self) -> None:
        self.invocations: list[InvocationSpec] = []
        self.samples: list[tuple[ArtifactRef, str, float]] = []

    async def sample_frames_on_worker(
        self,
        artifact: ArtifactRef,
        target_worker_id: str,
        *,
        duration_s: float,
        sample_count: int = 20,
        columns: int = 5,
        frame_width: int = 448,
        parent_action_id: str | None = None,
    ) -> ExecutionResult:
        del sample_count, columns, frame_width, parent_action_id
        self.samples.append((artifact, target_worker_id, duration_s))
        index = len(self.samples)
        return ExecutionResult(
            request_id=f"sample-{index}",
            executor_id=f"{target_worker_id}:sample_frames",
            output_artifacts=[
                ArtifactRef(
                    id=f"run/frame-{index}.jpg",
                    artifact_type="image/jpeg",
                    size_bytes=200,
                    locations=[target_worker_id],
                )
            ],
            queue_ms=0,
            service_ms=5,
            metadata={"semantic_operator": "sample_frames"},
        )

    async def invoke(
        self,
        invocation: InvocationSpec,
        *,
        parent_action_id: str | None = None,
    ) -> ExecutionResult:
        del parent_action_id
        self.invocations.append(invocation)
        index = len(self.invocations)
        is_local = invocation.model_id == "local-video"
        output = ArtifactRef(
            id=f"run/output-{index}.txt",
            artifact_type="text/plain",
            size_bytes=100 if is_local else 20,
            locations=[invocation.input_artifacts[0].locations[0] if is_local else "gpu4090"],
        )
        return ExecutionResult(
            request_id=f"request-{index}",
            executor_id=(f"orin-{index}" if is_local else "gpu4090-vlm"),
            output_artifacts=[output],
            queue_ms=0,
            service_ms=10 if is_local else 50,
            metadata={"input_tokens": index, "output_tokens": 2, "api_cost_usd": 0.01},
        )


def test_video_task_uses_only_generic_registered_operator() -> None:
    interaction = build_video_task_interaction(
        "video-1", "Answer the long-video question.", _chunks(), "video_mme"
    )

    assert interaction.operators == ["sample_frames", "invoke_model"]
    assert [item.kind for item in interaction.initial_artifacts] == [
        "fixed_time_video_chunk",
        "fixed_time_video_chunk",
        "fixed_time_video_chunk",
    ]


def test_sweep_config_contains_no_gold_and_controls_only_network() -> None:
    config = CalibrationV0SweepConfig.model_validate(
        {
            "experiment_config": "experiment.yaml",
            "local_model_id": "local-video-vlm",
            "strong_model_id": "strong-video-vlm",
            "input_workers": ["a4", "a5", "a28"],
            "eligible_executor_ids": ["a4-vlm", "a5-vlm", "a28-vlm", "gpu-vlm"],
            "tasks": [
                {
                    "task_id": "video-mme-1",
                    "instruction": "What happened?",
                    "options": ["first", "second"],
                    "evaluator_id": "video_mme_official",
                    "inputs": [
                        {
                            "artifact_id": f"chunk-{index}",
                            "path": f"chunk-{index}.mp4",
                            "duration_s": 600,
                        }
                        for index in range(3)
                    ],
                }
            ],
            "worlds": [
                {
                    "world_id": "H1_distributed_constrained",
                    "bandwidth_mbps": 10,
                    "rtt_ms": 50,
                },
                {
                    "world_id": "H2_distributed_favorable",
                    "bandwidth_mbps": 100,
                    "rtt_ms": 0,
                },
            ],
        }
    )

    dumped = config.model_dump(mode="json")
    assert "expected_answer" not in str(dumped)
    assert config.warmup_runs == 1
    assert config.repeats == 3
    assert config.sample_count_per_chunk == 12
    assert config.input_source == "controller_upload"

    dumped["workflows"] = ["visual_reduction"]
    visual_config = CalibrationV0SweepConfig.model_validate(dumped)
    assert visual_config.workflows == ["visual_reduction"]

    dumped["repeats"] = 4
    with pytest.raises(ValueError, match="Input should be 3"):
        CalibrationV0SweepConfig.model_validate(dumped)


def test_formal_worker_local_reference_manifests_are_narrow_and_use_orin_paths() -> None:
    config_root = Path(__file__).parents[2] / "configs" / "calibration_v0"
    sweep_747 = CalibrationV0SweepConfig.from_yaml(
        config_root / "reference_747_sweep.yaml"
    )
    sweep_795 = CalibrationV0SweepConfig.from_yaml(
        config_root / "reference_795_all_workflows_sweep.yaml"
    )

    assert sweep_747.input_source == "worker_local"
    assert [task.task_id for task in sweep_747.tasks] == ["video_mme:747"]
    assert sweep_747.workflows == ["centralized_raw", "local_reduction"]
    assert (
        "What happened to the team on the counterattack after Sabonis' first steal?"
        in sweep_747.tasks[0].instruction
    )
    assert "How many athletic goals did LIT score" in sweep_747.tasks[0].instruction
    assert (
        "How many timeouts does the HUN actively consume throughout the game?"
        in sweep_747.tasks[0].instruction
    )
    assert all(
        item.path.as_posix().startswith(
            "/home/edge/xiaoming/calibration_v0/candidates/747/chunk-"
        )
        for item in sweep_747.tasks[0].inputs
    )
    assert [item.expected_size_bytes for item in sweep_747.tasks[0].inputs] == [
        108644854,
        111078348,
        110353684,
    ]

    assert sweep_795.input_source == "worker_local"
    assert [task.task_id for task in sweep_795.tasks] == ["video_mme:795"]
    assert sweep_795.workflows == [
        "centralized_raw",
        "visual_reduction",
        "local_reduction",
    ]
    assert all(
        item.path.as_posix().startswith(
            "/home/edge/xiaoming/calibration_v0/artifacts/795/chunk-"
        )
        for item in sweep_795.tasks[0].inputs
    )
    assert [item.expected_size_bytes for item in sweep_795.tasks[0].inputs] == [
        93677242,
        99396939,
        89370679,
    ]


def test_attempt_schedule_has_one_excluded_warmup_then_measured_repeats() -> None:
    assert _attempt_schedule(3) == (
        (0, True),
        (1, False),
        (2, False),
        (3, False),
    )


def test_resume_is_scoped_to_measurement_series_and_warmup(tmp_path: Path) -> None:
    raw_runs = tmp_path / "raw_runs.jsonl"
    base = {
        "task_id": "video_mme:795",
        "workflow_id": "centralized_raw",
        "world_id": "H1_distributed_constrained",
        "repeat": 1,
        "warmup": False,
        "status": "completed",
    }
    rows = [
        {**base, "run_id": "calibration-v0-video_mme-795-old"},
        {
            **base,
            "run_id": "calibration-v0-795-all-workflows-current",
            "measurement_series_id": "calibration-v0-795-all-workflows",
            "metadata": {
                "measurement_series_id": "calibration-v0-795-all-workflows"
            },
        },
        {
            **base,
            "run_id": "calibration-v0-795-all-workflows-warmup",
            "repeat": 0,
            "warmup": True,
        },
        {
            **base,
            "run_id": "other-series-current",
            "metadata": {"measurement_series_id": "other-series"},
        },
    ]
    raw_runs.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    assert _completed_keys(raw_runs, "calibration-v0-795-all-workflows") == {
        (
            "video_mme:795",
            "centralized_raw",
            "H1_distributed_constrained",
            1,
            False,
        ),
        (
            "video_mme:795",
            "centralized_raw",
            "H1_distributed_constrained",
            0,
            True,
        ),
    }


def test_trace_export_contains_timestamped_artifact_dependencies() -> None:
    registry = ExecutorRegistry.from_yaml(
        Path(__file__).parents[2] / "configs" / "calibration_v0" / "executors.yaml"
    )
    events = [
        {
            "event_type": "execution.request",
            "action_id": "local-a",
            "model_id": "local-video-vlm",
            "semantic_operator": "invoke_model",
            "input_artifacts": ["raw-a"],
        },
        {
            "event_type": "execution.request",
            "action_id": "local-b",
            "model_id": "local-video-vlm",
            "semantic_operator": "invoke_model",
            "input_artifacts": ["raw-b"],
        },
        {
            "event_type": "worker.execution.start",
            "action_id": "local-a",
            "timestamp": 0.0,
        },
        {
            "event_type": "worker.execution.start",
            "action_id": "local-b",
            "timestamp": 0.0,
        },
        {
            "event_type": "worker.execution.end",
            "action_id": "local-a",
            "timestamp": 100.0,
            "success": True,
            "executor": "a4-vlm",
            "worker_id": "a4",
            "service_ms": 100.0,
            "input_tokens": 10,
            "output_tokens": 5,
            "output_artifacts": ["evidence-a"],
        },
        {
            "event_type": "artifact.created",
            "action_id": "local-a",
            "artifact_id": "evidence-a",
            "size_bytes": 100,
        },
        {
            "event_type": "worker.execution.end",
            "action_id": "local-b",
            "timestamp": 200.0,
            "success": True,
            "executor": "a5-vlm",
            "worker_id": "a5",
            "service_ms": 200.0,
            "input_tokens": 10,
            "output_tokens": 5,
            "output_artifacts": ["evidence-b"],
        },
        {
            "event_type": "artifact.created",
            "action_id": "local-b",
            "artifact_id": "evidence-b",
            "size_bytes": 200,
        },
        {
            "event_type": "execution.request",
            "action_id": "remote",
            "model_id": "strong-video-vlm",
            "semantic_operator": "invoke_model",
            "input_artifacts": ["evidence-a", "evidence-b"],
        },
        {
            "event_type": "artifact.transfer.start",
            "action_id": "remote",
            "artifact_id": "evidence-a",
            "source_worker_id": "a4",
            "target_worker_id": "strong-4090",
            "timestamp": 100.0,
        },
        {
            "event_type": "artifact.transfer.end",
            "action_id": "remote",
            "artifact_id": "evidence-a",
            "source_worker_id": "a4",
            "target_worker_id": "strong-4090",
            "timestamp": 120.0,
            "success": True,
            "bytes_transferred": 100,
            "transfer_ms": 20.0,
        },
        {
            "event_type": "artifact.transfer.start",
            "action_id": "remote",
            "artifact_id": "evidence-b",
            "source_worker_id": "a5",
            "target_worker_id": "strong-4090",
            "timestamp": 200.0,
        },
        {
            "event_type": "artifact.transfer.end",
            "action_id": "remote",
            "artifact_id": "evidence-b",
            "source_worker_id": "a5",
            "target_worker_id": "strong-4090",
            "timestamp": 230.0,
            "success": True,
            "bytes_transferred": 200,
            "transfer_ms": 30.0,
        },
        {
            "event_type": "worker.execution.start",
            "action_id": "remote",
            "timestamp": 230.0,
        },
        {
            "event_type": "worker.execution.end",
            "action_id": "remote",
            "timestamp": 280.0,
            "success": True,
            "executor": "strong-4090-vlm",
            "worker_id": "strong-4090",
            "service_ms": 50.0,
            "input_tokens": 20,
            "output_tokens": 2,
            "output_artifacts": ["answer"],
        },
        {
            "event_type": "artifact.created",
            "action_id": "remote",
            "artifact_id": "answer",
            "size_bytes": 20,
        },
    ]
    transfers = _transfer_rows(events, registry)
    executions = _execution_rows(
        events,
        registry,
        {"raw-a": 1_000, "raw-b": 2_000},
        transfers,
    )
    trace = _realized_trace(
        run_id="run",
        task_id="task",
        workflow_id="local_reduction",
        warmup=False,
        executions=executions,
        transfers=transfers,
        initial_artifact_sites={"raw-a": "A4", "raw-b": "A5"},
        e2e_latency_ms=280.0,
    )

    assert [item["depends_on"] for item in transfers] == [
        ["local-a"],
        ["local-b"],
    ]
    remote = next(item for item in executions if item["action_id"] == "remote")
    assert remote["depends_on"] == [
        "remote:transfer-1",
        "remote:transfer-2",
    ]
    assert remote["input_artifacts"] == ["evidence-a", "evidence-b"]
    assert remote["output_artifacts"] == ["answer"]
    assert trace["timestamp_evidence"] == "complete"
    assert trace["dependency_evidence"] == "complete"
    assert trace["trace_coverage"] == "complete"
    assert len(cast(list[object], trace["spans"])) == 5

    incomplete_executions = [dict(item) for item in executions]
    next(
        item for item in incomplete_executions if item["action_id"] == "remote"
    )["depends_on"] = []
    incomplete = _realized_trace(
        run_id="run-incomplete",
        task_id="task",
        workflow_id="local_reduction",
        warmup=False,
        executions=incomplete_executions,
        transfers=transfers,
        initial_artifact_sites={"raw-a": "A4", "raw-b": "A5"},
        e2e_latency_ms=280.0,
    )
    assert incomplete["dependency_evidence"] == "partial"
    assert incomplete["trace_coverage"] == "partial"

    missing_initial_transfer = _realized_trace(
        run_id="run-missing-raw-transfer",
        task_id="task",
        workflow_id="centralized_raw",
        warmup=False,
        executions=executions,
        transfers=transfers,
        initial_artifact_sites={"raw-a": "A28", "raw-b": "A5"},
        e2e_latency_ms=280.0,
    )
    assert missing_initial_transfer["dependency_evidence"] == "partial"
    assert missing_initial_transfer["trace_coverage"] == "partial"


async def test_centralized_raw_sends_all_chunks_to_strong_model() -> None:
    runtime = FakeRuntime()

    result = await execute_video_reference_workflow(
        runtime,
        workflow_id="centralized_raw",
        task="What happened across the video?",
        raw_video_chunks=_chunks(),
        local_model_id="local-video",
        strong_model_id="strong-video",
        reasoning_worker_id="gpu4090",
        chunk_durations_s=[600, 600, 600],
    )

    assert len(runtime.invocations) == 1
    assert runtime.invocations[0].model_id == "strong-video"
    assert len(runtime.invocations[0].input_artifacts) == 3
    assert [target for _, target, _ in runtime.samples] == ["gpu4090"] * 3
    assert result.raw_artifact_bytes == 6_000
    assert result.reduced_artifact_bytes == 0
    assert result.reduced_visual_bytes == 0
    assert result.semantic_evidence_bytes == 0
    assert result.local_service_ms == 0
    assert result.remote_service_ms == 50


async def test_local_reduction_runs_one_local_call_per_chunk_and_counts_cost() -> None:
    runtime = FakeRuntime()

    result = await execute_video_reference_workflow(
        runtime,
        workflow_id="local_reduction",
        task="What happened across the video?",
        raw_video_chunks=_chunks(),
        local_model_id="local-video",
        strong_model_id="strong-video",
        reasoning_worker_id="gpu4090",
        chunk_durations_s=[600, 600, 600],
    )

    local = [item for item in runtime.invocations if item.model_id == "local-video"]
    remote = [item for item in runtime.invocations if item.model_id == "strong-video"]
    assert len(local) == 3
    assert [target for _, target, _ in runtime.samples] == ["a4", "a5", "a28"]
    assert len(remote) == 1
    assert [item.size_bytes for item in remote[0].input_artifacts] == [100, 100, 100]
    assert result.reduced_artifact_bytes == 300
    assert result.reduced_visual_bytes == 0
    assert result.semantic_evidence_bytes == 300
    assert result.local_service_ms == 45
    assert result.remote_service_ms == 50
    assert result.input_tokens == 10
    assert result.output_tokens == 8
    assert result.api_cost_usd == 0.04


async def test_visual_reduction_transfers_local_contact_sheets_to_strong_model() -> None:
    runtime = FakeRuntime()

    result = await execute_video_reference_workflow(
        runtime,
        workflow_id="visual_reduction",
        task="What happened across the video?",
        raw_video_chunks=_chunks(),
        local_model_id="local-video",
        strong_model_id="strong-video",
        reasoning_worker_id="gpu4090",
        chunk_durations_s=[600, 600, 600],
    )

    assert [target for _, target, _ in runtime.samples] == ["a4", "a5", "a28"]
    assert len(runtime.invocations) == 1
    remote = runtime.invocations[0]
    assert remote.model_id == "strong-video"
    assert [item.artifact_type for item in remote.input_artifacts] == [
        "image/jpeg",
        "image/jpeg",
        "image/jpeg",
    ]
    assert [item.locations for item in remote.input_artifacts] == [
        ["a4"],
        ["a5"],
        ["a28"],
    ]
    assert result.reduced_artifact_bytes == 600
    assert result.reduced_visual_bytes == 600
    assert result.semantic_evidence_bytes == 0
    assert result.local_service_ms == 15
    assert result.remote_service_ms == 50
    assert result.input_tokens == 1
    assert result.output_tokens == 2
    assert result.api_cost_usd == 0.01
