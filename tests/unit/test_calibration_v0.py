from __future__ import annotations

from infra_mas.calibration_v0 import (
    CalibrationV0SweepConfig,
    build_video_task_interaction,
    execute_video_reference_workflow,
)
from infra_mas.core.artifact import ArtifactRef
from infra_mas.core.execution import ExecutionResult, InvocationSpec


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

    async def sample_frames(
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
    assert config.repeats == 3
    assert config.sample_count_per_chunk == 12


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
    assert result.local_service_ms == 45
    assert result.remote_service_ms == 50
    assert result.input_tokens == 10
    assert result.output_tokens == 8
    assert result.api_cost_usd == 0.04
