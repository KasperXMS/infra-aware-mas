import json
from pathlib import Path

from infra_mas.bench import (
    ArtifactBinding,
    MASRunSpec,
    export_realized_workflow,
    resolve_source_path,
    summarize_mas_run,
)
from infra_mas.operators import (
    ExternalEvaluatorSpec,
    InitialArtifactSpec,
    ObservationSpec,
    RuntimeVerifierSpec,
    TaskInteractionSpec,
)


def _task_interaction() -> TaskInteractionSpec:
    return TaskInteractionSpec(
        task_id="task",
        objective="inspect",
        initial_artifacts=[
            InitialArtifactSpec(artifact_id="img_01", kind="image", source_ref="one.jpg"),
            InitialArtifactSpec(artifact_id="img_02", kind="image", source_ref="two.jpg"),
        ],
        operators=["invoke_model"],
        observations=[
            ObservationSpec(
                observation_id="model_output",
                produced_by=["invoke_model"],
                description="Model output.",
            )
        ],
        runtime_verifier=RuntimeVerifierSpec(level="none"),
        external_evaluator=ExternalEvaluatorSpec(evaluator_id="hidden"),
    )


def test_realized_workflow_uses_artifact_dependencies_not_timestamp_order(
    tmp_path: Path,
) -> None:
    events = [
        {
            "event_type": "execution.request",
            "action_id": "a",
            "parent_action_id": "planner-a",
            "timestamp": "2026-01-01T00:00:00Z",
            "model_id": "vlm",
            "agent": "first",
            "input_artifacts": ["raw-1"],
        },
        {
            "event_type": "execution.request",
            "action_id": "b",
            "parent_action_id": "planner-b",
            "timestamp": "2026-01-01T00:00:01Z",
            "model_id": "vlm",
            "agent": "parallel",
            "input_artifacts": ["raw-2"],
        },
        {
            "event_type": "worker.execution.end",
            "action_id": "a",
            "timestamp": "2026-01-01T00:00:03Z",
            "success": True,
            "output_artifacts": ["derived-a"],
        },
        {
            "event_type": "worker.execution.end",
            "action_id": "b",
            "timestamp": "2026-01-01T00:00:02Z",
            "success": True,
            "output_artifacts": ["derived-b"],
        },
        {
            "event_type": "execution.request",
            "action_id": "c",
            "parent_action_id": "planner-c",
            "timestamp": "2026-01-01T00:00:04Z",
            "model_id": "vlm",
            "agent": "synthesis",
            "input_artifacts": ["derived-a", "derived-b"],
        },
        {
            "event_type": "worker.execution.end",
            "action_id": "c",
            "timestamp": "2026-01-01T00:00:05Z",
            "success": True,
            "output_artifacts": ["final"],
        },
    ]
    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
    )

    workflow = export_realized_workflow(trace)

    assert workflow.dependencies == [("a", "c"), ("b", "c")]
    assert ("a", "b") not in workflow.dependencies
    assert ("b", "a") not in workflow.dependencies


def test_source_path_accepts_native_windows_absolute_path(tmp_path: Path) -> None:
    windows_path = r"D:\benchmark\images\one.jpg"

    assert resolve_source_path(windows_path, tmp_path) == Path(windows_path).resolve()


def test_summary_computes_cross_worker_alignment_and_grouping(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    events = [
        {"event_type": "run.start", "run_id": "run"},
        {
            "event_type": "planner.ledger.initialized",
            "planning_state": {
                "initial_inputs": [
                    {"artifact_id": "run/input-001-img_01.jpg"},
                    {"artifact_id": "run/input-002-img_02.jpg"},
                ]
            },
        },
        {
            "event_type": "execution.request",
            "action_id": "request-a",
            "model_id": "edge-vlm",
            "agent": "auditor",
            "input_artifacts": [
                "run/input-001-img_01.jpg",
                "run/input-002-img_02.jpg",
            ],
        },
        {
            "event_type": "executor.selected",
            "action_id": "request-a",
            "executor": "a4-vlm",
            "site": "A4",
        },
        {
            "event_type": "artifact.transfer.end",
            "action_id": "request-a",
            "artifact_id": "run/input-002-img_02.jpg",
            "source_worker_id": "a5",
            "target_worker_id": "a4",
            "bytes_transferred": 20,
            "transfer_ms": 2.0,
            "success": True,
        },
        {
            "event_type": "worker.execution.end",
            "action_id": "request-a",
            "output_artifacts": ["run/output-a"],
            "service_ms": 10.0,
            "success": True,
        },
    ]
    (run / "trace.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
    )
    (run / "result.json").write_text(
        json.dumps({"answer": "img_01", "e2e_ms": 20.0}), encoding="utf-8"
    )
    spec = MASRunSpec(
        schema_version="mas-run-spec-v2",
        case_id="case",
        group_id="group",
        task_id="task",
        instruction="inspect",
        task_interaction=_task_interaction(),
        artifacts=[
            ArtifactBinding(
                artifact_id="img_01", source_ref="one.jpg", site_id="A4"
            ),
            ArtifactBinding(
                artifact_id="img_02", source_ref="two.jpg", site_id="A5"
            ),
        ],
        infra_world={},
    )

    result = summarize_mas_run(run, spec, "snapshot")

    assert result.cross_worker_transfer_bytes == 20
    assert result.cross_worker_transfer_count == 1
    assert result.multimodal_invocation_count == 1
    assert result.site_aligned_multimodal_invocations == 0
    assert result.site_alignment_rate == 0
    assert result.artifact_grouping == [["img_01", "img_02"]]
    assert result.initial_artifact_grouping == [["img_01", "img_02"]]
    assert result.later_refinement_invocation_count == 0
    assert result.later_refinement_cross_worker_bytes == 0
    assert result.later_refinement_service_ms == 0


def test_summary_separates_initial_coverage_from_later_refinement(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    run.mkdir()
    events = [
        {"event_type": "run.start", "run_id": "run"},
        {
            "event_type": "planner.ledger.initialized",
            "planning_state": {
                "initial_inputs": [
                    {"artifact_id": "raw-1"},
                    {"artifact_id": "raw-2"},
                ]
            },
        },
        {
            "event_type": "execution.request",
            "action_id": "initial-a",
            "model_id": "edge-vlm",
            "input_artifacts": ["raw-1"],
        },
        {
            "event_type": "worker.execution.end",
            "action_id": "initial-a",
            "output_artifacts": ["evidence-1"],
            "service_ms": 10.0,
            "success": True,
        },
        {
            "event_type": "execution.request",
            "action_id": "initial-b",
            "model_id": "edge-vlm",
            "input_artifacts": ["raw-2"],
        },
        {
            "event_type": "worker.execution.end",
            "action_id": "initial-b",
            "output_artifacts": ["evidence-2"],
            "service_ms": 11.0,
            "success": True,
        },
        {
            "event_type": "execution.request",
            "action_id": "verify",
            "model_id": "edge-vlm",
            "input_artifacts": ["raw-2"],
        },
        {
            "event_type": "artifact.transfer.end",
            "action_id": "verify",
            "artifact_id": "raw-2",
            "source_worker_id": "a5",
            "target_worker_id": "a4",
            "bytes_transferred": 42,
            "transfer_ms": 2.0,
            "success": True,
        },
        {
            "event_type": "worker.execution.end",
            "action_id": "verify",
            "output_artifacts": ["verified"],
            "service_ms": 12.0,
            "success": True,
        },
    ]
    (run / "trace.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
    )
    (run / "result.json").write_text(
        json.dumps({"answer": "ANSWER: img_02", "e2e_ms": 40.0}),
        encoding="utf-8",
    )
    spec = MASRunSpec(
        schema_version="mas-run-spec-v2",
        case_id="case",
        group_id="group",
        task_id="task",
        instruction="inspect",
        task_interaction=_task_interaction(),
        artifacts=[
            ArtifactBinding(artifact_id="img_01", source_ref="one.jpg", site_id="A4"),
            ArtifactBinding(artifact_id="img_02", source_ref="two.jpg", site_id="A5"),
        ],
        infra_world={},
    )

    result = summarize_mas_run(run, spec, "snapshot")

    assert result.initial_artifact_grouping == [["img_01"], ["img_02"]]
    assert result.later_refinement_invocation_count == 1
    assert result.later_refinement_cross_worker_bytes == 42
    assert result.later_refinement_service_ms == 12
