from infra_mas.calibration import CalibrationRow, build_calibration_summary


def _row(world: str, workflow: str, repeat: int, e2e_ms: float) -> CalibrationRow:
    return CalibrationRow.model_validate(
        {
            "world_id": world,
            "workflow_id": workflow,
            "repeat": repeat,
            "run_id": f"{world}-{workflow}-{repeat}",
            "e2e_ms": e2e_ms,
            "service_ms_sum": e2e_ms,
            "transfer_bytes": 0,
            "transfer_ms_sum": 0,
            "vlm_calls": (
                1
                if workflow == "centralized"
                else 6
                if workflow == "distributed_6x1"
                else 3
            ),
            "synthesis_calls": 0 if workflow == "centralized" else 1,
            "success": True,
            "correct": True,
            "final_answer": "ANSWER: img_06",
        }
    )


def test_calibration_admits_only_correct_reversal_with_twenty_percent_margins() -> None:
    rows = [
        _row("single", "centralized", 1, 100),
        _row("single", "centralized", 2, 100),
        _row("single", "distributed_3x2", 1, 140),
        _row("single", "distributed_3x2", 2, 140),
        _row("multi", "centralized", 1, 150),
        _row("multi", "centralized", 2, 150),
        _row("multi", "distributed_3x2", 1, 100),
        _row("multi", "distributed_3x2", 2, 100),
    ]

    summary = build_calibration_summary(rows)

    assert summary["semantic_switch_pair_found"] is True
    assert {summary["reference_winner_a"], summary["reference_winner_b"]} == {
        "centralized",
        "distributed_3x2",
    }


def test_calibration_can_admit_six_by_one_reference_pair() -> None:
    rows = [
        _row("single", "centralized", 1, 100),
        _row("single", "centralized", 2, 100),
        _row("single", "distributed_6x1", 1, 130),
        _row("single", "distributed_6x1", 2, 130),
        _row("multi", "centralized", 1, 150),
        _row("multi", "centralized", 2, 150),
        _row("multi", "distributed_6x1", 1, 100),
        _row("multi", "distributed_6x1", 2, 100),
    ]

    summary = build_calibration_summary(rows)

    assert summary["semantic_switch_pair_found"] is True
    assert summary["compared_workflow"] == "distributed_6x1"
    assert summary["workflow_1"] == "distributed_6x1"
    assert summary["workflow_2"] == "centralized"
