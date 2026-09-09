"""Tests for deterministic, workflow-agnostic planner state."""

import asyncio

import pytest

from infra_mas.planner.ledger import CompletedInvocation, PlanningLedger


def completion(
    action_id: str,
    input_artifact_ids: list[str],
    output_artifact_ids: list[str],
) -> CompletedInvocation:
    return CompletedInvocation(
        action_id=action_id,
        model_id="test-model",
        role="ad_hoc_researcher",
        task="Investigate the evidence.",
        input_artifact_ids=input_artifact_ids,
        output_artifact_ids=output_artifact_ids,
    )


async def test_ledger_counts_initial_input_use_and_unused_inputs() -> None:
    ledger = PlanningLedger(["run/input-a", "run/input-b", "run/input-c"])

    await ledger.record_completed(
        completion("run/spawn-0001", ["run/input-a"], ["run/output-a"])
    )
    state = await ledger.record_completed(
        completion(
            "run/spawn-0002",
            ["run/input-a", "run/input-b", "run/input-a"],
            ["run/output-b"],
        )
    )

    assert [item.model_dump() for item in state.initial_inputs] == [
        {"artifact_id": "run/input-a", "use_count": 2},
        {"artifact_id": "run/input-b", "use_count": 1},
        {"artifact_id": "run/input-c", "use_count": 0},
    ]
    assert state.unused_initial_inputs == ["run/input-c"]
    assert [item.action_id for item in state.completed_invocations] == [
        "run/spawn-0001",
        "run/spawn-0002",
    ]


async def test_ledger_history_order_is_independent_of_completion_order() -> None:
    ledger = PlanningLedger(["run/input"])

    await asyncio.gather(
        ledger.record_completed(
            completion("run/spawn-0002", ["run/input"], ["run/output-b"])
        ),
        ledger.record_completed(
            completion("run/delegate-0001", [], ["run/output-a"])
        ),
    )

    state = await ledger.snapshot()
    assert [item.action_id for item in state.completed_invocations] == [
        "run/delegate-0001",
        "run/spawn-0002",
    ]


async def test_ledger_rejects_duplicate_completion() -> None:
    ledger = PlanningLedger([])
    invocation = completion("run/spawn-0001", [], ["run/output"])
    await ledger.record_completed(invocation)

    with pytest.raises(ValueError, match="already recorded"):
        await ledger.record_completed(invocation)
