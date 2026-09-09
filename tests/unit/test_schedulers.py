"""Unit tests for resource-blind scheduling strategies."""

import asyncio

import pytest

from infra_mas.core.errors import NoExecutorAvailableError
from infra_mas.core.execution import InvocationSpec
from infra_mas.core.executor import ExecutorSpec
from infra_mas.execution.executor_registry import ExecutorRegistry
from infra_mas.scheduler.fixed import FixedScheduler
from infra_mas.scheduler.round_robin import RoundRobinScheduler


def executor(
    executor_id: str,
    capability: str = "reasoning",
    model_id: str = "mock",
) -> ExecutorSpec:
    return ExecutorSpec(
        id=executor_id,
        capability=capability,
        worker_id=executor_id,
        model_id=model_id,
        device="cpu",
        site="local",
    )


def invocation(model_id: str = "mock") -> InvocationSpec:
    return InvocationSpec(
        model_id=model_id,
        role="reasoner",
        instructions="Reason carefully.",
        task="Reason.",
        input_artifacts=[],
    )


def registry() -> ExecutorRegistry:
    executors = [executor("executor-a"), executor("executor-b")]
    endpoints = {item.worker_id: f"http://{item.worker_id}.test" for item in executors}
    return ExecutorRegistry(executors, endpoints)


async def test_fixed_scheduler_selects_configured_executor() -> None:
    scheduler = FixedScheduler(registry(), {"mock": "executor-b"})

    selected = await scheduler.select(invocation())

    assert selected.id == "executor-b"


async def test_fixed_scheduler_rejects_unmapped_model() -> None:
    scheduler = FixedScheduler(registry(), {})

    with pytest.raises(NoExecutorAvailableError, match="missing-model"):
        await scheduler.select(invocation("missing-model"))


async def test_round_robin_scheduler_rotates_compatible_executors() -> None:
    scheduler = RoundRobinScheduler(registry())

    selected = [await scheduler.select(invocation()) for _ in range(3)]

    assert [item.id for item in selected] == ["executor-a", "executor-b", "executor-a"]


async def test_round_robin_scheduler_is_safe_for_concurrent_calls() -> None:
    scheduler = RoundRobinScheduler(registry())

    selected = await asyncio.gather(*(scheduler.select(invocation()) for _ in range(10)))

    assert [item.id for item in selected].count("executor-a") == 5
    assert [item.id for item in selected].count("executor-b") == 5


async def test_round_robin_scheduler_rejects_unsupported_model() -> None:
    scheduler = RoundRobinScheduler(registry())

    with pytest.raises(NoExecutorAvailableError, match="missing-model"):
        await scheduler.select(invocation("missing-model"))


async def test_round_robin_rotates_only_replicas_of_requested_model() -> None:
    executors = [
        executor("small-a", model_id="small"),
        executor("large", model_id="large"),
        executor("small-b", model_id="small"),
    ]
    registry_with_models = ExecutorRegistry(
        executors,
        {item.worker_id: f"http://{item.worker_id}.test" for item in executors},
    )
    scheduler = RoundRobinScheduler(registry_with_models)

    selected = [await scheduler.select(invocation("small")) for _ in range(3)]

    assert [item.id for item in selected] == ["small-a", "small-b", "small-a"]
