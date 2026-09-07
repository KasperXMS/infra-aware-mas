"""Unit tests for resource-blind scheduling strategies."""

import asyncio

import pytest

from infra_mas.core.errors import NoExecutorAvailableError
from infra_mas.core.execution import ExecutionRequest
from infra_mas.core.executor import ExecutorSpec
from infra_mas.execution.executor_registry import ExecutorRegistry
from infra_mas.scheduler.fixed import FixedScheduler
from infra_mas.scheduler.round_robin import RoundRobinScheduler


def executor(executor_id: str, capability: str = "reasoning") -> ExecutorSpec:
    return ExecutorSpec(
        id=executor_id,
        capability=capability,
        worker_id=executor_id,
        model="mock",
        device="cpu",
        site="local",
    )


def request(capability: str = "reasoning") -> ExecutionRequest:
    return ExecutionRequest(
        request_id="request-001",
        agent="reasoner",
        capability=capability,
        instructions="Reason carefully.",
        task="Reason.",
        inputs=[],
    )


def registry() -> ExecutorRegistry:
    executors = [executor("executor-a"), executor("executor-b")]
    endpoints = {item.worker_id: f"http://{item.worker_id}.test" for item in executors}
    return ExecutorRegistry(executors, endpoints)


async def test_fixed_scheduler_selects_configured_executor() -> None:
    scheduler = FixedScheduler(registry(), {"reasoning": "executor-b"})

    selected = await scheduler.select(request())

    assert selected.id == "executor-b"


async def test_fixed_scheduler_rejects_unmapped_capability() -> None:
    scheduler = FixedScheduler(registry(), {})

    with pytest.raises(NoExecutorAvailableError, match="summarization"):
        await scheduler.select(request("summarization"))


async def test_round_robin_scheduler_rotates_compatible_executors() -> None:
    scheduler = RoundRobinScheduler(registry())

    selected = [await scheduler.select(request()) for _ in range(3)]

    assert [item.id for item in selected] == ["executor-a", "executor-b", "executor-a"]


async def test_round_robin_scheduler_is_safe_for_concurrent_calls() -> None:
    scheduler = RoundRobinScheduler(registry())

    selected = await asyncio.gather(*(scheduler.select(request()) for _ in range(10)))

    assert [item.id for item in selected].count("executor-a") == 5
    assert [item.id for item in selected].count("executor-b") == 5


async def test_round_robin_scheduler_rejects_unsupported_capability() -> None:
    scheduler = RoundRobinScheduler(registry())

    with pytest.raises(NoExecutorAvailableError, match="summarization"):
        await scheduler.select(request("summarization"))
