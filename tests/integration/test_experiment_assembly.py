"""Integration tests for real-machine experiment assembly boundaries."""

from contextlib import AsyncExitStack
from pathlib import Path

import httpx
import pytest

from infra_mas.core.agent import AgentSpec
from infra_mas.core.executor import ExecutorSpec
from infra_mas.execution.executor_registry import ExecutorRegistry
from infra_mas.execution.worker_client import WorkerClient
from infra_mas.experiment import (
    FixedSchedulerConfig,
    build_blind_scheduler,
    preflight_workers,
    upload_inputs,
)
from infra_mas.runtime.agent_registry import AgentRegistry
from infra_mas.worker.artifact_store import ArtifactStore
from infra_mas.worker.backends.mock import MockBackend
from infra_mas.worker.server import create_app
from infra_mas.worker.service import WorkerExecutor, WorkerService


async def test_preflight_and_initial_artifact_upload(tmp_path: Path) -> None:
    service = WorkerService(
        "worker-a",
        ArtifactStore(tmp_path / "artifacts", "worker-a"),
        [WorkerExecutor("worker-a-vision", "visual_understanding", MockBackend())],
    )
    registry = ExecutorRegistry(
        [
            ExecutorSpec(
                id="worker-a-vision",
                capability="visual_understanding",
                worker_id="worker-a",
                model="mock",
                device="test",
                site="local",
            )
        ],
        {"worker-a": "http://worker-a.test"},
    )
    source = tmp_path / "input.jpg"
    source.write_bytes(b"image")

    async with AsyncExitStack() as stack:
        http_client = await stack.enter_async_context(
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=create_app(service)),
                base_url="http://worker-a.test",
            )
        )
        clients = {"worker-a": WorkerClient(client=http_client)}
        checked = await preflight_workers(registry, clients)
        uploaded = await upload_inputs([source], "run-001", "worker-a", clients["worker-a"])

    assert checked.workers == {"worker-a": ["worker-a-vision"]}
    assert uploaded[0].id == "run-001/input-001-input.jpg"
    assert uploaded[0].artifact_type == "image/jpeg"
    assert uploaded[0].locations == ["worker-a"]


async def test_preflight_rejects_executor_drift(tmp_path: Path) -> None:
    service = WorkerService(
        "worker-a",
        ArtifactStore(tmp_path / "artifacts", "worker-a"),
        [WorkerExecutor("unexpected", "reasoning", MockBackend())],
    )
    registry = ExecutorRegistry([], {"worker-a": "http://worker-a.test"})

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_app(service)),
        base_url="http://worker-a.test",
    ) as http_client:
        with pytest.raises(ValueError, match="exposes executors"):
            await preflight_workers(registry, {"worker-a": WorkerClient(client=http_client)})


def test_fixed_scheduler_requires_every_agent_capability() -> None:
    agents = AgentRegistry([AgentSpec(name="reasoner", capability="reasoning", instructions="x")])
    registry = ExecutorRegistry(
        [
            ExecutorSpec(
                id="worker-a-llm",
                capability="reasoning",
                worker_id="worker-a",
                model="mock",
                device="test",
                site="local",
            )
        ],
        {"worker-a": "http://worker-a.test"},
    )

    with pytest.raises(ValueError, match="no assignments"):
        build_blind_scheduler(FixedSchedulerConfig(type="fixed", assignments={}), registry, agents)
