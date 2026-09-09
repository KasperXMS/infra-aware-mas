"""Integration tests for real-machine experiment assembly boundaries."""

import json
from contextlib import AsyncExitStack
from pathlib import Path
from typing import cast

import httpx
import pytest
import yaml

from infra_mas.core.agent import AgentSpec
from infra_mas.core.executor import ExecutorSpec
from infra_mas.execution.executor_registry import ExecutorRegistry
from infra_mas.execution.worker_client import WorkerClient
from infra_mas.experiment import (
    FixedSchedulerConfig,
    PreflightResult,
    build_blind_scheduler,
    preflight_workers,
    run_blind_experiment,
    upload_inputs,
)
from infra_mas.planner.model_factory import PlannerModelConfig
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


class FakeClosable:
    async def aclose(self) -> None:
        return None

    async def close(self) -> None:
        return None


class FakeCoordinator:
    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs

    async def run(self, task: str) -> str:
        assert task == "Reproduce this run."
        return "answer"


async def test_run_saves_complete_effective_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "agents.yaml").write_text(
        """agents:
  reasoner:
    capability: reasoning
    instructions: Reason exactly.
""",
        encoding="utf-8",
    )
    (tmp_path / "executors.yaml").write_text(
        """workers:
  worker-a:
    endpoint: http://worker-a.test
    site: local
executors:
  worker-a-llm:
    worker_id: worker-a
    capability: reasoning
    model: test-llm
    device: cpu
    site: local
""",
        encoding="utf-8",
    )
    config_path = tmp_path / "blind.yaml"
    config_path.write_text(
        """agents_config: agents.yaml
executors_config: executors.yaml
runs_root: runs
temporary_root: runtime
input_worker: worker-a
scheduler:
  type: fixed
  assignments:
    reasoning: worker-a-llm
planner:
  model: planner-model
  api: chat_completions
  base_url: http://planner.test/v1
  api_key_env: null
""",
        encoding="utf-8",
    )

    async def fake_preflight(*args: object, **kwargs: object) -> PreflightResult:
        del args, kwargs
        return PreflightResult(workers={"worker-a": ["worker-a-llm"]})

    fake_worker = FakeClosable()
    fake_planner = FakeClosable()

    def fake_create_worker_clients(
        registry: ExecutorRegistry,
        timeout_seconds: float,
    ) -> dict[str, WorkerClient]:
        del registry, timeout_seconds
        return {"worker-a": cast(WorkerClient, fake_worker)}

    def fake_create_planner_model(
        config: PlannerModelConfig,
    ) -> tuple[object, FakeClosable]:
        del config
        return object(), fake_planner

    monkeypatch.setattr(
        "infra_mas.experiment.create_worker_clients",
        fake_create_worker_clients,
    )
    monkeypatch.setattr("infra_mas.experiment.preflight_workers", fake_preflight)
    monkeypatch.setattr(
        "infra_mas.experiment.create_planner_model",
        fake_create_planner_model,
    )
    monkeypatch.setattr("infra_mas.experiment.Coordinator", FakeCoordinator)

    answer, run_directory = await run_blind_experiment(
        config_path,
        "Reproduce this run.",
        [],
        run_id="run-config",
    )

    assert answer == "answer"
    snapshot = yaml.safe_load((run_directory / "config.yaml").read_text(encoding="utf-8"))
    assert snapshot["run_id"] == "run-config"
    assert snapshot["task"] == "Reproduce this run."
    assert snapshot["resource_aware"] is False
    assert snapshot["agents"] == [
        {
            "name": "reasoner",
            "capability": "reasoning",
            "instructions": "Reason exactly.",
        }
    ]
    assert snapshot["executors"][0]["id"] == "worker-a-llm"
    assert snapshot["worker_endpoints"] == {"worker-a": "http://worker-a.test"}
    assert snapshot["scheduler"] == {
        "type": "fixed",
        "assignments": {"reasoning": "worker-a-llm"},
    }
    assert snapshot["planner"] == {
        "model": "planner-model",
        "api": "chat_completions",
        "base_url": "http://planner.test/v1",
        "api_key_env": None,
        "timeout_seconds": 120.0,
    }
    events = [
        json.loads(line)
        for line in (run_directory / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [event["event_type"] for event in events] == ["run.start", "run.end"]
    assert json.loads((run_directory / "result.json").read_text(encoding="utf-8")) == {
        "success": True,
        "answer": "answer",
    }

    original_trace = (run_directory / "trace.jsonl").read_bytes()
    with pytest.raises(ValueError, match="already exists and is non-empty"):
        await run_blind_experiment(
            config_path,
            "Reproduce this run.",
            [],
            run_id="run-config",
        )
    assert (run_directory / "trace.jsonl").read_bytes() == original_trace
