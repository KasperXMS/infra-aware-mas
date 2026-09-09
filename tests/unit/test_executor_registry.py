"""Unit tests for the static physical executor registry."""

from pathlib import Path

import pytest

from infra_mas.core.errors import ExecutorNotFoundError
from infra_mas.core.executor import ExecutorSpec
from infra_mas.execution.executor_registry import ExecutorRegistry


def executor(executor_id: str = "worker-1-llm") -> ExecutorSpec:
    return ExecutorSpec(
        id=executor_id,
        capability="reasoning",
        worker_id="worker-1",
        model_id="mock",
        device="cpu",
        site="local",
    )


def test_registry_lookup_filter_and_endpoint_resolution() -> None:
    registry = ExecutorRegistry([executor()], {"worker-1": "http://worker-1.test"})

    assert registry.get("worker-1-llm").worker_id == "worker-1"
    assert [item.id for item in registry.candidates("reasoning")] == ["worker-1-llm"]
    assert [item.id for item in registry.model_candidates("mock")] == ["worker-1-llm"]
    assert registry.executor_endpoint("worker-1-llm") == "http://worker-1.test"


def test_registry_rejects_duplicate_executor_ids() -> None:
    with pytest.raises(ValueError, match="unique"):
        ExecutorRegistry(
            [executor(), executor()],
            {"worker-1": "http://worker-1.test"},
        )


def test_registry_rejects_unknown_worker_reference() -> None:
    with pytest.raises(ValueError, match="unknown workers"):
        ExecutorRegistry([executor()], {})


def test_unknown_executor_raises_typed_error() -> None:
    registry = ExecutorRegistry([], {})

    with pytest.raises(ExecutorNotFoundError, match="missing"):
        registry.get("missing")


def test_registry_loads_yaml(tmp_path: Path) -> None:
    path = tmp_path / "executors.yaml"
    path.write_text(
        """workers:
  worker-1:
    endpoint: http://worker-1.test
    site: local
executors:
  worker-1-llm:
    worker_id: worker-1
    capability: reasoning
    model: mock
    device: cpu
    site: local
""",
        encoding="utf-8",
    )

    registry = ExecutorRegistry.from_yaml(path)

    assert registry.get("worker-1-llm") == executor()
