"""Unit tests for the deterministic mock backend."""

import pytest

from infra_mas.core.errors import ExecutionFailedError
from infra_mas.core.model import ModelRequest
from infra_mas.worker.backends.mock import MockBackend


async def test_mock_backend_returns_configured_result_without_delay() -> None:
    backend = MockBackend(latency_ms=250, output="deterministic")

    result = await backend.infer(ModelRequest(task="test", input_paths=[]))

    assert result.output_text == "deterministic"
    assert result.latency_ms == 250


async def test_mock_backend_supports_artificial_failure() -> None:
    backend = MockBackend(fail=True)

    with pytest.raises(ExecutionFailedError, match="mock model execution failed"):
        await backend.infer(ModelRequest(task="test", input_paths=[]))
