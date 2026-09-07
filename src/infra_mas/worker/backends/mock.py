"""Deterministic mock model backend."""

import asyncio

from infra_mas.core.errors import ExecutionFailedError
from infra_mas.core.model import ModelRequest, ModelResult


class MockBackend:
    """Return deterministic output with an optional integration-test delay."""

    def __init__(
        self,
        latency_ms: float = 0.0,
        output: str = "mock result",
        *,
        fail: bool = False,
        real_delay: bool = False,
    ) -> None:
        if latency_ms < 0:
            raise ValueError("latency_ms must be non-negative")
        if not output.strip():
            raise ValueError("output must not be empty")

        self._latency_ms = latency_ms
        self._output = output
        self._fail = fail
        self._real_delay = real_delay

    async def infer(self, request: ModelRequest) -> ModelResult:
        """Return configured output without sleeping unless explicitly requested."""
        del request

        if self._real_delay and self._latency_ms:
            await asyncio.sleep(self._latency_ms / 1000)
        if self._fail:
            raise ExecutionFailedError("mock model execution failed")

        return ModelResult(output_text=self._output, latency_ms=self._latency_ms)
