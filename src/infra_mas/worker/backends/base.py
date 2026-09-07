"""Model backend interface."""

from typing import Protocol

from infra_mas.core.model import ModelRequest, ModelResult


class ModelBackend(Protocol):
    """Execute one worker-local model request."""

    async def infer(self, request: ModelRequest) -> ModelResult:
        """Run inference and return a model-level result."""
        ...
