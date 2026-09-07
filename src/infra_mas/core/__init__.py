"""Shared domain models and exceptions."""

from infra_mas.core.agent import AgentSpec
from infra_mas.core.artifact import ArtifactRef
from infra_mas.core.errors import (
    ArtifactNotFoundError,
    ExecutionFailedError,
    InfraMasError,
    InvalidModelResponseError,
    WorkerUnavailableError,
)
from infra_mas.core.execution import (
    ExecutionRequest,
    ExecutionResult,
    HealthResponse,
    WorkerStatus,
)
from infra_mas.core.executor import ExecutorSpec
from infra_mas.core.model import ModelRequest, ModelResult

__all__ = [
    "AgentSpec",
    "ArtifactNotFoundError",
    "ArtifactRef",
    "ExecutionFailedError",
    "ExecutionRequest",
    "ExecutionResult",
    "ExecutorSpec",
    "HealthResponse",
    "InfraMasError",
    "InvalidModelResponseError",
    "ModelRequest",
    "ModelResult",
    "WorkerStatus",
    "WorkerUnavailableError",
]
