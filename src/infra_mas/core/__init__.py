"""Shared domain models and exceptions."""

from infra_mas.core.agent import AgentSpec
from infra_mas.core.artifact import ArtifactRef
from infra_mas.core.errors import (
    ArtifactNotFoundError,
    ArtifactTransferError,
    ExecutionFailedError,
    InfraMasError,
    InvalidModelResponseError,
    WorkerUnavailableError,
)
from infra_mas.core.execution import (
    ExecutionRequest,
    ExecutionResult,
    HealthResponse,
    TransferResult,
    WorkerStatus,
)
from infra_mas.core.executor import ExecutorSpec
from infra_mas.core.model import ModelRequest, ModelResult

__all__ = [
    "AgentSpec",
    "ArtifactNotFoundError",
    "ArtifactRef",
    "ArtifactTransferError",
    "ExecutionFailedError",
    "ExecutionRequest",
    "ExecutionResult",
    "ExecutorSpec",
    "HealthResponse",
    "InfraMasError",
    "InvalidModelResponseError",
    "ModelRequest",
    "ModelResult",
    "TransferResult",
    "WorkerStatus",
    "WorkerUnavailableError",
]
