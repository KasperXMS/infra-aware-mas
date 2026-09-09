"""Typed errors shared across execution layers."""


class InfraMasError(Exception):
    """Base class for expected infrastructure MAS failures."""


class AgentNotFoundError(InfraMasError):
    """Raised when a semantic agent name is not registered."""


class ModelNotFoundError(InfraMasError):
    """Raised when a logical model ID is not registered."""


class ExecutorNotFoundError(InfraMasError):
    """Raised when a physical executor ID is not registered."""


class NoExecutorAvailableError(InfraMasError):
    """Raised when scheduling finds no compatible physical executor."""


class ArtifactNotFoundError(InfraMasError):
    """Raised when a referenced artifact is unavailable."""


class ArtifactTransferError(InfraMasError):
    """Raised when an artifact cannot be copied between workers."""


class WorkerUnavailableError(InfraMasError):
    """Raised when a worker cannot be reached."""


class ExecutionFailedError(InfraMasError):
    """Raised when a worker or model cannot complete an execution request."""


class InvalidModelResponseError(ExecutionFailedError):
    """Raised when a backend violates the model result contract."""
