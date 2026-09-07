"""Typed errors shared across execution layers."""


class InfraMasError(Exception):
    """Base class for expected infrastructure MAS failures."""


class ArtifactNotFoundError(InfraMasError):
    """Raised when a referenced artifact is unavailable."""


class WorkerUnavailableError(InfraMasError):
    """Raised when a worker cannot be reached."""


class ExecutionFailedError(InfraMasError):
    """Raised when a worker or model cannot complete an execution request."""


class InvalidModelResponseError(ExecutionFailedError):
    """Raised when a backend violates the model result contract."""
