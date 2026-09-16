"""Infrastructure resource views and providers."""

from infra_mas.resources.provider import (
    ResourceProvider,
    StaticResourceConfig,
    StaticResourceProvider,
)
from infra_mas.resources.view import InfraSnapshot

__all__ = [
    "InfraSnapshot",
    "ResourceProvider",
    "StaticResourceConfig",
    "StaticResourceProvider",
]
