"""Static infrastructure snapshot provider and transfer estimator."""

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Protocol

import yaml
from pydantic import BaseModel, ConfigDict

from infra_mas.core.artifact import ArtifactRef
from infra_mas.core.executor import ExecutorSpec
from infra_mas.core.resource import (
    ArtifactPlacement,
    ModelReplicaPlacement,
    NetworkLink,
    ServiceTimeEstimate,
)
from infra_mas.execution.executor_registry import ExecutorRegistry
from infra_mas.resources.view import InfraSnapshot


class StaticResourceConfig(BaseModel):
    """Validate a static infrastructure facts file."""

    model_config = ConfigDict(extra="forbid")

    service_times: list[ServiceTimeEstimate] = []
    network_links: list[NetworkLink] = []

    @classmethod
    def from_yaml(cls, path: Path) -> "StaticResourceConfig":
        """Load static infrastructure facts from YAML."""
        raw: object = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls.model_validate(raw or {})


class ResourceProvider(Protocol):
    """Provide raw infrastructure facts and point-to-point transfer estimates."""

    def snapshot(self, artifacts: Sequence[ArtifactRef]) -> InfraSnapshot:
        """Return one immutable view using the supplied current artifact locations."""
        ...

    def estimate_input_transfer_ms(
        self,
        artifacts: Sequence[ArtifactRef],
        executor: ExecutorSpec,
    ) -> float:
        """Estimate serialized input transfer latency for one candidate executor."""
        ...


class StaticResourceProvider:
    """Combine static replica/network facts with current artifact placement."""

    def __init__(
        self,
        registry: ExecutorRegistry,
        *,
        service_times: Iterable[ServiceTimeEstimate] = (),
        network_links: Iterable[NetworkLink] = (),
    ) -> None:
        self._registry = registry
        self._service_times = sorted(
            (item.model_copy(deep=True) for item in service_times),
            key=lambda item: item.executor_id,
        )
        self._network_links = sorted(
            (item.model_copy(deep=True) for item in network_links),
            key=lambda item: (item.source_site, item.target_site),
        )
        known_executors = {executor.id: executor for executor in registry.list()}
        for estimate in self._service_times:
            executor = known_executors.get(estimate.executor_id)
            if executor is None:
                raise ValueError(
                    f"service estimate references unknown executor {estimate.executor_id!r}"
                )
            if executor.model_id != estimate.model_id:
                raise ValueError(
                    f"service estimate model mismatch for executor {estimate.executor_id!r}"
                )
        self._worker_sites = registry.worker_sites()

    @classmethod
    def from_yaml(cls, registry: ExecutorRegistry, path: Path) -> "StaticResourceProvider":
        """Build a provider from one validated facts file."""
        config = StaticResourceConfig.from_yaml(path)
        return cls(
            registry,
            service_times=config.service_times,
            network_links=config.network_links,
        )

    def snapshot(self, artifacts: Sequence[ArtifactRef]) -> InfraSnapshot:
        """Return sorted raw facts without derived workflow-level estimates."""
        return InfraSnapshot(
            artifacts=[
                ArtifactPlacement(
                    artifact_id=artifact.id,
                    size_bytes=artifact.size_bytes,
                    locations=sorted(artifact.locations),
                )
                for artifact in sorted(artifacts, key=lambda item: item.id)
            ],
            model_replicas=[
                ModelReplicaPlacement(
                    model_id=executor.model_id,
                    executor_id=executor.id,
                    worker_id=executor.worker_id,
                    site=executor.site,
                )
                for executor in sorted(self._registry.list(), key=lambda item: item.id)
            ],
            service_times=[item.model_copy(deep=True) for item in self._service_times],
            network_links=[item.model_copy(deep=True) for item in self._network_links],
        )

    def estimate_input_transfer_ms(
        self,
        artifacts: Sequence[ArtifactRef],
        executor: ExecutorSpec,
    ) -> float:
        """Sum the cheapest source-to-target estimate for every missing artifact."""
        total_ms = 0.0
        for artifact in artifacts:
            if executor.worker_id in artifact.locations:
                continue
            choices = [
                self._estimate_one_transfer_ms(
                    source_worker_id,
                    executor.worker_id,
                    artifact.size_bytes,
                )
                for source_worker_id in artifact.locations
                if source_worker_id in self._worker_sites
            ]
            if not choices:
                return float("inf")
            total_ms += min(choices)
        return total_ms

    def _estimate_one_transfer_ms(
        self,
        source_worker_id: str,
        target_worker_id: str,
        size_bytes: int,
    ) -> float:
        if source_worker_id == target_worker_id:
            return 0.0
        source_site = self._worker_sites[source_worker_id]
        target_site = self._worker_sites[target_worker_id]
        if source_site == target_site:
            return 0.0
        link = next(
            (
                item
                for item in self._network_links
                if (
                    item.source_site == source_site
                    and item.target_site == target_site
                )
                or (
                    item.bidirectional
                    and item.source_site == target_site
                    and item.target_site == source_site
                )
            ),
            None,
        )
        if link is None:
            return float("inf")
        serialization_ms = size_bytes * 8.0 / (link.bandwidth_mbps * 1_000.0)
        return link.rtt_ms + serialization_ms
