"""Static physical executor and worker endpoint registry."""

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Annotated

import httpx
import yaml
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, StringConstraints

from infra_mas.core.errors import ExecutorNotFoundError, WorkerUnavailableError
from infra_mas.core.executor import ExecutorSpec

NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class _WorkerEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    endpoint: NonEmptyString
    site: NonEmptyString


class _ExecutorEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    worker_id: NonEmptyString
    capability: NonEmptyString
    model_id: NonEmptyString = Field(validation_alias=AliasChoices("model_id", "model"))
    device: NonEmptyString
    site: NonEmptyString


class _ExecutorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    workers: dict[str, _WorkerEntry]
    executors: dict[str, _ExecutorEntry]


class ExecutorRegistry:
    """Store static executor definitions and resolve their worker endpoints."""

    def __init__(
        self,
        executors: Iterable[ExecutorSpec],
        worker_endpoints: Mapping[str, str],
        worker_sites: Mapping[str, str] | None = None,
    ) -> None:
        executor_list = [executor.model_copy(deep=True) for executor in executors]
        executor_ids = [executor.id for executor in executor_list]
        if len(executor_ids) != len(set(executor_ids)):
            raise ValueError("executor IDs must be unique")

        self._worker_endpoints = {
            worker_id: self._validate_endpoint(worker_id, endpoint)
            for worker_id, endpoint in worker_endpoints.items()
        }
        self._worker_sites = dict(worker_sites or {})
        missing_workers = sorted(
            {executor.worker_id for executor in executor_list} - self._worker_endpoints.keys()
        )
        if missing_workers:
            raise ValueError(f"executors reference unknown workers: {missing_workers}")
        unknown_site_workers = sorted(self._worker_sites.keys() - self._worker_endpoints.keys())
        if unknown_site_workers:
            raise ValueError(f"worker sites reference unknown workers: {unknown_site_workers}")
        for executor in executor_list:
            configured_site = self._worker_sites.setdefault(executor.worker_id, executor.site)
            if configured_site != executor.site:
                raise ValueError(
                    f"executor {executor.id!r} site does not match its Worker site"
                )
        self._executors = {executor.id: executor for executor in executor_list}

    @classmethod
    def from_yaml(cls, path: Path) -> "ExecutorRegistry":
        """Load static workers and executors from a YAML configuration file."""
        raw: object = yaml.safe_load(path.read_text(encoding="utf-8"))
        config = _ExecutorConfig.model_validate(raw)
        executors: list[ExecutorSpec] = []
        for executor_id, entry in config.executors.items():
            worker = config.workers.get(entry.worker_id)
            if worker is None:
                raise ValueError(
                    f"executor {executor_id!r} references unknown worker {entry.worker_id!r}"
                )
            if worker.site != entry.site:
                raise ValueError(
                    f"executor {executor_id!r} site does not match worker {entry.worker_id!r}"
                )
            executors.append(
                ExecutorSpec(
                    id=executor_id,
                    capability=entry.capability,
                    worker_id=entry.worker_id,
                    model_id=entry.model_id,
                    device=entry.device,
                    site=entry.site,
                )
            )
        return cls(
            executors,
            {worker_id: worker.endpoint for worker_id, worker in config.workers.items()},
            {worker_id: worker.site for worker_id, worker in config.workers.items()},
        )

    def filtered(self, executor_ids: Iterable[str]) -> "ExecutorRegistry":
        """Return an eligibility-filtered view while preserving all Worker locations."""
        selected_ids = set(executor_ids)
        unknown = sorted(selected_ids - self._executors.keys())
        if unknown:
            raise ValueError(f"unknown eligible executors: {unknown}")
        return ExecutorRegistry(
            (executor for key, executor in self._executors.items() if key in selected_ids),
            self._worker_endpoints,
            self._worker_sites,
        )

    def get(self, executor_id: str) -> ExecutorSpec:
        """Look up a physical executor by ID."""
        try:
            return self._executors[executor_id].model_copy(deep=True)
        except KeyError as error:
            raise ExecutorNotFoundError(f"unknown executor: {executor_id}") from error

    def candidates(self, capability: str) -> list[ExecutorSpec]:
        """Return all executors compatible with a semantic capability."""
        return [
            executor.model_copy(deep=True)
            for executor in self._executors.values()
            if executor.capability == capability
        ]

    def model_candidates(self, model_id: str) -> list[ExecutorSpec]:
        """Return every physical replica serving one logical model."""
        return [
            executor.model_copy(deep=True)
            for executor in self._executors.values()
            if executor.model_id == model_id
        ]

    def list(self) -> list[ExecutorSpec]:
        """Return all configured physical executors in declaration order."""
        return [executor.model_copy(deep=True) for executor in self._executors.values()]

    def worker_endpoints(self) -> dict[str, str]:
        """Return a copy of all configured Worker endpoints."""
        return dict(self._worker_endpoints)

    def worker_sites(self) -> dict[str, str]:
        """Return the unique configured site for every Worker."""
        return dict(self._worker_sites)

    def worker_endpoint(self, worker_id: str) -> str:
        """Resolve a worker ID to its static HTTP endpoint."""
        try:
            return self._worker_endpoints[worker_id]
        except KeyError as error:
            raise WorkerUnavailableError(f"unknown worker: {worker_id}") from error

    def executor_endpoint(self, executor_id: str) -> str:
        """Resolve an executor ID to its hosting worker endpoint."""
        return self.worker_endpoint(self.get(executor_id).worker_id)

    @staticmethod
    def _validate_endpoint(worker_id: str, endpoint: str) -> str:
        if not worker_id.strip():
            raise ValueError("worker IDs must not be empty")
        url = httpx.URL(endpoint)
        if url.scheme not in {"http", "https"} or not url.host:
            raise ValueError(f"worker {worker_id!r} has an invalid HTTP endpoint")
        return str(url)
