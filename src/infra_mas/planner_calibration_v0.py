"""Configuration-only checks for the open-ended calibration_v0 Planner experiment.

This module does not construct or run a Planner.  It validates the controlled 2x2
configuration and offers an explicitly Jetson-only Worker preflight.
"""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from infra_mas.execution.executor_registry import ExecutorRegistry
from infra_mas.experiment import (
    BlindExperimentConfig,
    close_worker_clients,
    create_worker_clients,
    preflight_workers,
    resolve_config_path,
)
from infra_mas.resources.provider import StaticResourceConfig

NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
PlannerWorld = Literal["H1_distributed_constrained", "H2_distributed_favorable"]
PlannerArm = Literal["blind", "aware"]

_EXPECTED_SITES = ("A4", "A5", "A28")
_FORBIDDEN_RUNTIME_TEXT = (
    "centralized_raw",
    "local_reduction",
    "bandwidth_star",
    "bw*",
    "reference winner",
    "reference workflow",
    "winner",
    "gold",
    "infra-sensitive",
    "break-even",
    "break_even",
    "correct_option",
)


class PlannerVideoArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artifact_id: NonEmptyString
    source_ref: NonEmptyString
    site_id: Literal["A4", "A5", "A28"]
    chunk_index: int = Field(ge=0, le=2)
    duration_s: float = Field(gt=0.0)


class PlannerTask795(BaseModel):
    """Gold-free, fixed-sampling inputs shared by all four Planner cells."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["calibration-v0-planner-task-v1"] = (
        "calibration-v0-planner-task-v1"
    )
    task_id: Literal["video_mme:795"]
    instruction: NonEmptyString
    evaluator_id: Literal["video_mme_multiple_choice"]
    artifact_policy: Literal["original_fixed_time_chunks"] = "original_fixed_time_chunks"
    source_chunk_count: Literal[3] = 3
    frames_per_chunk: Literal[12] = 12
    frame_width: Literal[320] = 320
    artifacts: Annotated[list[PlannerVideoArtifact], Field(min_length=3, max_length=3)]

    @model_validator(mode="after")
    def validate_fixed_sampling_and_placement(self) -> PlannerTask795:
        ids = [item.artifact_id for item in self.artifacts]
        if len(ids) != len(set(ids)):
            raise ValueError("Planner input artifact IDs must be unique")
        if {item.chunk_index for item in self.artifacts} != {0, 1, 2}:
            raise ValueError("Planner inputs must contain exactly the three original chunks")
        sites = {0: "A4", 1: "A5", 2: "A28"}
        invalid = [
            item.artifact_id
            for item in self.artifacts
            if item.site_id != sites[item.chunk_index]
        ]
        if invalid:
            raise ValueError(
                "chunk placement must be chunk 0/A4, chunk 1/A5, chunk 2/A28: "
                f"{invalid}"
            )
        invalid_paths = [
            item.source_ref
            for item in self.artifacts
            if PurePosixPath(item.source_ref)
            != PurePosixPath(
                f"/home/edge/xiaoming/calibration_v0/artifacts/795/"
                f"chunk-{item.chunk_index}.mp4"
            )
        ]
        if invalid_paths:
            raise ValueError(
                "Planner chunks must use the existing Orin-local Task 795 paths: "
                f"{invalid_paths}"
            )
        return self

    @classmethod
    def from_yaml(cls, path: Path) -> PlannerTask795:
        raw: object = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls.model_validate(raw)


class PlannerNetworkCondition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bandwidth_mbps: float = Field(gt=0.0)
    added_rtt_ms: float = Field(ge=0.0)
    observed_baseline_rtt_ms: float = Field(default=33.0, ge=0.0)
    effective_rtt_ms: float = Field(ge=0.0)

    @model_validator(mode="after")
    def validate_effective_rtt(self) -> PlannerNetworkCondition:
        if abs(
            self.effective_rtt_ms
            - (self.observed_baseline_rtt_ms + self.added_rtt_ms)
        ) > 1e-6:
            raise ValueError("effective_rtt_ms must equal baseline plus added RTT")
        return self


class PlannerExperimentCell(BaseModel):
    """One open-ended Planner cell; references contain no workflow or evaluator answer."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["calibration-v0-planner-cell-v1"] = (
        "calibration-v0-planner-cell-v1"
    )
    experiment_id: NonEmptyString
    task_config: Path
    experiment_config: Path
    world_id: PlannerWorld
    arm: PlannerArm
    infrastructure_visibility: Literal["none", "snapshot"]
    network: PlannerNetworkCondition
    input_workers: tuple[Literal["a4"], Literal["a5"], Literal["a28"]] = (
        "a4",
        "a5",
        "a28",
    )
    eligible_executor_ids: tuple[
        Literal["a4-vlm"],
        Literal["a5-vlm"],
        Literal["a28-vlm"],
        Literal["strong-4090-vlm"],
    ] = ("a4-vlm", "a5-vlm", "a28-vlm", "strong-4090-vlm")
    planner_tools: tuple[Literal["spawn_agent"], Literal["inspect_artifact"]] = (
        "spawn_agent",
        "inspect_artifact",
    )

    @model_validator(mode="after")
    def validate_arm(self) -> PlannerExperimentCell:
        expected = "none" if self.arm == "blind" else "snapshot"
        if self.infrastructure_visibility != expected:
            raise ValueError(f"{self.arm} arm requires infrastructure_visibility: {expected}")
        return self

    @classmethod
    def from_yaml(cls, path: Path) -> PlannerExperimentCell:
        raw: object = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls.model_validate(raw)


class PlannerPreflightReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["calibration-v0-planner-preflight-v1"] = (
        "calibration-v0-planner-preflight-v1"
    )
    config_validation: Literal["passed"] = "passed"
    artifact_placement_validation: Literal["passed"] = "passed"
    jetson_workers: dict[str, list[str]]
    strong_4090: dict[str, object]
    capability_validation: dict[str, object]


def _load_cell(
    path: Path,
) -> tuple[
    PlannerExperimentCell,
    PlannerTask795,
    BlindExperimentConfig,
    StaticResourceConfig,
]:
    path = path.resolve()
    cell = PlannerExperimentCell.from_yaml(path)
    task_path = resolve_config_path(cell.task_config, path.parent)
    experiment_path = resolve_config_path(cell.experiment_config, path.parent)
    task = PlannerTask795.from_yaml(task_path)
    experiment = BlindExperimentConfig.from_yaml(experiment_path)
    if experiment.resources_config is None:
        raise ValueError("open-ended calibration requires resources_config")
    resources = StaticResourceConfig.from_yaml(
        resolve_config_path(experiment.resources_config, experiment_path.parent)
    )
    if experiment.planner_mode != "dynamic_models":
        raise ValueError("open-ended calibration requires planner_mode: dynamic_models")
    if experiment.planner_harness != "efficient":
        raise ValueError("open-ended calibration requires planner_harness: efficient")
    if experiment.max_turns <= 1:
        raise ValueError("open-ended calibration requires more than one Planner turn")
    if experiment.infrastructure_visibility != cell.infrastructure_visibility:
        raise ValueError("cell visibility must match its executable experiment config")
    return cell, task, experiment, resources


def validate_planner_experiment_cells(paths: list[Path]) -> list[PlannerExperimentCell]:
    """Validate the four cells and prove only visibility/network dimensions vary."""
    if len(paths) != 4:
        raise ValueError("exactly four Planner experiment cell configs are required")
    loaded = [_load_cell(path) for path in paths]
    cells = [item[0] for item in loaded]
    expected_cells = {
        (world, arm)
        for world in ("H1_distributed_constrained", "H2_distributed_favorable")
        for arm in ("blind", "aware")
    }
    actual_cells = {(cell.world_id, cell.arm) for cell in cells}
    if actual_cells != expected_cells:
        raise ValueError("configs must form the complete H1/H2 x blind/aware grid")

    task_payloads = [item[1].model_dump(mode="json") for item in loaded]
    if any(payload != task_payloads[0] for payload in task_payloads[1:]):
        raise ValueError("all Planner cells must use the identical task and sampled artifacts")
    for world in ("H1_distributed_constrained", "H2_distributed_favorable"):
        pair_indexes = [index for index, cell in enumerate(cells) if cell.world_id == world]
        blind_index = next(index for index in pair_indexes if cells[index].arm == "blind")
        aware_index = next(index for index in pair_indexes if cells[index].arm == "aware")
        ignored = {
            "experiment_id",
            "experiment_config",
            "arm",
            "infrastructure_visibility",
        }
        if cells[blind_index].model_dump(exclude=ignored) != cells[
            aware_index
        ].model_dump(exclude=ignored):
            raise ValueError("blind/aware cells may differ only in Planner visibility")
        blind_experiment = loaded[blind_index][2]
        aware_experiment = loaded[aware_index][2]
        if blind_experiment.model_dump(exclude={"infrastructure_visibility"}) != (
            aware_experiment.model_dump(exclude={"infrastructure_visibility"})
        ):
            raise ValueError(
                "blind/aware executable configs may differ only in Planner visibility"
            )

    h1 = next(
        cell
        for cell in cells
        if cell.world_id == "H1_distributed_constrained" and cell.arm == "blind"
    )
    h2 = next(
        cell
        for cell in cells
        if cell.world_id == "H2_distributed_favorable" and cell.arm == "blind"
    )
    if not (
        h1.network.bandwidth_mbps < h2.network.bandwidth_mbps
        and h1.network.added_rtt_ms > h2.network.added_rtt_ms
    ):
        raise ValueError("H1 must have lower bandwidth and higher added RTT than H2")
    ignored_world = {"experiment_id", "experiment_config", "world_id", "network"}
    for arm in ("blind", "aware"):
        world_1_index = next(
            index
            for index, cell in enumerate(cells)
            if cell.world_id == "H1_distributed_constrained" and cell.arm == arm
        )
        world_2_index = next(
            index
            for index, cell in enumerate(cells)
            if cell.world_id == "H2_distributed_favorable" and cell.arm == arm
        )
        world_1 = cells[world_1_index]
        world_2 = cells[world_2_index]
        if world_1.model_dump(exclude=ignored_world) != world_2.model_dump(
            exclude=ignored_world
        ):
            raise ValueError("H1/H2 cells may differ only in world/network conditions")
        experiment_1 = loaded[world_1_index][2]
        experiment_2 = loaded[world_2_index][2]
        if experiment_1.model_dump(exclude={"resources_config"}) != (
            experiment_2.model_dump(exclude={"resources_config"})
        ):
            raise ValueError(
                "H1/H2 executable configs may differ only in resources_config"
            )

    service_payloads = [
        [item.model_dump(mode="json") for item in loaded_item[3].service_times]
        for loaded_item in loaded
    ]
    if any(payload != service_payloads[0] for payload in service_payloads[1:]):
        raise ValueError("model/device service assumptions must be identical in all cells")
    for cell, _, _, resources in loaded:
        if len(resources.network_links) != 6:
            raise ValueError("each Planner world requires the complete six-link topology")
        if any(
            link.bandwidth_mbps != cell.network.bandwidth_mbps
            or link.rtt_ms != cell.network.effective_rtt_ms
            for link in resources.network_links
        ):
            raise ValueError("resource links must match the cell network condition")

    serialized = json.dumps(
        {
            "cells": [cell.model_dump(mode="json") for cell in cells],
            "task": task_payloads[0],
            "experiments": [item[2].model_dump(mode="json") for item in loaded],
            "resources": [item[3].model_dump(mode="json") for item in loaded],
        },
        ensure_ascii=False,
    ).casefold()
    leaked = [token for token in _FORBIDDEN_RUNTIME_TEXT if token in serialized]
    if leaked:
        raise ValueError(f"Planner configs contain forbidden calibration leakage: {leaked}")
    return cells


async def preflight_planner_jetsons(
    paths: list[Path],
    *,
    timeout_seconds: float = 30.0,
) -> PlannerPreflightReport:
    """Check A4/A5/A28 only; deliberately never construct a client for the 4090."""
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    validate_planner_experiment_cells(paths)
    first = paths[0].resolve()
    cell = PlannerExperimentCell.from_yaml(first)
    experiment_path = resolve_config_path(cell.experiment_config, first.parent)
    experiment = BlindExperimentConfig.from_yaml(experiment_path)
    registry_path = resolve_config_path(experiment.executors_config, experiment_path.parent)
    full_registry = ExecutorRegistry.from_yaml(registry_path)
    jetson_executors = [
        executor for executor in full_registry.list() if executor.site in _EXPECTED_SITES
    ]
    jetson_worker_ids = {executor.worker_id for executor in jetson_executors}
    jetson_registry = ExecutorRegistry(
        jetson_executors,
        {
            worker_id: endpoint
            for worker_id, endpoint in full_registry.worker_endpoints().items()
            if worker_id in jetson_worker_ids
        },
        {
            worker_id: site
            for worker_id, site in full_registry.worker_sites().items()
            if worker_id in jetson_worker_ids
        },
    )
    clients = create_worker_clients(jetson_registry, timeout_seconds)
    try:
        checked = await preflight_workers(jetson_registry, clients)
    finally:
        await close_worker_clients(clients)
    return PlannerPreflightReport(
        jetson_workers=checked.workers,
        strong_4090={
            "worker_id": "strong-4090",
            "site_id": "4090",
            "status": "expected_unavailable",
            "checked": False,
            "reason": "Task 6 preflight is intentionally Jetson-only",
        },
        capability_validation={
            "status": "execution_blocked",
            "raw_artifact_type": "video/mp4",
            "planner_tools": ["spawn_agent", "inspect_artifact"],
            "missing_planner_operator": "sample_frames",
            "configured_model_input_modalities": ["text", "image"],
            "direct_video_supported": False,
            "artifact_binding": "worker_local_paths",
            "worker_local_source_binding_supported": False,
            "reason": (
                "The unchanged open-ended Planner cannot invoke sample_frames, and the "
                "deployed Ollama VLM path does not accept raw MP4 input. The current benchmark "
                "bridge also materializes controller-local source paths rather than binding "
                "pre-existing Worker-local files. Config and placement validation passed, but "
                "execution must remain blocked without changing the Planner/runtime contract."
            ),
        },
    )
