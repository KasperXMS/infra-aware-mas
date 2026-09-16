from pathlib import Path

from infra_mas.experiment import BlindExperimentConfig, LocalityAwareSchedulerConfig
from infra_mas.pilot import PilotConfig, validate_paired_planner_configs


def _experiment(visibility: str) -> BlindExperimentConfig:
    return BlindExperimentConfig.model_validate(
        {
            "planner_mode": "dynamic_models",
            "planner_harness": "efficient",
            "agents_config": None,
            "models_config": "models.yaml",
            "executors_config": "executors.yaml",
            "resources_config": "resources.yaml",
            "infrastructure_visibility": visibility,
            "scheduler": {"type": "locality_aware"},
            "planner": {"model": "qwen3.8-max"},
        }
    )


def test_paired_configs_differ_only_in_visibility() -> None:
    validate_paired_planner_configs(_experiment("none"), _experiment("snapshot"))


def test_checked_in_pilot_has_exact_six_image_worlds() -> None:
    root = Path(__file__).parents[2]
    config = PilotConfig.from_yaml(root / "configs" / "pilot" / "pilot.yaml")

    assert config.worlds["colocated"].input_workers == ["a28"] * 6
    assert config.worlds["distributed"].input_workers == [
        "a4",
        "a4",
        "a5",
        "a5",
        "a28",
        "a28",
    ]


def test_static_and_snapshot_configs_share_planner_harness_tools_and_scheduler() -> None:
    root = Path(__file__).parents[2] / "configs" / "pilot"
    static = BlindExperimentConfig.from_yaml(root / "planner-static.yaml")
    snapshot = BlindExperimentConfig.from_yaml(root / "planner-snapshot.yaml")

    assert isinstance(static.scheduler, LocalityAwareSchedulerConfig)
    assert type(static.scheduler) is type(snapshot.scheduler)
    assert static.model_dump(exclude={"infrastructure_visibility"}) == snapshot.model_dump(
        exclude={"infrastructure_visibility"}
    )
