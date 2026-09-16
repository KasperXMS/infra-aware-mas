from infra_mas.core.artifact import ArtifactRef
from infra_mas.core.executor import ExecutorSpec
from infra_mas.core.model import ModelSpec
from infra_mas.core.resource import NetworkLink, ServiceTimeEstimate
from infra_mas.execution.executor_registry import ExecutorRegistry
from infra_mas.planner.prompts import build_blind_coordinator_instructions
from infra_mas.resources.provider import StaticResourceProvider
from infra_mas.runtime.model_registry import ModelRegistry


def _registry() -> ExecutorRegistry:
    return ExecutorRegistry(
        [
            ExecutorSpec(
                id="vlm-a",
                capability="vision",
                worker_id="worker-a",
                model_id="vlm",
                device="device-a",
                site="site-a",
            ),
            ExecutorSpec(
                id="vlm-b",
                capability="vision",
                worker_id="worker-b",
                model_id="vlm",
                device="device-b",
                site="site-b",
            ),
        ],
        {"worker-a": "http://worker-a.test", "worker-b": "http://worker-b.test"},
    )


def test_snapshot_contains_raw_facts_and_no_workflow_costs() -> None:
    provider = StaticResourceProvider(
        _registry(),
        service_times=[
            ServiceTimeEstimate(
                executor_id="vlm-a",
                model_id="vlm",
                service_ms=120,
            )
        ],
        network_links=[
            NetworkLink(
                source_site="site-a",
                target_site="site-b",
                bandwidth_mbps=100,
                rtt_ms=5,
            )
        ],
    )
    artifact = ArtifactRef(
        id="run/input.jpg",
        artifact_type="image/jpeg",
        size_bytes=1_000_000,
        locations=["worker-a"],
    )

    snapshot = provider.snapshot([artifact])
    rendered = snapshot.render_for_planner()

    assert snapshot.artifacts[0].locations == ["worker-a"]
    assert len(snapshot.model_replicas) == 2
    assert "service_ms=120" in rendered
    assert "bandwidth_mbps=100" in rendered
    assert "total workflow" not in rendered.lower()
    assert "choose" not in rendered.lower()


def test_transfer_estimate_uses_rtt_and_serialization() -> None:
    registry = _registry()
    provider = StaticResourceProvider(
        registry,
        network_links=[
            NetworkLink(
                source_site="site-a",
                target_site="site-b",
                bandwidth_mbps=100,
                rtt_ms=5,
            )
        ],
    )
    artifact = ArtifactRef(
        id="run/input.jpg",
        artifact_type="image/jpeg",
        size_bytes=1_000_000,
        locations=["worker-a"],
    )

    assert provider.estimate_input_transfer_ms([artifact], registry.get("vlm-a")) == 0
    assert provider.estimate_input_transfer_ms([artifact], registry.get("vlm-b")) == 85


def test_snapshot_is_static_context_plus_dynamic_world_state() -> None:
    provider = StaticResourceProvider(
        _registry(),
        service_times=[
            ServiceTimeEstimate(executor_id="vlm-a", model_id="vlm", service_ms=120)
        ],
    )
    snapshot = provider.snapshot(
        [
            ArtifactRef(
                id="run/input.jpg",
                artifact_type="image/jpeg",
                size_bytes=10,
                locations=["worker-a"],
            )
        ]
    )

    static = snapshot.render_static_for_planner()
    dynamic = snapshot.render_dynamic_for_planner()

    assert snapshot.render_for_planner() == f"{static}\n\n{dynamic}"
    assert "input.jpg" not in static
    assert "worker-a" not in static
    assert "input.jpg" in dynamic
    assert "worker-a" in dynamic
    assert "candidate" not in snapshot.render_for_planner().lower()
    assert "oracle" not in snapshot.render_for_planner().lower()
    assert "centralized" not in snapshot.render_for_planner().lower()
    assert "distributed_3x2" not in snapshot.render_for_planner().lower()


def test_reference_workflow_ids_are_absent_from_planner_instructions() -> None:
    models = ModelRegistry(
        [
            ModelSpec(
                model_id="vlm",
                description="Inspect images.",
                input_modalities=["image"],
                output_modalities=["text"],
                context_window=8192,
            )
        ]
    )

    prompt = build_blind_coordinator_instructions(
        models, None, "dynamic_models", "efficient"
    ).lower()

    assert "centralized" not in prompt
    assert "distributed_3x2" not in prompt
