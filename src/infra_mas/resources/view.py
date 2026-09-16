"""Planner-facing infrastructure snapshot."""

from pydantic import BaseModel, ConfigDict

from infra_mas.core.resource import (
    ArtifactPlacement,
    ModelReplicaPlacement,
    NetworkLink,
    ServiceTimeEstimate,
)


class InfraSnapshot(BaseModel):
    """Contain raw infrastructure facts without workflow recommendations or costs."""

    model_config = ConfigDict(extra="forbid")

    artifacts: list[ArtifactPlacement] = []
    model_replicas: list[ModelReplicaPlacement] = []
    service_times: list[ServiceTimeEstimate] = []
    network_links: list[NetworkLink] = []

    def render_static_for_planner(self) -> str:
        """Render world-invariant execution semantics and logical service profiles."""
        profiles: dict[str, list[float]] = {}
        for item in self.service_times:
            profiles.setdefault(item.model_id, []).append(item.service_ms)
        service_lines = [
            f"- model_id={model_id}; per_invocation_service_ms="
            f"{','.join(f'{value:g}' for value in sorted(values))}"
            for model_id, values in sorted(profiles.items())
        ] or ["- none"]
        return "\n".join(
            [
                "Static execution context:",
                "Logical-model service profiles:",
                *service_lines,
                "Generic execution semantics:",
                "- The Planner selects logical model invocations and artifact inputs; it does "
                "not select physical executors.",
                "- A logical model may have multiple physical replicas.",
                "- The scheduler independently binds every invocation using the same "
                "deterministic policy in all visibility modes.",
                "- Every input artifact absent from the selected Worker is transferred before "
                "model service begins.",
                "- Outputs remain on the Worker that produced them until another operation "
                "requires a transfer.",
                "- A link transfer estimate is RTT plus 8 * bytes / bandwidth; actual transfer "
                "and service times are measured in the execution trace.",
                "- Each service-time estimate applies to exactly one model invocation.",
            ]
        )

    def render_dynamic_for_planner(self) -> str:
        """Render only the current world's placement, replica, and link facts."""
        artifact_lines = [
            f"- artifact_id={item.artifact_id}; size_bytes={item.size_bytes}; "
            f"locations={','.join(item.locations)}"
            for item in sorted(self.artifacts, key=lambda value: value.artifact_id)
        ] or ["- none"]
        replica_lines = [
            f"- model_id={item.model_id}; executor_id={item.executor_id}; "
            f"worker_id={item.worker_id}; site={item.site}"
            for item in sorted(self.model_replicas, key=lambda value: value.executor_id)
        ] or ["- none"]
        service_lines = [
            f"- executor_id={item.executor_id}; model_id={item.model_id}; "
            f"service_ms={item.service_ms:g}; source={item.source}"
            for item in sorted(self.service_times, key=lambda value: value.executor_id)
        ] or ["- none"]
        link_lines = [
            f"- source_site={item.source_site}; target_site={item.target_site}; "
            f"bandwidth_mbps={item.bandwidth_mbps:g}; rtt_ms={item.rtt_ms:g}; "
            f"bidirectional={str(item.bidirectional).lower()}"
            for item in sorted(
                self.network_links,
                key=lambda value: (value.source_site, value.target_site),
            )
        ] or ["- none"]
        return "\n".join(
            [
                "Dynamic infrastructure snapshot (raw facts):",
                "Artifacts:",
                *artifact_lines,
                "Logical-model replicas:",
                *replica_lines,
                "Current per-replica service-time estimates:",
                *service_lines,
                "Network links:",
                *link_lines,
            ]
        )

    def render_for_planner(self) -> str:
        """Render static semantics followed by the current dynamic world state."""
        return f"{self.render_static_for_planner()}\n\n{self.render_dynamic_for_planner()}"
