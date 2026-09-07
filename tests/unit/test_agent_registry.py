"""Unit tests for the semantic agent registry."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from infra_mas.core.agent import AgentSpec
from infra_mas.core.errors import AgentNotFoundError
from infra_mas.runtime.agent_registry import AgentRegistry


def test_registry_lookup_and_capability_filter() -> None:
    registry = AgentRegistry(
        [
            AgentSpec(name="reasoner", capability="reasoning", instructions="Reason."),
            AgentSpec(name="critic", capability="reasoning", instructions="Critique."),
        ]
    )

    assert registry.get("reasoner").instructions == "Reason."
    assert [agent.name for agent in registry.list()] == ["reasoner", "critic"]
    assert [agent.name for agent in registry.by_capability("reasoning")] == [
        "reasoner",
        "critic",
    ]


def test_registry_rejects_duplicate_names() -> None:
    duplicate = AgentSpec(name="reasoner", capability="reasoning", instructions="Reason.")

    with pytest.raises(ValueError, match="unique"):
        AgentRegistry([duplicate, duplicate])


def test_unknown_agent_raises_typed_error() -> None:
    registry = AgentRegistry([])

    with pytest.raises(AgentNotFoundError, match="missing"):
        registry.get("missing")


def test_registry_loads_yaml(tmp_path: Path) -> None:
    path = tmp_path / "agents.yaml"
    path.write_text(
        "agents:\n  reasoner:\n    capability: reasoning\n    instructions: Reason.\n",
        encoding="utf-8",
    )

    registry = AgentRegistry.from_yaml(path)

    assert registry.get("reasoner").capability == "reasoning"


def test_registry_rejects_invalid_yaml_config(tmp_path: Path) -> None:
    path = tmp_path / "agents.yaml"
    path.write_text(
        "agents:\n  reasoner:\n    capability: reasoning\n    instructions: ''\n",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        AgentRegistry.from_yaml(path)
