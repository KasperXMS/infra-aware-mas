"""Semantic agent registry."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Annotated

import yaml
from pydantic import BaseModel, ConfigDict, StringConstraints

from infra_mas.core.agent import AgentSpec
from infra_mas.core.errors import AgentNotFoundError

NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class _AgentEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    capability: NonEmptyString
    instructions: NonEmptyString


class _AgentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agents: dict[str, _AgentEntry]


class AgentRegistry:
    """Store immutable snapshots of configured semantic agent definitions."""

    def __init__(self, agents: Iterable[AgentSpec]) -> None:
        agent_list = [agent.model_copy(deep=True) for agent in agents]
        names = [agent.name for agent in agent_list]
        if len(names) != len(set(names)):
            raise ValueError("agent names must be unique")
        self._agents = {agent.name: agent for agent in agent_list}

    @classmethod
    def from_yaml(cls, path: Path) -> AgentRegistry:
        """Load and validate agent definitions from a YAML configuration file."""
        raw: object = yaml.safe_load(path.read_text(encoding="utf-8"))
        config = _AgentConfig.model_validate(raw)
        return cls(
            AgentSpec(
                name=name,
                capability=entry.capability,
                instructions=entry.instructions,
            )
            for name, entry in config.agents.items()
        )

    def get(self, name: str) -> AgentSpec:
        """Look up an agent by its unique semantic name."""
        try:
            return self._agents[name].model_copy(deep=True)
        except KeyError as error:
            raise AgentNotFoundError(f"unknown agent: {name}") from error

    def list(self) -> list[AgentSpec]:
        """Return all configured semantic agents in declaration order."""
        return [agent.model_copy(deep=True) for agent in self._agents.values()]

    def by_capability(self, capability: str) -> list[AgentSpec]:
        """Return agents that provide one semantic capability."""
        return [
            agent.model_copy(deep=True)
            for agent in self._agents.values()
            if agent.capability == capability
        ]
