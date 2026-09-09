"""Logical deployed-model registry."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Annotated

import yaml
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from infra_mas.core.errors import ModelNotFoundError
from infra_mas.core.model import ModelSpec

NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class _ModelEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: NonEmptyString
    input_modalities: Annotated[list[NonEmptyString], Field(min_length=1)]
    output_modalities: Annotated[list[NonEmptyString], Field(min_length=1)]
    context_window: Annotated[int, Field(gt=0)]


class _ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    models: Annotated[dict[str, _ModelEntry], Field(min_length=1)]


class ModelRegistry:
    """Expose logical model capabilities without physical infrastructure details."""

    def __init__(self, models: Iterable[ModelSpec]) -> None:
        model_list = [model.model_copy(deep=True) for model in models]
        model_ids = [model.model_id for model in model_list]
        if len(model_ids) != len(set(model_ids)):
            raise ValueError("model IDs must be unique")
        self._models = {model.model_id: model for model in model_list}

    @classmethod
    def from_yaml(cls, path: Path) -> ModelRegistry:
        """Load logical deployed models from YAML."""
        raw: object = yaml.safe_load(path.read_text(encoding="utf-8"))
        config = _ModelConfig.model_validate(raw)
        return cls(
            ModelSpec(
                model_id=model_id,
                description=entry.description,
                input_modalities=entry.input_modalities,
                output_modalities=entry.output_modalities,
                context_window=entry.context_window,
            )
            for model_id, entry in config.models.items()
        )

    def get(self, model_id: str) -> ModelSpec:
        """Return one logical model by ID."""
        try:
            return self._models[model_id].model_copy(deep=True)
        except KeyError as error:
            raise ModelNotFoundError(f"unknown model: {model_id}") from error

    def list(self) -> list[ModelSpec]:
        """Return deployed logical models in declaration order."""
        return [model.model_copy(deep=True) for model in self._models.values()]
