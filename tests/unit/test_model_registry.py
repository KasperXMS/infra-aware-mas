"""Unit tests for the logical deployed-model registry."""

from pathlib import Path

import pytest

from infra_mas.core.errors import ModelNotFoundError
from infra_mas.core.model import ModelSpec
from infra_mas.runtime.model_registry import ModelRegistry


def model(model_id: str = "test-llm") -> ModelSpec:
    return ModelSpec(
        model_id=model_id,
        description="General test model.",
        input_modalities=["text"],
        output_modalities=["text"],
        context_window=8192,
    )


def test_registry_lookup_and_listing() -> None:
    registry = ModelRegistry([model("small"), model("large")])

    assert registry.get("small").description == "General test model."
    assert [item.model_id for item in registry.list()] == ["small", "large"]


def test_registry_rejects_duplicate_ids() -> None:
    with pytest.raises(ValueError, match="unique"):
        ModelRegistry([model(), model()])


def test_unknown_model_raises_typed_error() -> None:
    with pytest.raises(ModelNotFoundError, match="missing"):
        ModelRegistry([]).get("missing")


def test_registry_loads_yaml(tmp_path: Path) -> None:
    path = tmp_path / "models.yaml"
    path.write_text(
        """models:
  test-llm:
    description: General test model.
    input_modalities: [text]
    output_modalities: [text]
    context_window: 8192
""",
        encoding="utf-8",
    )

    registry = ModelRegistry.from_yaml(path)

    assert registry.get("test-llm") == model()
