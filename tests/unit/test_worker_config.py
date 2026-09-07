"""Tests for Worker process configuration."""

from pathlib import Path

import pytest

from infra_mas.worker.config import WorkerConfig, build_worker_service


def test_builds_mock_worker_with_relative_artifact_root(tmp_path: Path) -> None:
    config = WorkerConfig.model_validate(
        {
            "worker_id": "worker-a",
            "artifact_root": "artifacts",
            "executors": [
                {
                    "id": "worker-a-reasoner",
                    "capability": "reasoning",
                    "backend": {"type": "mock", "output": "answer"},
                }
            ],
        }
    )

    service = build_worker_service(config, tmp_path)

    assert service.status().worker_id == "worker-a"
    assert service.status().executors == ["worker-a-reasoner"]
    assert (tmp_path / "artifacts").is_dir()


def test_missing_backend_api_key_fails_at_startup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MISSING_MODEL_KEY", raising=False)
    config = WorkerConfig.model_validate(
        {
            "worker_id": "worker-a",
            "executors": [
                {
                    "id": "worker-a-reasoner",
                    "capability": "reasoning",
                    "backend": {
                        "type": "openai_compatible",
                        "base_url": "http://model.test/v1",
                        "model": "test-model",
                        "api_key_env": "MISSING_MODEL_KEY",
                    },
                }
            ],
        }
    )

    with pytest.raises(ValueError, match="MISSING_MODEL_KEY"):
        build_worker_service(config, tmp_path)
