import pytest

from infra_mas.operators import (
    CODE_OPERATOR_REGISTRY,
    GENERAL_OPERATOR_REGISTRY,
    ExternalEvaluatorSpec,
    InitialArtifactSpec,
    ObservationSpec,
    RuntimeVerifierSpec,
    TaskInteractionSpec,
)


def _code_task() -> TaskInteractionSpec:
    return TaskInteractionSpec(
        task_id="code",
        objective="fix",
        initial_artifacts=[
            InitialArtifactSpec(
                artifact_id="repository", kind="git_repository", source_ref="repo://x"
            )
        ],
        operators=["search_code", "invoke_model", "submit_patch"],
        observations=[
            ObservationSpec(
                observation_id="result",
                produced_by=["search_code", "invoke_model", "submit_patch"],
                description="result",
            )
        ],
        runtime_verifier=RuntimeVerifierSpec(level="partial", signals=["test"]),
        external_evaluator=ExternalEvaluatorSpec(evaluator_id="swebench_official"),
    )


def test_runtime_registry_rejects_unbound_task_before_execution() -> None:
    task = _code_task()
    CODE_OPERATOR_REGISTRY.validate_task(task)
    with pytest.raises(ValueError, match="not_realizable"):
        GENERAL_OPERATOR_REGISTRY.validate_task(task)


def test_verifier_levels_are_explicit() -> None:
    assert _code_task().runtime_verifier.level == "partial"
    with pytest.raises(ValueError):
        RuntimeVerifierSpec(level="hidden")  # type: ignore[arg-type]
