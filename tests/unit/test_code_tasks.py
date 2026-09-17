import json
import subprocess
from pathlib import Path

import httpx
import pytest

from infra_mas.code_tasks.client import CodeWorkerClient
from infra_mas.code_tasks.context import CodePlannerContext
from infra_mas.code_tasks.coordinator import build_code_planner_instructions
from infra_mas.code_tasks.experiment import CodeBenchmarkConfig, load_admission_export
from infra_mas.code_tasks.models import CodeExecutor, RepositoryWorld
from infra_mas.code_tasks.scheduler import RepositoryLocalityScheduler
from infra_mas.code_tasks.server import (
    CodeWorkerConfig,
    CodeWorkspaceService,
    create_code_worker_app,
)
from infra_mas.tracing.recorder import TraceRecorder


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=repo, check=True, text=True, capture_output=True
    )
    return completed.stdout.strip()


def _repository(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "tests@example.com")
    _git(repo, "config", "user.name", "Tests")
    (repo / "sample.py").write_text("def value():\n    return 1\n", encoding="utf-8")
    _git(repo, "add", "sample.py")
    _git(repo, "commit", "-m", "base")
    return repo, _git(repo, "rev-parse", "HEAD")


async def test_code_worker_supports_search_read_patch_and_submit(tmp_path: Path) -> None:
    repo, base_commit = _repository(tmp_path)
    service = CodeWorkspaceService(
        CodeWorkerConfig(
            worker_id="edge-code",
            site="edge",
            port=9011,
            repository_root=repo,
            base_commit=base_commit,
            targeted_test_prefix=["python", "-m", "pytest", "-q"],
            full_test_command=["python", "-m", "pytest", "-q"],
        )
    )
    app = create_code_worker_app(service)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://worker.test"
    ) as http_client:
        client = CodeWorkerClient("http://worker.test", client=http_client)
        reset = await client.reset("run")
        assert reset.clean
        searched, *_ = await client.invoke(
            "search_code", {"query": "return 1", "path": ".", "max_results": 50}
        )
        assert searched.success
        assert "sample.py:2" in searched.output
        literal, *_ = await client.invoke(
            "search_code", {"query": "value(", "path": ".", "max_results": 50}
        )
        assert literal.success
        assert "sample.py:1" in literal.output
        no_match, *_ = await client.invoke(
            "search_code", {"query": "not present", "path": ".", "max_results": 50}
        )
        assert no_match.success
        assert no_match.output == "No matches found."
        read, *_ = await client.invoke(
            "read_file", {"path": "sample.py", "start_line": 1, "end_line": 3}
        )
        assert "2:     return 1" in read.output
        bounded, *_ = await client.invoke(
            "read_file", {"path": "sample.py", "start_line": 1, "end_line": 999}
        )
        assert bounded.success
        assert bounded.truncated
        edited, *_ = await client.invoke(
            "edit_file",
            {"path": "sample.py", "old_text": "return 1", "new_text": "return 2"},
        )
        assert edited.success
        reset = await client.reset("run-after-edit")
        assert reset.clean
        patch = """diff --git a/sample.py b/sample.py
--- a/sample.py
+++ b/sample.py
@@ -1,2 +1,2 @@
 def value():
-    return 1
+    return 2
"""
        applied, *_ = await client.invoke("apply_patch", {"patch": patch})
        assert applied.success
        submitted, *_ = await client.invoke("submit_patch", {})
        assert submitted.success
        assert submitted.patch is not None
        assert "+    return 2" in submitted.patch


def test_repository_scheduler_is_world_sensitive_but_policy_identical() -> None:
    scheduler = RepositoryLocalityScheduler(
        [
            CodeExecutor(executor_id="edge-code", site="edge", endpoint="http://edge"),
            CodeExecutor(executor_id="cloud-code", site="cloud", endpoint="http://cloud"),
        ]
    )
    edge = RepositoryWorld(
        world_id="H1",
        repository_site="edge",
        repository_artifact_id="repo://example",
        repository_size_bytes=10,
        bandwidth_mbps=1,
        rtt_ms=50,
    )
    cloud = edge.model_copy(update={"world_id": "H2", "repository_site": "cloud"})
    assert scheduler.select(edge).executor_id == "edge-code"
    assert scheduler.select(cloud).executor_id == "cloud-code"


def test_static_context_is_world_invariant_and_snapshot_adds_only_raw_facts(
    tmp_path: Path,
) -> None:
    scheduler = RepositoryLocalityScheduler(
        [
            CodeExecutor(executor_id="edge-code", site="edge", endpoint="http://edge"),
            CodeExecutor(executor_id="cloud-code", site="cloud", endpoint="http://cloud"),
        ]
    )
    contexts: list[CodePlannerContext] = []
    for world_id, site in [("H1", "edge"), ("H2", "cloud")]:
        trace = TraceRecorder(tmp_path / world_id, "run")
        context = CodePlannerContext(
            trace=trace,
            scheduler=scheduler,
            world=RepositoryWorld(
                world_id=world_id,
                repository_site=site,
                repository_artifact_id="repo://example",
                repository_size_bytes=10,
                bandwidth_mbps=1,
                rtt_ms=50,
            ),
            visibility="snapshot",
            planner_site="cloud",
        )
        contexts.append(context)
    assert contexts[0].render_static_context() == contexts[1].render_static_context()
    assert "location=edge" in contexts[0].render_dynamic_context()
    assert "location=cloud" in contexts[1].render_dynamic_context()
    assert "budget: 16 calls" in contexts[0].render_static_context()
    instructions = build_code_planner_instructions()
    for forbidden in ["whole_repo_remote_reasoning", "oracle winner", "verified patch"]:
        assert forbidden not in instructions


def test_invalidated_admission_exports_cannot_run() -> None:
    config_path = Path("configs/semantic_switch/astropy_14309.yaml").resolve()
    config = CodeBenchmarkConfig.from_yaml(config_path)
    with pytest.raises(ValueError, match="not_realizable"):
        load_admission_export(config, config_path, "H1_edge")
    with pytest.raises(ValueError, match="not_realizable"):
        load_admission_export(config, config_path, "H2_cloud")
    assert config.worlds["H1_edge"].model_dump(exclude={"world_id", "repository_site"}) == (
        config.worlds["H2_cloud"].model_dump(exclude={"world_id", "repository_site"})
    )


def test_config_contains_no_reference_workflow_or_evaluator_leakage() -> None:
    paths = [
        Path("configs/semantic_switch/astropy_14309.yaml"),
        Path("src/infra_mas/code_tasks/coordinator.py"),
    ]
    content = "\n".join(path.read_text(encoding="utf-8") for path in paths).lower()
    forbidden = [
        "whole_repo_remote_reasoning",
        "local_search_test_compact_remote_reasoning",
        "fail_to_pass",
        "pass_to_pass",
    ]
    assert not [term for term in forbidden if term in content]


def test_code_result_json_is_machine_readable(tmp_path: Path) -> None:
    payload = {"task_id": "astropy__astropy-14309", "resolved": True}
    path = tmp_path / "result.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert json.loads(path.read_text(encoding="utf-8")) == payload
