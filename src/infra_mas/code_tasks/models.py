"""Schemas shared by code-task controllers and Workers."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
CodeToolName = Literal[
    "search_code",
    "read_file",
    "edit_file",
    "apply_patch",
    "run_targeted_test",
    "run_full_test",
    "submit_patch",
]


class CodeExecutor(BaseModel):
    """One physical endpoint capable of operating a repository workspace."""

    model_config = ConfigDict(extra="forbid")

    executor_id: NonEmptyString
    site: NonEmptyString
    endpoint: NonEmptyString


class RepositoryWorld(BaseModel):
    """Raw world facts required to place and operate one repository artifact."""

    model_config = ConfigDict(extra="forbid")

    world_id: NonEmptyString
    repository_artifact_id: NonEmptyString
    repository_site: NonEmptyString
    repository_size_bytes: Annotated[int, Field(ge=0)]
    bandwidth_mbps: Annotated[float, Field(gt=0)]
    rtt_ms: Annotated[float, Field(ge=0)]


class ResetRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: NonEmptyString


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: NonEmptyString
    path: str = "."
    max_results: Annotated[int, Field(ge=1, le=200)] = 50


class ReadRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: NonEmptyString
    start_line: Annotated[int, Field(ge=1)] = 1
    end_line: Annotated[int, Field(ge=1)] = 200


class PatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    patch: Annotated[str, Field(min_length=1)]


class EditRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: NonEmptyString
    old_text: Annotated[str, Field(min_length=1, max_length=100_000)]
    new_text: Annotated[str, Field(max_length=100_000)]


class TargetedTestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    test_path: NonEmptyString


class CodeToolResult(BaseModel):
    """Bounded tool output plus Worker-measured service data."""

    model_config = ConfigDict(extra="forbid")

    tool: CodeToolName
    success: bool
    output: str
    service_ms: Annotated[float, Field(ge=0)]
    truncated: bool = False
    patch: str | None = None


class ResetResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_commit: NonEmptyString
    head_commit: NonEmptyString
    clean: bool
    service_ms: Annotated[float, Field(ge=0)]
