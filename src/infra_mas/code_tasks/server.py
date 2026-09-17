"""Sandboxed repository Worker for generic Planner code tools."""

from __future__ import annotations

import asyncio
import os
import re
from collections.abc import Awaitable, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import TypeVar
from uuid import uuid4

import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from infra_mas.code_tasks.models import (
    CodeToolResult,
    EditRequest,
    PatchRequest,
    ReadRequest,
    ResetRequest,
    ResetResult,
    SearchRequest,
    TargetedTestRequest,
)


class CodeWorkerConfig(BaseModel):
    """Configure one repository workspace without exposing it to the Planner."""

    model_config = ConfigDict(extra="forbid")

    worker_id: str
    site: str
    host: str = "0.0.0.0"
    port: int = Field(ge=1, le=65535)
    repository_root: Path
    base_commit: str
    targeted_test_prefix: list[str] = ["python", "-m", "pytest", "-q"]
    full_test_command: list[str] = ["python", "-m", "pytest", "-q"]
    max_output_chars: int = Field(default=20_000, ge=1000, le=200_000)

    @classmethod
    def from_yaml(cls, path: Path) -> CodeWorkerConfig:
        raw: object = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls.model_validate(raw)


class CodeWorkspaceService:
    """Execute a fixed set of repository operations beneath one checked-out root."""

    def __init__(self, config: CodeWorkerConfig) -> None:
        self.config = config
        self.root = config.repository_root.resolve()
        if not (self.root / ".git").is_dir():
            raise ValueError(f"repository root is not a Git checkout: {self.root}")
        self._lock = asyncio.Lock()

    async def reset(self, request: ResetRequest) -> ResetResult:
        del request
        async with self._lock:
            started = perf_counter()
            await self._run(["git", "reset", "--hard", self.config.base_commit])
            await self._run(["git", "clean", "-fd"])
            head = (await self._run(["git", "rev-parse", "HEAD"])).stdout.strip()
            status = await self._run(["git", "status", "--porcelain"])
            return ResetResult(
                base_commit=self.config.base_commit,
                head_commit=head,
                clean=not status.stdout.strip(),
                service_ms=(perf_counter() - started) * 1000,
            )

    async def search(self, request: SearchRequest) -> CodeToolResult:
        path = self._safe_relative(request.path, allow_directory=True)
        started = perf_counter()
        async with self._lock:
            completed = await self._run(
                ["git", "grep", "-F", "-n", "-I", "-i", "-e", request.query, "--", path],
                check=False,
            )
        if completed.returncode == 1 and not completed.stderr:
            output = "No matches found."
            success = True
        else:
            output = completed.stdout
            if completed.stderr:
                output = f"{output}\n{completed.stderr}".strip()
            success = completed.returncode == 0
        lines = output.splitlines()
        result_limited = len(lines) > request.max_results
        if result_limited:
            output = "\n".join(lines[: request.max_results]) + "\n[result limit reached]"
        output, char_limited = self._truncate(output)
        return CodeToolResult(
            tool="search_code",
            success=success,
            output=output,
            service_ms=(perf_counter() - started) * 1000,
            truncated=result_limited or char_limited,
        )

    async def read(self, request: ReadRequest) -> CodeToolResult:
        started = perf_counter()
        if request.end_line < request.start_line:
            raise ValueError("end_line must be greater than or equal to start_line")
        end_line = min(request.end_line, request.start_line + 500)
        relative = self._safe_relative(request.path)
        candidate = (self.root / relative).resolve()
        if not candidate.is_file():
            return CodeToolResult(
                tool="read_file",
                success=False,
                output=f"File not found: {relative}",
                service_ms=(perf_counter() - started) * 1000,
            )
        try:
            lines = await asyncio.to_thread(candidate.read_text, encoding="utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("read_file only accepts UTF-8 text") from error
        selected = lines.splitlines()[request.start_line - 1 : end_line]
        output = "\n".join(
            f"{number}: {line}"
            for number, line in enumerate(selected, start=request.start_line)
        )
        output, truncated = self._truncate(output)
        return CodeToolResult(
            tool="read_file",
            success=True,
            output=output,
            service_ms=(perf_counter() - started) * 1000,
            truncated=truncated or end_line < request.end_line,
        )

    async def apply_patch(self, request: PatchRequest) -> CodeToolResult:
        if len(request.patch) > 200_000:
            raise ValueError("patch exceeds 200000 characters")
        self._validate_patch_paths(request.patch)
        async with self._lock:
            started = perf_counter()
            checked = await self._run(
                ["git", "apply", "--check", "-"], input_text=request.patch, check=False
            )
            if checked.returncode != 0:
                output, truncated = self._truncate(checked.stderr or checked.stdout)
                return CodeToolResult(
                    tool="apply_patch",
                    success=False,
                    output=output,
                    service_ms=(perf_counter() - started) * 1000,
                    truncated=truncated,
                )
            applied = await self._run(
                ["git", "apply", "-"], input_text=request.patch, check=False
            )
            output = applied.stderr or applied.stdout or "Patch applied successfully."
            output, truncated = self._truncate(output)
            return CodeToolResult(
                tool="apply_patch",
                success=applied.returncode == 0,
                output=output,
                service_ms=(perf_counter() - started) * 1000,
                truncated=truncated,
            )

    async def edit_file(self, request: EditRequest) -> CodeToolResult:
        started = perf_counter()
        relative = self._safe_relative(request.path)
        candidate = (self.root / relative).resolve()
        if not candidate.is_file():
            return CodeToolResult(
                tool="edit_file",
                success=False,
                output=f"File not found: {relative}",
                service_ms=(perf_counter() - started) * 1000,
            )
        async with self._lock:
            try:
                content = await asyncio.to_thread(candidate.read_text, encoding="utf-8")
            except UnicodeDecodeError:
                return CodeToolResult(
                    tool="edit_file",
                    success=False,
                    output="edit_file only accepts UTF-8 text",
                    service_ms=(perf_counter() - started) * 1000,
                )
            occurrences = content.count(request.old_text)
            if occurrences != 1:
                return CodeToolResult(
                    tool="edit_file",
                    success=False,
                    output=f"old_text must occur exactly once; found {occurrences} occurrences.",
                    service_ms=(perf_counter() - started) * 1000,
                )
            updated = content.replace(request.old_text, request.new_text, 1)
            temporary = candidate.with_name(f".{candidate.name}.{uuid4().hex}.tmp")
            try:
                await asyncio.to_thread(temporary.write_text, updated, encoding="utf-8")
                await asyncio.to_thread(os.replace, temporary, candidate)
            finally:
                if await asyncio.to_thread(temporary.is_file):
                    await asyncio.to_thread(temporary.unlink)
        return CodeToolResult(
            tool="edit_file",
            success=True,
            output=f"Updated {relative} by one exact replacement.",
            service_ms=(perf_counter() - started) * 1000,
        )

    async def targeted_test(self, request: TargetedTestRequest) -> CodeToolResult:
        path = self._safe_relative(request.test_path)
        return await self._command_result(
            "run_targeted_test", [*self.config.targeted_test_prefix, path], allow_failure=True
        )

    async def full_test(self) -> CodeToolResult:
        return await self._command_result(
            "run_full_test", self.config.full_test_command, allow_failure=True
        )

    async def submit(self) -> CodeToolResult:
        started = perf_counter()
        patch_result = await self._run(
            ["git", "diff", "--binary", self.config.base_commit, "--"], check=False
        )
        patch = patch_result.stdout
        success = patch_result.returncode == 0 and bool(patch.strip())
        summary = (
            f"Submitted a {len(patch.encode('utf-8'))}-byte patch."
            if success
            else "No non-empty patch is available to submit."
        )
        return CodeToolResult(
            tool="submit_patch",
            success=success,
            output=summary,
            service_ms=(perf_counter() - started) * 1000,
            patch=patch if success else None,
        )

    async def _command_result(
        self, tool: str, command: Sequence[str], *, allow_failure: bool
    ) -> CodeToolResult:
        started = perf_counter()
        async with self._lock:
            completed = await self._run(command, check=not allow_failure)
        output = completed.stdout
        if completed.stderr:
            output = f"{output}\n{completed.stderr}".strip()
        output, truncated = self._truncate(output)
        return CodeToolResult.model_validate(
            {
                "tool": tool,
                "success": completed.returncode == 0,
                "output": output or f"Command exited with status {completed.returncode}.",
                "service_ms": (perf_counter() - started) * 1000,
                "truncated": truncated,
            }
        )

    async def _run(
        self,
        command: Sequence[str],
        *,
        input_text: str | None = None,
        check: bool = True,
    ) -> _CompletedProcess:
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=self.root,
            stdin=asyncio.subprocess.PIPE if input_text is not None else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate(
            input_text.encode("utf-8") if input_text is not None else None
        )
        stdout_text = stdout.decode("utf-8", errors="replace")
        stderr_text = stderr.decode("utf-8", errors="replace")
        if check and process.returncode != 0:
            raise RuntimeError(stderr_text or stdout_text)
        return _CompletedProcess(
            process.returncode or 0,
            stdout_text,
            stderr_text,
        )

    def _safe_relative(self, value: str, *, allow_directory: bool = False) -> str:
        del allow_directory
        raw = value.strip() or "."
        candidate = (self.root / raw).resolve()
        relative = candidate.relative_to(self.root) if candidate.is_relative_to(self.root) else None
        if relative is None or ".git" in relative.parts:
            raise ValueError("path escapes the repository workspace")
        return relative.as_posix() or "."

    def _validate_patch_paths(self, patch: str) -> None:
        paths = re.findall(r"^diff --git a/(.+?) b/(.+?)$", patch, flags=re.MULTILINE)
        if not paths:
            raise ValueError("apply_patch requires a unified Git diff")
        for old_path, new_path in paths:
            self._safe_relative(old_path)
            self._safe_relative(new_path)

    def _truncate(self, value: str) -> tuple[str, bool]:
        if len(value) <= self.config.max_output_chars:
            return value, False
        return value[: self.config.max_output_chars] + "\n[truncated]", True


@dataclass(frozen=True, slots=True)
class _CompletedProcess:
    returncode: int
    stdout: str
    stderr: str


def create_code_worker_app(service: CodeWorkspaceService) -> FastAPI:
    app = FastAPI(title="Infra-Aware MAS Code Worker")

    async def health() -> dict[str, object]:
        return {
            "status": "ok",
            "worker_id": service.config.worker_id,
            "site": service.config.site,
            "base_commit": service.config.base_commit,
        }

    async def reset(request: ResetRequest) -> ResetResult:
        return await _translate_errors(service.reset(request))

    async def search(request: SearchRequest) -> CodeToolResult:
        return await _translate_errors(service.search(request))

    async def read(request: ReadRequest) -> CodeToolResult:
        return await _translate_errors(service.read(request))

    async def apply_patch(request: PatchRequest) -> CodeToolResult:
        return await _translate_errors(service.apply_patch(request))

    async def edit_file(request: EditRequest) -> CodeToolResult:
        return await _translate_errors(service.edit_file(request))

    async def targeted_test(request: TargetedTestRequest) -> CodeToolResult:
        return await _translate_errors(service.targeted_test(request))

    async def full_test() -> CodeToolResult:
        return await _translate_errors(service.full_test())

    async def submit() -> CodeToolResult:
        return await _translate_errors(service.submit())

    app.add_api_route("/health", health, methods=["GET"])
    app.add_api_route("/reset", reset, methods=["POST"], response_model=ResetResult)
    app.add_api_route(
        "/tools/search_code", search, methods=["POST"], response_model=CodeToolResult
    )
    app.add_api_route(
        "/tools/read_file", read, methods=["POST"], response_model=CodeToolResult
    )
    app.add_api_route(
        "/tools/edit_file", edit_file, methods=["POST"], response_model=CodeToolResult
    )
    app.add_api_route(
        "/tools/apply_patch", apply_patch, methods=["POST"], response_model=CodeToolResult
    )
    app.add_api_route(
        "/tools/run_targeted_test",
        targeted_test,
        methods=["POST"],
        response_model=CodeToolResult,
    )
    app.add_api_route(
        "/tools/run_full_test", full_test, methods=["POST"], response_model=CodeToolResult
    )
    app.add_api_route(
        "/tools/submit_patch", submit, methods=["POST"], response_model=CodeToolResult
    )
    return app


T = TypeVar("T")


async def _translate_errors(awaitable: Awaitable[T]) -> T:
    try:
        return await awaitable
    except (ValueError, FileNotFoundError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
