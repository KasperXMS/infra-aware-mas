"""HTTP client for a repository code Worker."""

from __future__ import annotations

import json
from time import perf_counter
from typing import cast

import httpx

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


class CodeWorkerClient:
    """Invoke bounded repository operations and expose observed transport metrics."""

    def __init__(
        self,
        endpoint: str,
        *,
        timeout_seconds: float = 900,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._endpoint = endpoint.rstrip("/")
        self._client = client or httpx.AsyncClient(timeout=timeout_seconds)
        self._owns_client = client is None

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def health(self) -> dict[str, object]:
        response = await self._client.get(f"{self._endpoint}/health")
        response.raise_for_status()
        value: object = response.json()
        if not isinstance(value, dict):
            raise ValueError("code Worker health response must be an object")
        return cast(dict[str, object], value)

    async def reset(self, run_id: str) -> ResetResult:
        response = await self._client.post(
            f"{self._endpoint}/reset", json=ResetRequest(run_id=run_id).model_dump()
        )
        response.raise_for_status()
        return ResetResult.model_validate(response.json())

    async def invoke(
        self, tool: str, payload: dict[str, object]
    ) -> tuple[CodeToolResult, int, int, float]:
        models = {
            "search_code": SearchRequest,
            "read_file": ReadRequest,
            "edit_file": EditRequest,
            "apply_patch": PatchRequest,
            "run_targeted_test": TargetedTestRequest,
        }
        request_model = models.get(tool)
        body = request_model.model_validate(payload).model_dump() if request_model else {}
        request_bytes = len(json.dumps(body, ensure_ascii=False).encode("utf-8"))
        started = perf_counter()
        response = await self._client.post(f"{self._endpoint}/tools/{tool}", json=body)
        elapsed_ms = (perf_counter() - started) * 1000
        response_bytes = len(response.content)
        response.raise_for_status()
        result = CodeToolResult.model_validate(response.json())
        return result, request_bytes, response_bytes, elapsed_ms
