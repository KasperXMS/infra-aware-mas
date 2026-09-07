"""OpenAI-compatible Chat Completions model backend."""

import asyncio
import base64
import mimetypes
from pathlib import Path
from time import perf_counter

from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletionContentPartImageParam,
    ChatCompletionContentPartTextParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

from infra_mas.core.errors import ExecutionFailedError, InvalidModelResponseError
from infra_mas.core.model import ModelRequest, ModelResult


class OpenAICompatibleBackend:
    """Invoke a worker-local OpenAI-compatible Chat Completions endpoint."""

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: str = "not-required",
        timeout_seconds: float = 120.0,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        verify_model: bool = True,
        client: AsyncOpenAI | None = None,
    ) -> None:
        if not base_url.strip():
            raise ValueError("base_url must not be empty")
        if not model.strip():
            raise ValueError("model must not be empty")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if not 0 <= temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")

        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._verify_model = verify_model
        self._client = client or AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout_seconds,
            max_retries=0,
        )
        self._owns_client = client is None

    async def check(self) -> None:
        """Verify endpoint connectivity and configured model visibility."""
        if not self._verify_model:
            return
        try:
            models = await self._client.models.list()
        except Exception as error:
            raise ExecutionFailedError(
                f"OpenAI-compatible endpoint readiness check failed: {error}"
            ) from error
        available = {model.id for model in models.data}
        if self._model not in available:
            raise ExecutionFailedError(
                f"configured model {self._model!r} is not exposed by the endpoint; "
                f"available models: {sorted(available)}"
            )

    async def infer(self, request: ModelRequest) -> ModelResult:
        """Build a multimodal request, invoke the endpoint, and measure latency."""
        content: list[ChatCompletionContentPartTextParam | ChatCompletionContentPartImageParam] = [
            {"type": "text", "text": request.task}
        ]
        for raw_path in request.input_paths:
            content.append(await self._content_part(Path(raw_path)))

        system_message: ChatCompletionSystemMessageParam = {
            "role": "system",
            "content": request.instructions,
        }
        user_message: ChatCompletionUserMessageParam = {
            "role": "user",
            "content": content,
        }
        started_at = perf_counter()
        try:
            completion = await self._client.chat.completions.create(
                model=self._model,
                messages=[system_message, user_message],
                max_tokens=self._max_tokens,
                temperature=self._temperature,
            )
        except Exception as error:
            raise ExecutionFailedError(f"OpenAI-compatible endpoint failed: {error}") from error
        latency_ms = (perf_counter() - started_at) * 1000

        if not completion.choices:
            raise InvalidModelResponseError("model response did not contain any choices")
        output = completion.choices[0].message.content
        if not isinstance(output, str) or not output.strip():
            raise InvalidModelResponseError("model response did not contain textual content")
        return ModelResult(output_text=output, latency_ms=latency_ms)

    async def aclose(self) -> None:
        """Close the internally owned OpenAI client."""
        if self._owns_client:
            await self._client.close()

    @staticmethod
    async def _content_part(
        path: Path,
    ) -> ChatCompletionContentPartTextParam | ChatCompletionContentPartImageParam:
        if not await asyncio.to_thread(path.is_file):
            raise ExecutionFailedError(f"model input file not found: {path}")

        media_type, _ = mimetypes.guess_type(path.name)
        if media_type is not None and media_type.startswith("image/"):
            data = await asyncio.to_thread(path.read_bytes)
            encoded = base64.b64encode(data).decode("ascii")
            return {
                "type": "image_url",
                "image_url": {"url": f"data:{media_type};base64,{encoded}"},
            }
        if (
            media_type is None
            or media_type.startswith("text/")
            or media_type
            in {
                "application/json",
                "application/xml",
                "application/x-yaml",
                "application/yaml",
            }
        ):
            try:
                text = await asyncio.to_thread(path.read_text, encoding="utf-8")
            except UnicodeDecodeError as error:
                raise ExecutionFailedError(
                    f"unsupported binary model input type for {path.name!r}"
                ) from error
            return {
                "type": "text",
                "text": f"Input artifact {path.name}:\n{text}",
            }
        raise ExecutionFailedError(
            f"unsupported model input media type {media_type!r} for {path.name!r}"
        )
