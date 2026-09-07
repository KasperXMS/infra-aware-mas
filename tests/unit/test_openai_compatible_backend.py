"""Tests for the real OpenAI-compatible Worker backend."""

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice

from infra_mas.core.errors import ExecutionFailedError, InvalidModelResponseError
from infra_mas.core.model import ModelRequest
from infra_mas.worker.backends.openai_compatible import OpenAICompatibleBackend


class FakeCompletions:
    """Capture Chat Completions arguments and return a typed response."""

    def __init__(self, output: str | None = "model answer") -> None:
        self.output = output
        self.arguments: dict[str, object] = {}

    async def create(self, **kwargs: object) -> ChatCompletion:
        self.arguments = kwargs
        return ChatCompletion(
            id="completion-1",
            choices=[
                Choice(
                    finish_reason="stop",
                    index=0,
                    message=ChatCompletionMessage(role="assistant", content=self.output),
                )
            ],
            created=0,
            model="test-model",
            object="chat.completion",
        )


class FakeChat:
    def __init__(self, completions: FakeCompletions) -> None:
        self.completions = completions


class FakeModels:
    def __init__(self, model_ids: list[str]) -> None:
        self.model_ids = model_ids

    async def list(self) -> object:
        return SimpleNamespace(data=[SimpleNamespace(id=model_id) for model_id in self.model_ids])


class FakeClient:
    def __init__(
        self,
        completions: FakeCompletions,
        model_ids: list[str] | None = None,
    ) -> None:
        self.chat = FakeChat(completions)
        self.models = FakeModels(model_ids or ["test-model"])


async def test_backend_builds_text_and_image_request(tmp_path: Path) -> None:
    text_path = tmp_path / "evidence.txt"
    text_path.write_text("observed text", encoding="utf-8")
    image_path = tmp_path / "frame.png"
    image_path.write_bytes(b"fake-png")
    completions = FakeCompletions()
    backend = OpenAICompatibleBackend(
        "http://model.test/v1",
        "test-model",
        client=cast(AsyncOpenAI, FakeClient(completions)),
    )

    result = await backend.infer(
        ModelRequest(
            task="Analyze the inputs.",
            input_paths=[str(text_path), str(image_path)],
        )
    )

    assert result.output_text == "model answer"
    assert result.latency_ms >= 0
    messages = cast(list[dict[str, object]], completions.arguments["messages"])
    content = cast(list[dict[str, object]], messages[0]["content"])
    assert content[0] == {"type": "text", "text": "Analyze the inputs."}
    assert "observed text" in cast(str, content[1]["text"])
    image_url = cast(dict[str, str], content[2]["image_url"])["url"]
    assert image_url == "data:image/png;base64,ZmFrZS1wbmc="


async def test_backend_rejects_unsupported_binary_input(tmp_path: Path) -> None:
    binary = tmp_path / "archive.zip"
    binary.write_bytes(b"binary")
    backend = OpenAICompatibleBackend(
        "http://model.test/v1",
        "test-model",
        client=cast(AsyncOpenAI, FakeClient(FakeCompletions())),
    )

    with pytest.raises(ExecutionFailedError, match="unsupported model input media type"):
        await backend.infer(ModelRequest(task="Read it.", input_paths=[str(binary)]))


async def test_backend_rejects_empty_model_response() -> None:
    backend = OpenAICompatibleBackend(
        "http://model.test/v1",
        "test-model",
        client=cast(AsyncOpenAI, FakeClient(FakeCompletions(output=None))),
    )

    with pytest.raises(InvalidModelResponseError, match="textual content"):
        await backend.infer(ModelRequest(task="Answer.", input_paths=[]))


async def test_backend_readiness_requires_configured_model() -> None:
    backend = OpenAICompatibleBackend(
        "http://model.test/v1",
        "test-model",
        client=cast(AsyncOpenAI, FakeClient(FakeCompletions(), ["other-model"])),
    )

    with pytest.raises(ExecutionFailedError, match="is not exposed"):
        await backend.check()
