"""Planner model configuration kept inside the SDK-owning planner layer."""

import os
from typing import Annotated, Literal

from agents import OpenAIChatCompletionsModel, OpenAIResponsesModel
from agents.models.interface import Model
from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class PlannerModelConfig(BaseModel):
    """Configure either the official API or an OpenAI-compatible planner endpoint."""

    model_config = ConfigDict(extra="forbid")

    model: NonEmptyString
    api: Literal["responses", "chat_completions"] = "responses"
    base_url: NonEmptyString | None = None
    api_key_env: NonEmptyString | None = "OPENAI_API_KEY"
    timeout_seconds: Annotated[float, Field(gt=0)] = 120.0


def create_planner_model(config: PlannerModelConfig) -> tuple[Model, AsyncOpenAI]:
    """Build a per-run SDK model and its explicitly managed OpenAI client."""
    api_key = "not-required"
    if config.api_key_env is not None:
        api_key = os.getenv(config.api_key_env, "")
        if not api_key:
            raise ValueError(f"required environment variable {config.api_key_env!r} is not set")

    client = AsyncOpenAI(
        api_key=api_key,
        base_url=config.base_url,
        timeout=config.timeout_seconds,
        max_retries=0,
    )
    if config.api == "chat_completions":
        return OpenAIChatCompletionsModel(config.model, client), client
    return OpenAIResponsesModel(config.model, client), client
