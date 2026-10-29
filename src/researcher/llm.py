"""Thin async wrapper over the Anthropic SDK: typed errors, usage, refusal fallbacks."""

import os
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import anthropic
from anthropic import AsyncAnthropic

Role = Literal["orchestrator", "worker"]
Effort = Literal["low", "medium", "high"]

DEFAULT_ORCHESTRATOR_MODEL = "claude-opus-5-5"
DEFAULT_WORKER_MODEL = "claude-sonnet-5-5"
FALLBACK_BETA = "server-side-fallback-2026-07-01"
# Only these models accept `fallbacks: "default"`; sending it elsewhere is a 400.
FALLBACK_MODELS = frozenset({"claude-opus-5-5", "claude-sonnet-5-5"})


class ConfigError(Exception):
    """Required configuration is missing."""


class LlmError(Exception):
    """Base class for model-side failures."""


class LlmRefusalError(LlmError):
    """The model declined the request."""


class LlmMaxTokensError(LlmError):
    """The response was truncated at max_tokens."""


class LlmInvalidOutputError(LlmError):
    """Structured output was missing, truncated or failed schema validation."""


class LlmUpstreamError(LlmError):
    """The Anthropic API call failed."""


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            self.input_tokens + other.input_tokens, self.output_tokens + other.output_tokens
        )


@dataclass(frozen=True)
class Completion:
    model: str
    stop_reason: str
    content: list[Any]
    usage: Usage


@dataclass(frozen=True)
class Parsed[T]:
    model: str
    usage: Usage
    parsed: T


class Llm(Protocol):
    async def parse[T](
        self,
        *,
        role: Role,
        system: str,
        prompt: str,
        output_type: type[T],
        effort: Effort,
        max_tokens: int,
    ) -> Parsed[T]: ...

    async def complete(
        self,
        *,
        role: Role,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        effort: Effort,
        max_tokens: int,
    ) -> Completion: ...


def _check_stop_reason(stop_reason: str | None) -> None:
    if stop_reason == "refusal":
        raise LlmRefusalError("The model refused the request.")
    if stop_reason == "max_tokens":
        raise LlmMaxTokensError("The response was truncated at max_tokens.")


def _usage(response: Any) -> Usage:
    return Usage(int(response.usage.input_tokens), int(response.usage.output_tokens))


class ClaudeClient:
    def __init__(
        self, client: AsyncAnthropic, *, orchestrator_model: str, worker_model: str
    ) -> None:
        self._client = client
        self._models: dict[Role, str] = {
            "orchestrator": orchestrator_model,
            "worker": worker_model,
        }

    def _base_request(
        self, role: Role, system: str, effort: Effort, max_tokens: int
    ) -> dict[str, Any]:
        model = self._models[role]
        request: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system,
            "output_config": {"effort": effort},
        }
        if model in FALLBACK_MODELS:
            request["betas"] = [FALLBACK_BETA]
            request["fallbacks"] = "default"
        return request

    async def parse[T](
        self,
        *,
        role: Role,
        system: str,
        prompt: str,
        output_type: type[T],
        effort: Effort,
        max_tokens: int,
    ) -> Parsed[T]:
        request = self._base_request(role, system, effort, max_tokens)
        try:
            response: Any = await self._client.beta.messages.parse(
                **request,
                messages=[{"role": "user", "content": prompt}],
                output_format=output_type,
            )
        except anthropic.APIError as exc:
            raise LlmUpstreamError(str(exc)) from exc
        except (ValueError, SyntaxError) as exc:
            raise LlmInvalidOutputError(f"Model output failed validation: {exc}") from exc
        _check_stop_reason(response.stop_reason)
        if response.parsed_output is None:
            raise LlmInvalidOutputError("Model returned no structured output.")
        parsed: T = response.parsed_output
        return Parsed(model=self._models[role], usage=_usage(response), parsed=parsed)

    async def complete(
        self,
        *,
        role: Role,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        effort: Effort,
        max_tokens: int,
    ) -> Completion:
        request = self._base_request(role, system, effort, max_tokens)
        request["messages"] = messages
        if tools:
            request["tools"] = tools
        try:
            response: Any = await self._client.beta.messages.create(**request)
        except anthropic.APIError as exc:
            raise LlmUpstreamError(str(exc)) from exc
        _check_stop_reason(response.stop_reason)
        return Completion(
            model=self._models[role],
            stop_reason=str(response.stop_reason),
            content=list(response.content),
            usage=_usage(response),
        )


def get_client() -> Llm:
    """Build the production client; fails fast when the API key is missing."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise ConfigError("ANTHROPIC_API_KEY is not set.")
    return ClaudeClient(
        AsyncAnthropic(),
        orchestrator_model=os.environ.get("ANTHROPIC_MODEL", DEFAULT_ORCHESTRATOR_MODEL),
        worker_model=os.environ.get("RESEARCH_WORKER_MODEL", DEFAULT_WORKER_MODEL),
    )
