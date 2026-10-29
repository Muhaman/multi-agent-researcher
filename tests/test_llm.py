from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import anthropic
import httpx
import pytest

from researcher.llm import (
    ClaudeClient,
    ConfigError,
    LlmInvalidOutputError,
    LlmMaxTokensError,
    LlmRefusalError,
    LlmUpstreamError,
    get_client,
)
from researcher.models import Plan, SubQuestion


def _client(*, create: Any = None, parse: Any = None) -> tuple[ClaudeClient, SimpleNamespace]:
    messages = SimpleNamespace(create=create or AsyncMock(), parse=parse or AsyncMock())
    sdk = SimpleNamespace(beta=SimpleNamespace(messages=messages))
    return ClaudeClient(sdk, orchestrator_model="claude-opus-5-5", worker_model="claude-sonnet-5-5"), messages  # type: ignore[arg-type]


def _response(stop_reason: str = "end_turn", parsed: Any = None) -> SimpleNamespace:
    return SimpleNamespace(
        stop_reason=stop_reason,
        content=[],
        parsed_output=parsed,
        usage=SimpleNamespace(input_tokens=7, output_tokens=3),
    )


async def _complete(client: ClaudeClient, role: str = "worker"):
    return await client.complete(
        role=role,  # type: ignore[arg-type]
        system="s",
        messages=[{"role": "user", "content": "q"}],
        tools=None,
        effort="medium",
        max_tokens=100,
    )


async def test_complete_sends_effort_and_fallbacks():
    client, messages = _client(create=AsyncMock(return_value=_response()))

    result = await _complete(client)

    kwargs = messages.create.await_args.kwargs
    assert kwargs["model"] == "claude-sonnet-5-5"
    assert kwargs["output_config"] == {"effort": "medium"}
    assert kwargs["fallbacks"] == "default"
    assert kwargs["betas"] == ["server-side-fallback-2026-07-01"]
    assert "temperature" not in kwargs
    assert (result.usage.input_tokens, result.usage.output_tokens) == (7, 3)


async def test_unsupported_model_gets_no_fallbacks():
    messages = SimpleNamespace(create=AsyncMock(return_value=_response()), parse=AsyncMock())
    sdk = SimpleNamespace(beta=SimpleNamespace(messages=messages))
    client = ClaudeClient(sdk, orchestrator_model="claude-haiku-5-5", worker_model="claude-haiku-5-5")  # type: ignore[arg-type]

    await _complete(client)

    assert "fallbacks" not in messages.create.await_args.kwargs


@pytest.mark.parametrize(
    ("stop_reason", "error"),
    [("refusal", LlmRefusalError), ("max_tokens", LlmMaxTokensError)],
)
async def test_bad_stop_reason_raises_typed_error(stop_reason: str, error: type[Exception]):
    client, _ = _client(create=AsyncMock(return_value=_response(stop_reason)))

    with pytest.raises(error):
        await _complete(client)


async def test_api_error_becomes_upstream_error():
    exc = anthropic.APIConnectionError(request=httpx.Request("POST", "https://example.com"))
    client, _ = _client(create=AsyncMock(side_effect=exc))

    with pytest.raises(LlmUpstreamError):
        await _complete(client)


async def test_parse_returns_parsed_output():
    plan = Plan(sub_questions=[SubQuestion(id="q1", question="a", rationale="b")])
    client, messages = _client(parse=AsyncMock(return_value=_response(parsed=plan)))

    result = await client.parse(
        role="orchestrator", system="s", prompt="p", output_type=Plan, effort="high", max_tokens=100
    )

    assert result.parsed == plan
    assert messages.parse.await_args.kwargs["output_format"] is Plan
    assert messages.parse.await_args.kwargs["model"] == "claude-opus-5-5"


@pytest.mark.parametrize("failure", [ValueError("bad json"), SyntaxError("truncated")])
async def test_parse_failure_becomes_invalid_output(failure: Exception):
    client, _ = _client(parse=AsyncMock(side_effect=failure))

    with pytest.raises(LlmInvalidOutputError):
        await client.parse(
            role="orchestrator", system="s", prompt="p", output_type=Plan, effort="high", max_tokens=100
        )


async def test_parse_with_no_parsed_output_is_invalid():
    client, _ = _client(parse=AsyncMock(return_value=_response(parsed=None)))

    with pytest.raises(LlmInvalidOutputError):
        await client.parse(
            role="orchestrator", system="s", prompt="p", output_type=Plan, effort="high", max_tokens=100
        )


def test_get_client_requires_api_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(ConfigError, match="ANTHROPIC_API_KEY"):
        get_client()
