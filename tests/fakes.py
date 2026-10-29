"""Deterministic stand-ins for the Claude client (synthetic data only)."""

import asyncio
import time
from collections.abc import Iterable
from types import SimpleNamespace
from typing import Any

from researcher.llm import Completion, Parsed, Usage
from researcher.models import Plan, SubQuestion

USAGE = Usage(input_tokens=10, output_tokens=5)


def text_block(text: str) -> Any:
    return SimpleNamespace(type="text", text=text, citations=None)


def search_result_block(urls: Iterable[str]) -> Any:
    results = [
        SimpleNamespace(type="web_search_result", url=url, title=f"Title of {url}") for url in urls
    ]
    return SimpleNamespace(type="web_search_tool_result", content=results)


def search_error_block(code: str) -> Any:
    error = SimpleNamespace(type="web_search_tool_result_error", error_code=code)
    return SimpleNamespace(type="web_search_tool_result", content=error)


def search_use_block() -> Any:
    return SimpleNamespace(type="server_tool_use", name="web_search")


def completion(content: list[Any], stop_reason: str = "end_turn") -> Completion:
    return Completion(model="fake-worker", stop_reason=stop_reason, content=content, usage=USAGE)


class ScriptedLlm:
    """Returns queued completions in order and records every `complete` call."""

    def __init__(self, completions: list[Completion]) -> None:
        self._queue = list(completions)
        self.calls: list[dict[str, Any]] = []

    async def parse(self, **_: Any) -> Parsed[Any]:
        raise AssertionError("parse is not scripted")

    async def complete(self, **kwargs: Any) -> Completion:
        self.calls.append({**kwargs, "messages": list(kwargs["messages"])})
        return self._queue.pop(0)


class FakeLlm:
    """Plans N sub-questions, answers workers concurrently, records timings and prompts."""

    def __init__(self, questions: int = 3, fail: Iterable[str] = (), delay: float = 0.02) -> None:
        self._questions = questions
        self._fail = set(fail)
        self._delay = delay
        self.windows: dict[str, tuple[float, float]] = {}
        self.worker_calls = 0
        self.synthesis_prompts: list[str] = []

    async def parse(self, *, output_type: type[Any], **_: Any) -> Parsed[Any]:
        plan = Plan(
            sub_questions=[
                SubQuestion(id=f"q{i}", question=f"Question {i}?", rationale=f"Because {i}")
                for i in range(1, self._questions + 1)
            ]
        )
        return Parsed(model="fake-orchestrator", usage=USAGE, parsed=plan)

    async def complete(
        self, *, messages: list[dict[str, Any]], tools: Any = None, **_: Any
    ) -> Completion:
        prompt = str(messages[0]["content"])
        if not tools:
            self.synthesis_prompts.append(prompt)
            return completion([text_block("# Report\n\nAnswer [1].\n\n## Sources\n[1] x")])
        self.worker_calls += 1
        key = next(f"Question {i}?" for i in range(1, 99) if f"Question {i}?" in prompt)
        start = time.perf_counter()
        await asyncio.sleep(self._delay)
        self.windows[key] = (start, time.perf_counter())
        if key in self._fail:
            raise RuntimeError(f"boom {key}")
        n = key.removeprefix("Question ").removesuffix("?")
        return completion(
            [
                search_result_block([f"https://example.com/{n}"]),
                text_block(f"Finding for {key}"),
            ]
        )
