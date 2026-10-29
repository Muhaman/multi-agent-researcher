import time
from typing import Any

import structlog

from researcher.budget import Budget
from researcher.llm import Llm, Usage
from researcher.models import Finding, Source, SubQuestion
from researcher.trace import Trace

log = structlog.get_logger()

MAX_RESUMES = 3
MAX_TOKENS = 8000

SYSTEM = (
    "You are a research sub-agent. Answer the sub-question using web_search. "
    "Write a concise, factual summary (150-300 words) grounded only in what the search "
    "results say. Do not speculate beyond the sources."
)


class WorkerError(Exception):
    """A worker could not produce a usable finding."""


def _scan(blocks: list[Any]) -> tuple[str, list[Source], str | None]:
    """Return (summary text, de-duplicated sources, first search error code)."""
    texts: list[str] = []
    sources: dict[str, Source] = {}
    error: str | None = None

    def add(url: str | None, title: str | None) -> None:
        if url and url not in sources:
            sources[url] = Source(url=url, title=title or url)

    for block in blocks:
        kind = getattr(block, "type", None)
        if kind == "text":
            texts.append(str(block.text))
            for citation in getattr(block, "citations", None) or []:
                add(getattr(citation, "url", None), getattr(citation, "title", None))
        elif kind == "web_search_tool_result":
            content = block.content
            if isinstance(content, list):
                for result in content:
                    if getattr(result, "type", None) == "web_search_result":
                        add(result.url, result.title)
            else:
                error = error or str(getattr(content, "error_code", "unknown_error"))
    return "".join(texts).strip(), list(sources.values()), error


async def _research(llm: Llm, sq: SubQuestion, budget: Budget, trace: Trace | None) -> Finding:
    started = time.perf_counter()
    first = {"role": "user", "content": f"Sub-question: {sq.question}\nContext: {sq.rationale}"}
    tools = [{"type": "web_search_20260209", "name": "web_search", "max_uses": budget.max_searches}]
    messages: list[dict[str, Any]] = [first]
    blocks: list[Any] = []
    usage = Usage()
    model = ""

    for _ in range(MAX_RESUMES + 1):
        completion = await llm.complete(
            role="worker",
            system=SYSTEM,
            messages=messages,
            tools=tools,
            effort="medium",
            max_tokens=MAX_TOKENS,
        )
        blocks.extend(completion.content)
        usage += completion.usage
        model = completion.model
        if completion.stop_reason != "pause_turn":
            break
        messages = [first, {"role": "assistant", "content": list(blocks)}]

    if trace is not None:
        trace.record(f"worker:{sq.id}", model, started, usage)

    summary, sources, search_error = _scan(blocks)
    if search_error and not sources:
        raise WorkerError(f"web_search failed: {search_error}")
    if not summary:
        raise WorkerError("worker produced no summary")
    return Finding(id=sq.id, summary=summary, sources=sources)


async def run_worker(
    llm: Llm, sq: SubQuestion, budget: Budget, *, trace: Trace | None = None
) -> Finding:
    """Research one sub-question; any failure is captured in `Finding.error`."""
    try:
        return await _research(llm, sq, budget, trace)
    except Exception as exc:
        log.warning("worker.failed", id=sq.id, error=str(exc))
        return Finding(id=sq.id, summary="", error=f"{type(exc).__name__}: {exc}")
