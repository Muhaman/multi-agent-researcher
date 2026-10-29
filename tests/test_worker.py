from researcher.budget import Budget
from researcher.models import SubQuestion
from researcher.worker import run_worker
from tests.fakes import (
    ScriptedLlm,
    completion,
    search_error_block,
    search_result_block,
    search_use_block,
    text_block,
)

SQ = SubQuestion(id="q1", question="What is X?", rationale="context")


async def test_collects_sources_from_search_results():
    llm = ScriptedLlm(
        [
            completion(
                [
                    search_use_block(),
                    search_result_block(["https://a.example/1", "https://b.example/2"]),
                    text_block("X is a thing."),
                ]
            )
        ]
    )
    finding = await run_worker(llm, SQ, Budget(max_searches=5))

    assert finding.error is None
    assert finding.summary == "X is a thing."
    assert [s.url for s in finding.sources] == ["https://a.example/1", "https://b.example/2"]
    tool = llm.calls[0]["tools"][0]
    assert tool == {"type": "web_search_20260209", "name": "web_search", "max_uses": 5}
    assert llm.calls[0]["effort"] == "medium"
    assert llm.calls[0]["max_tokens"] == 16000


async def test_pause_turn_is_resumed():
    paused = completion([search_use_block()], stop_reason="pause_turn")
    done = completion([search_result_block(["https://a.example/1"]), text_block("Done.")])
    llm = ScriptedLlm([paused, done])

    finding = await run_worker(llm, SQ, Budget())

    assert len(llm.calls) == 2
    assert finding.summary == "Done."
    resumed = llm.calls[1]["messages"]
    assert [m["role"] for m in resumed] == ["user", "assistant"]


async def test_pause_turn_resumes_at_most_three_times():
    llm = ScriptedLlm([completion([search_use_block()], "pause_turn") for _ in range(4)])

    finding = await run_worker(llm, SQ, Budget())

    assert len(llm.calls) == 4
    assert finding.error is not None
    assert "did not finish" in finding.error


async def test_pause_turn_with_partial_text_is_not_a_success():
    paused = completion(
        [search_result_block(["https://a.example/1"]), text_block("partial")], "pause_turn"
    )
    llm = ScriptedLlm([paused for _ in range(4)])

    finding = await run_worker(llm, SQ, Budget())

    assert finding.error is not None
    assert finding.summary == ""


async def test_search_error_becomes_worker_error():
    llm = ScriptedLlm([completion([search_error_block("unavailable"), text_block("guess")])])

    finding = await run_worker(llm, SQ, Budget())

    assert finding.error is not None
    assert "unavailable" in finding.error
    assert finding.sources == []


async def test_exception_becomes_error_finding():
    class Boom:
        async def complete(self, **_: object) -> None:
            raise RuntimeError("upstream down")

    finding = await run_worker(Boom(), SQ, Budget())  # type: ignore[arg-type]

    assert finding.id == "q1"
    assert finding.error is not None
    assert "upstream down" in finding.error
