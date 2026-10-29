import pytest

from researcher.budget import Budget
from researcher.orchestrator import ResearchFailed, research
from tests.fakes import FakeLlm


def _overlap(windows: dict[str, tuple[float, float]]) -> bool:
    starts = [s for s, _ in windows.values()]
    ends = [e for _, e in windows.values()]
    return max(starts) < min(ends)


async def test_three_sub_questions_run_concurrently():
    llm = FakeLlm(questions=3)

    report = await research("Why?", llm=llm, budget=Budget())

    assert llm.worker_calls == 3
    assert _overlap(llm.windows)
    assert len(report.findings) == 3
    assert report.markdown.startswith("# Report")


async def test_failed_worker_is_skipped_in_synthesis():
    llm = FakeLlm(questions=3, fail=["Question 2?"])

    report = await research("Why?", llm=llm, budget=Budget())

    assert llm.synthesis_prompts[0].count("<finding ") == 2
    assert [f.error is not None for f in report.findings] == [False, True, False]
    assert report.markdown


async def test_all_workers_failing_raises():
    llm = FakeLlm(questions=2, fail=["Question 1?", "Question 2?"])

    with pytest.raises(ResearchFailed):
        await research("Why?", llm=llm, budget=Budget())

    assert llm.synthesis_prompts == []


async def test_plan_is_truncated_to_max_workers():
    llm = FakeLlm(questions=6)

    report = await research("Why?", llm=llm, budget=Budget(max_workers=4))

    assert llm.worker_calls == 4
    assert len(report.findings) == 4


async def test_trace_records_every_step():
    report = await research("Why?", llm=FakeLlm(questions=2), budget=Budget())

    steps = [s.step for s in report.trace.steps]
    assert steps[0] == "plan"
    assert steps[-1] == "synthesize"
    assert sorted(steps[1:-1]) == ["worker:q1", "worker:q2"]
    assert all(s.input_tokens == 10 and s.output_tokens == 5 for s in report.trace.steps)
