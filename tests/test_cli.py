import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from researcher import llm as llm_module
from researcher.cli import app
from tests.fakes import FakeLlm

runner = CliRunner()


def test_writes_report_and_trace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(llm_module, "get_client", lambda: FakeLlm(questions=2))
    out = tmp_path / "report.md"
    trace = tmp_path / "trace.json"

    result = runner.invoke(
        app, ["Why?", "-o", str(out), "--workers", "2", "--searches", "1", "--trace", str(trace)]
    )

    assert result.exit_code == 0, result.output
    assert out.read_text().startswith("# Report")
    steps = json.loads(trace.read_text())
    assert {"step", "model", "duration_ms", "input_tokens", "output_tokens"} <= set(steps[0])


def test_prints_report_without_output_flag(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(llm_module, "get_client", lambda: FakeLlm(questions=1))

    result = runner.invoke(app, ["Why?"])

    assert result.exit_code == 0
    assert "# Report" in result.stdout


def test_missing_api_key_fails_with_variable_name(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    result = runner.invoke(app, ["Why?"])

    assert result.exit_code == 1
    assert "ANTHROPIC_API_KEY" in result.output


def test_workers_out_of_range_is_rejected():
    result = runner.invoke(app, ["Why?", "--workers", "9"])

    assert result.exit_code == 2
