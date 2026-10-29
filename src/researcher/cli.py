import asyncio
import logging
import sys
from pathlib import Path
from typing import Annotated

import structlog
import typer

from researcher import llm as llm_module
from researcher.budget import Budget
from researcher.llm import ConfigError, LlmError
from researcher.orchestrator import ResearchFailed, research

app = typer.Typer(add_completion=False, help="Research a question with parallel web-search agents.")


def _configure_logging() -> None:
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
    )


@app.command()
def main(
    question: Annotated[str, typer.Argument(help="The research question.")],
    output: Annotated[
        Path | None, typer.Option("-o", "--output", help="Write the report here.")
    ] = None,
    workers: Annotated[int, typer.Option(min=1, max=8, help="Parallel sub-agents.")] = 4,
    searches: Annotated[int, typer.Option(min=1, max=10, help="Searches per sub-agent.")] = 3,
    trace: Annotated[Path | None, typer.Option(help="Write the step trace (JSON) here.")] = None,
) -> None:
    _configure_logging()
    try:
        client = llm_module.get_client()
        budget = Budget(max_workers=workers, max_searches=searches)
        report = asyncio.run(research(question, llm=client, budget=budget))
    except (ConfigError, ResearchFailed, LlmError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc

    if output is None:
        typer.echo(report.markdown)
    else:
        output.write_text(report.markdown)
    if trace is not None:
        trace.write_text(report.trace.to_json())
