import json
import time

from pydantic import BaseModel, Field

from researcher.llm import Usage


class TraceStep(BaseModel):
    step: str
    model: str
    duration_ms: int
    input_tokens: int
    output_tokens: int


class Trace(BaseModel):
    steps: list[TraceStep] = Field(default_factory=list)

    def record(self, step: str, model: str, started: float, usage: Usage) -> None:
        """Append a step; `started` is a `time.perf_counter()` reading."""
        self.steps.append(
            TraceStep(
                step=step,
                model=model,
                duration_ms=round((time.perf_counter() - started) * 1000),
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
            )
        )

    def to_json(self) -> str:
        return json.dumps([s.model_dump() for s in self.steps], indent=2)
