from pydantic import BaseModel, Field

from researcher.trace import Trace


class SubQuestion(BaseModel):
    id: str
    question: str
    rationale: str


class Plan(BaseModel):
    sub_questions: list[SubQuestion]


class Source(BaseModel):
    url: str
    title: str


class Finding(BaseModel):
    id: str
    summary: str
    sources: list[Source] = Field(default_factory=list)
    error: str | None = None


class Report(BaseModel):
    markdown: str
    findings: list[Finding]
    trace: Trace
