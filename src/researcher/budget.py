from pydantic import BaseModel, ConfigDict, Field


class Budget(BaseModel):
    """Caps on parallelism and per-worker search effort."""

    model_config = ConfigDict(frozen=True)

    max_workers: int = Field(default=4, ge=1, le=8)
    max_searches: int = Field(default=3, ge=1, le=10)
