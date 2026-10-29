FROM python:3.12-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.13.6 /uv /bin/uv
WORKDIR /app
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1
COPY pyproject.toml README.md ./
COPY src ./src
RUN uv sync --no-dev --no-editable

FROM python:3.12-slim
RUN useradd --uid 10001 --create-home researcher
WORKDIR /home/researcher
COPY --from=build /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1
USER 10001
ENTRYPOINT ["research"]
