FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace

# Copy dependency metadata first for build cache efficiency.
COPY agent/pyproject.toml /workspace/agent/pyproject.toml

# Copy source trees needed at runtime.
COPY agent /workspace/agent
COPY api /workspace/api

# Install project dependencies via uv (no pip usage).
RUN uv pip install --system /workspace/agent

# Start the FastAPI service by default.
CMD ["uv", "run", "--project", "/workspace/agent", "uvicorn", "api.api:app", "--host", "0.0.0.0", "--port", "8000"]
