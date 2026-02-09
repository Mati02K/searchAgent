FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace/agent

# Copy dependency metadata first for build cache efficiency.
COPY agent/pyproject.toml /workspace/agent/pyproject.toml

# Copy the agent source.
COPY agent /workspace/agent

# Install project dependencies via uv (no pip usage).
RUN uv pip install --system /workspace/agent

# Temporary default entrypoint for smoke execution.
CMD ["uv", "run", "python", "-m", "scripts.run_graph_smoke"]
