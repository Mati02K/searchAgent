FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install uv
RUN pip install --no-cache-dir uv

# Set working directory INSIDE container
WORKDIR /agent

# Copy dependency metadata first (layer caching)
COPY pyproject.toml .

# Install dependencies into system Python
RUN uv pip install --system .

# Copy the rest of the code
COPY . .

# Default command: run smoke test
CMD ["python", "-m", "test"]
