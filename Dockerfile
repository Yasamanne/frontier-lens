FROM python:3.11-slim

# Install uv
RUN pip install --no-cache-dir uv

WORKDIR /app

# Copy dependency files first for layer caching
# (README.md is required: pyproject.toml declares readme = "README.md" and hatchling reads it)
COPY pyproject.toml uv.lock README.md ./

# Copy source so the package can be installed
COPY src/ ./src/

# Install production dependencies only
RUN uv sync --no-dev

# Use the project venv directly at runtime (no `uv run` re-resolve on start)
ENV PATH="/app/.venv/bin:$PATH"

# Copy the rest of the application
COPY app/ ./app/
COPY models.yaml ./

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
  CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://localhost:{os.environ.get(\"PORT\", \"8501\")}/_stcore/health')"

COPY docker-entrypoint.sh ./
RUN chmod +x docker-entrypoint.sh

ENTRYPOINT ["./docker-entrypoint.sh"]
