FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates

# Install uv
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin:${PATH}"

# Install dependencies first to cache them
COPY pyproject.toml .
COPY README.md .
COPY uv.lock .
RUN uv sync --locked --no-install-project --no-dev

# Copy the application
COPY bear ./bear
COPY api ./api
RUN uv sync --locked --no-dev

# Inject app_type specific commands
CMD ["uv", "run", "fastapi", "run", "api/main.py", "--port", "8000"]
EXPOSE 8000

