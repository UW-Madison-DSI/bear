# Application Dockerfile
FROM python:3.12-slim

WORKDIR /app
EXPOSE 8000

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates

# Install uv
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin:${PATH}"

# Install dependencies first to cache them
COPY pyproject.toml .
COPY README.md .
COPY uv.lock .
RUN uv sync --frozen

# Copy the application
COPY bear ./bear

# Inject app_type specific commands
CMD ["uv", "run", "fastapi", "run", "bear/api.py", "--port", "8000"]

