FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

LABEL org.opencontainers.image.source=https://github.com/uw-madison-dsi/bear

WORKDIR /app

# Install dependencies
COPY pyproject.toml README.md uv.lock ./
RUN uv sync --locked --no-install-project --no-dev

# install the application code
COPY bear ./bear
RUN uv sync --locked --no-dev
ENV PATH="/app/.venv/bin:$PATH"

CMD bear-api
EXPOSE 8000