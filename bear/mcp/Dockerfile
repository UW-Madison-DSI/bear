FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

LABEL org.opencontainers.image.source=https://github.com/uw-madison-dsi/bear

WORKDIR /app

# Install node
ARG NODE_VERSION=22
RUN apt update && apt install curl -y
RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash
ENV NVM_DIR=/root/.nvm
RUN bash -c "source $NVM_DIR/nvm.sh && nvm install $NODE_VERSION"
ENTRYPOINT ["bash", "-c", "source $NVM_DIR/nvm.sh && exec \"$@\"", "--"]

# Install dependencies
COPY pyproject.toml README.md uv.lock ./
RUN uv sync --locked --no-install-project --no-dev

# install the application code
COPY bear ./bear
RUN uv sync --locked --no-dev
ENV PATH="/app/.venv/bin:$PATH"

CMD bear-mcp
EXPOSE 6277
EXPOSE 6274
EXPOSE 8000