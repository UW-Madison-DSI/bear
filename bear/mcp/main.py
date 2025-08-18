import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from fastmcp import Context, FastMCP

from bear.search import SearchEngine

logger = logging.getLogger("bear-mcp")
logger.info("Starting BEAR MCP Server")


@dataclass
class AppContext:
    """Application context with typed dependencies."""

    search_engine: SearchEngine


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """Manage application lifecycle with type-safe context."""
    search_engine = SearchEngine()
    try:
        yield AppContext(search_engine=search_engine)
    finally:
        # Cleanup on shutdown
        del search_engine


mcp = FastMCP(
    name="BEAR MCP",
    instructions="This tool helps you find domain experts for a specific field or research topic.",
    lifespan=app_lifespan,
)


@mcp.tool
async def search_experts(query: str, ctx: Context) -> list[dict[str, Any]]:
    """Search for an author with the given query."""
    results = ctx.request_context.lifespan_context.search_engine.search_author(query=query)
    if not results:
        logging.info("No authors found.")
        return [{"error": "No authors found."}]
    logging.debug(f"Found authors: {results}")
    return results


def main() -> None:
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8001)


if __name__ == "__main__":
    main()
