# Backend for Embedded Academic Resources (BEAR)

Vector Search Made Simple — Like `WordPress` for Search.

## Setup steps for single institution deployment

1. Install dependencies: `uv sync`.
1. Configure system in `.env`, see [example](example.env) and [config](bear/config.py)
1. Crawl data from OpenAlex with your `institution`: e.g., `uv run bear/crawler.py uw-madison`.
1. Spin up a vector store backend: `docker compose up`.
1. Run ingest: `uv run bear/ingest` (use `--test` to do a test run).
1. Try the api: `http://localhost:8000/search_author?query=data%20science&top_k=100&since_year=2010&institutions=uw-madison`
