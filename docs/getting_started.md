# Getting Started

BEAR is built for easy institutional deployment. Use this guide to quickly set up a proof-of-concept for semantic search and expert discovery at your university.

## Prerequisites

- Git
- Docker and Docker Compose

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/uw-madison-dsi/bear.git
cd bear
```

### 2. Install Dependencies

BEAR uses [uv](https://docs.astral.sh/uv/) for dependency management:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

### 3. Configuration

Use the CLI to initialize BEAR. Follow the prompts to set up the institution ID, start Docker Compose, run the crawler, and ingest data.

```bash
uv run bear-init
```

## Testing the Installation

### API

The API will be available at `http://localhost:8000`.

Test if API is actually running, go to: <http://localhost:8000>

[API docs](http://localhost:8000/docs)

Test with a sample API call:

```bash
curl "http://localhost:8000/search_author?query=data%20science"
```

## Next Steps

- Explore the [API Usage](api_usage.ipynb) for hands-on examples

## Advanced/Manual setup

If you are using the bear-init CLI, these steps are already handled.

### Manually configure system

see the [Config Reference](reference/config.md) and [example.env](example.env) for detailed configuration options.

### Manually starting backend

```bash
docker compose up -d
```

This will start:

- API service: <http://localhost:8000>
- MCP service: <http://localhost:8001/mcp>
- attu GUI for Milvus: <http://localhost:3000>
- Milvus vector database:
  - Endpoint: <http://localhost:19530>
  - Diagnostic Web-UI: <http://localhost:9091/webui/>
- MinIO (internal service)
- etcd (internal service)

### Manually initialize DB

```bash
uv run bear/db.py
```

### Crawl Academic Data

Crawl data from OpenAlex for your institution:

```bash
# Test run (Crawl for 10 people)
uv run bear/crawler.py --test
```

```bash
# Full crawl
uv run bear/crawler.py
```

### Ingest Data

Process and vectorize the crawled data:

```bash
# Test ingest
uv run bear/ingest.py --test

# Full ingest
uv run bear/ingest.py
```
