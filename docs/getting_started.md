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

Copy the example environment file and configure your institutional settings:

```bash
cp example.env .env
```

Edit the `.env` file with your specific configuration. Key settings include:

- Institution identifier (OpenAlex format)
- Embedding model preferences
- Custom data source configurations
- API keys for external services

See the [Config Reference](reference/config.md) for detailed configuration options.

### 4. Start Backend

```bash
docker compose up -d
```

This will start:

- API service: <http://localhost:8000>
- attu GUI for Milvus: <http://localhost:3000>
- Milvus vector database:
  - Endpoint: <http://localhost:19530>
  - Diagnostic Web-UI: <http://localhost:9091/webui/>
- MinIO (internal service)
- etcd (internal service)

### 5. Crawl Academic Data

Crawl data from OpenAlex for your institution:

```bash
uv run bear/crawler.py <your-institution-name>
```

For example, for University of Wisconsin-Madison:

```bash
uv run bear/crawler.py uw-madison
```

### 6. Ingest Data

Process and vectorize the crawled data:

```bash
# Test run first
uv run bear/ingest.py --test

# Full ingest
uv run bear/ingest.py
```

The API will be available at `http://localhost:8000`.

## Testing the Installation

Test your installation with a sample API call:

```bash
curl "http://localhost:8000/search_author?query=data%20science"
```

## Next Steps

- Explore the [API Usage](api_usage.ipynb) for hands-on examples
