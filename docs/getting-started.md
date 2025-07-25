# Getting Started

This guide will help you set up and deploy BEAR for your institution.

## Prerequisites

- Python 3.12 or higher
- Docker and Docker Compose
- UV package manager
- Git

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/JasonLo/bear.git
cd bear
```

### 2. Install Dependencies

BEAR uses UV for dependency management:

```bash
uv sync
```

### 3. Configuration

Copy the example environment file and configure your settings:

```bash
cp example.env .env
```

Edit the `.env` file with your specific configuration. See the [Config Reference](reference/config.md) for detailed configuration options.

### 4. Start Vector Database Backend

BEAR uses Milvus as its vector database. Start the required services:

```bash
docker compose up -d
```

This will start:

- Milvus vector database
- MinIO object storage
- etcd coordination service

### 5. Crawl Academic Data

Crawl data from OpenAlex for your institution:

```bash
uv run bear/crawler.py your-institution-id
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

### 7. Start the API

Start the FastAPI server:

```bash
uv run bear/api.py
```

The API will be available at `http://localhost:8000`.

## Testing the Installation

Test your installation with a sample API call:

```bash
curl "http://localhost:8000/search_author?query=data%20science&top_k=100&since_year=2010&institutions=uw-madison"
```

## Next Steps

- Explore the [Usage Guide](usage.md) for detailed usage instructions
- Check out the [API Reference](reference/api.md) for complete API documentation
- Try the [Interactive Notebooks](notebooks/api_usage.ipynb) for hands-on examples

## Troubleshooting

### Common Issues

1. **Docker services not starting**: Ensure Docker is running and ports are available
2. **UV not found**: Install UV using the official installer
3. **Permission errors**: Check file permissions and Docker group membership
4. **Memory issues**: Ensure sufficient RAM for vector operations

For more help, check the project issues on GitHub or refer to the detailed documentation sections.
