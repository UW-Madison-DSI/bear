# Usage Guide

This guide covers the main workflows and features of BEAR.

## Data Crawling

### Crawling Institution Data

Use the crawler to fetch academic data from OpenAlex:

```bash
uv run bear/crawler.py <institution-id>
```

The crawler will:

1. Fetch works associated with the institution
2. Download author information
3. Store data in parquet format
4. Handle rate limiting and retries automatically

### Supported Institution IDs

Institution IDs follow the OpenAlex format. Common examples:

- `uw-madison` - University of Wisconsin-Madison
- `mit` - Massachusetts Institute of Technology
- `stanford` - Stanford University

## Data Ingestion

### Basic Ingestion

Process crawled data into the vector database:

```bash
# Test with a small subset
uv run bear/ingest.py --test

# Full ingestion
uv run bear/ingest.py
```

### Ingestion Process

The ingest pipeline:

1. Loads parquet files from the crawler
2. Generates embeddings for academic content
3. Stores vectors in Milvus
4. Creates searchable indexes

## Search Operations

### API Author Search

Search for authors by research interests:

```bash
curl "http://localhost:8000/search_author?query=machine%20learning&top_k=50"
```

Parameters:

- `query`: Search terms
- `top_k`: Number of results (default: 10)
- `since_year`: Filter by publication year
- `institutions`: Filter by institution

### API Work Search

Search for academic works:

```bash
curl "http://localhost:8000/search_works?query=deep%20learning&top_k=100"
```

## API Endpoints

### Health Check

```http
GET /health
```

Returns system status and component health.

### Author Search

```http
GET /search_author
```

Query parameters:

- `query` (required): Search query
- `top_k` (optional): Number of results (1-1000)
- `since_year` (optional): Year filter
- `institutions` (optional): Institution filter

### Work Search

```http
GET /search_works
```

Similar parameters to author search but returns academic works.

## Configuration

### Environment Variables

Key configuration options in `.env`:

```env
# Database
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=bear
POSTGRES_USER=bear
POSTGRES_PASSWORD=password

# Vector Database
MILVUS_HOST=localhost
MILVUS_PORT=19530

# Embedding Model
EMBEDDING_MODEL=text-embedding-3-large
OPENAI_API_KEY=your-key-here

# Crawler
INSTITUTION_ID=your-institution
CRAWL_LIMIT=10000
```

### Advanced Configuration

See [Config Reference](reference/config.md) for all available options.

## Data Management

### Backup and Restore

Backup your vector data:

```bash
# Backup Milvus data
docker compose exec milvus-standalone backup-tool

# Backup raw data
tar -czf bear-data-backup.tar.gz tmp/openalex_data/
```

### Data Updates

Incremental updates:

```bash
# Crawl new data
uv run bear/crawler.py --incremental

# Ingest updates
uv run bear/ingest.py --incremental
```

## Performance Tuning

### Vector Search Optimization

1. **Index Parameters**: Tune HNSW parameters for your data size
2. **Search Parameters**: Adjust `nprobe` for accuracy vs speed
3. **Batch Size**: Optimize ingestion batch sizes

### Resource Allocation

- **Memory**: Allocate sufficient RAM for vector operations
- **Storage**: Use SSD storage for better I/O performance
- **CPU**: Multi-core systems improve ingestion speed

## Monitoring

### Logs

Check application logs:

```bash
# API logs
uv run bear/api.py --log-level debug

# Ingestion logs
uv run bear/ingest.py --verbose
```

### Metrics

Monitor vector database performance:

```bash
# Milvus metrics
curl http://localhost:9091/metrics
```

## Troubleshooting

### Common Issues

#### Slow Search Performance

- Check index status
- Verify resource allocation
- Consider index parameter tuning

#### Memory Issues

- Reduce batch sizes
- Increase available RAM
- Use incremental processing

#### Data Quality Issues

- Validate source data
- Check embedding generation
- Review filtering parameters

For more detailed troubleshooting, see the specific component documentation in the API Reference section.
