# Ingest Reference

::: bear.ingest

## Data Ingestion Pipeline

The ingest module processes crawled data and loads it into the vector database.

## Features

- Parquet file processing
- Embedding generation
- Vector database insertion
- Batch processing
- Progress tracking

## Usage

```bash
# Test ingestion
uv run bear/ingest.py --test

# Full ingestion
uv run bear/ingest.py
```

## Process Flow

1. Load parquet files from crawler output
2. Process and clean text data
3. Generate embeddings
4. Insert into Milvus vector database
5. Create searchable indexes
