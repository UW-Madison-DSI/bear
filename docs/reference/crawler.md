# Crawler Reference

::: bear.crawler

## OpenAlex Data Crawler

The crawler module handles data collection from the OpenAlex API.

## Features

- Institution-specific data crawling
- Rate limiting and retry logic
- Parallel processing support
- Data validation and cleaning

## Usage

```bash
uv run bear/crawler.py <institution-id>
```

## Data Output

Crawled data is saved in parquet format in the `tmp/openalex_data/` directory.
