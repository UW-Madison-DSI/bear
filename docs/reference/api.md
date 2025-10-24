# API Reference

::: bear.api

## Endpoints

The BEAR API provides RESTful endpoints for searching academic resources and authors.

### Base URL

```http
http://localhost:8000
```

### Authentication

Currently, no authentication is required for API access.

### Response Format

All responses are in JSON format. Successful responses return the requested data, while errors return an error object with a message.

### Error Handling

The API uses standard HTTP status codes:

- `200 OK` - Request successful
- `404 Not Found` - No results found
- `422 Unprocessable Entity` - Invalid request parameters
- `500 Internal Server Error` - Server error

## Models

### ResourceSearchResult

Response model for resource search results.

### AuthorSearchResult

Response model for author search results.

### EmbedRequest

Request model for embedding generation.

**Fields:**

- `texts` (list[str]): List of text strings to embed
- `type` (Literal["query", "doc", "raw"]): Type of text, defaults to "query"
  - `"query"`: Adds query prefix if configured
  - `"doc"`: Adds document prefix if configured
  - `"raw"`: No prefix applied (use when you've manually added prefixes)

### EmbedResponse

Response model containing generated embeddings.

**Fields:**

- `embeddings` (list[list[float]]): List of embedding vectors, one per input text

## Usage Examples

### Embedding Generation

Generate embeddings for text using the default embedding model:

```python
import requests

# Embed queries (default behavior)
response = requests.post(
    "http://localhost:8000/embed",
    json={
        "texts": [
            "What is machine learning?",
            "How does neural network work?"
        ]
    }
)
embeddings = response.json()["embeddings"]

# Embed documents
response = requests.post(
    "http://localhost:8000/embed",
    json={
        "texts": ["Research paper abstract text"],
        "type": "doc"
    }
)

# Raw embeddings (no prefix)
response = requests.post(
    "http://localhost:8000/embed",
    json={
        "texts": ["query: custom prefixed text"],
        "type": "raw"
    }
)
```
