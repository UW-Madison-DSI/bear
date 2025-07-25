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
