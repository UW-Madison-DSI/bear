from enum import StrEnum
from functools import cache
from typing import Any, Protocol

import httpx
from openai import OpenAI

from bear import ResourceType
from bear.config import EmbeddingConfig, config, logger


class TextType(StrEnum):
    """Type of text to embed."""

    DOC = "doc"
    QUERY = "query"


class Provider(StrEnum):
    """Embedding providers."""

    OPENAI = "openai"
    TEXT_EMBEDDING_INFERENCE = "tei"


class Embedder(Protocol):
    """Protocol for embedding text into vector representations.

    Example:
        ```python
        from bear.config import config
        from bear.embedding import get_embedder

        embedder = get_embedder(config.embedding_config)

        # Show info
        print(embedder.info)

        # Embed a document
        embedder.embed("hi", text_type="doc")

        # Embed a query
        embedder.embed("What is good at cooking?", text_type="query")
        ```
    """

    def embed(self, text: str | list[str], text_type: TextType | str) -> list[list[float]]: ...

    @property
    def info(self) -> dict[str, Any]: ...

    @classmethod
    def from_config(cls, embedding_config: EmbeddingConfig) -> "Embedder": ...


def append_prefix(text: str | list[str], prefix: str) -> list[str]:
    """Append a prefix to the text or each item in the list."""
    if isinstance(text, str):
        return [f"{prefix} {text}"]
    return [f"{prefix} {t}" for t in text]


class OpenAIEmbedder:
    """Embedder using OpenAI's API."""

    def __init__(self, model: str, max_tokens: int, doc_prefix: str = "", query_prefix: str = "", api_key: str | None = None, **kwargs) -> None:
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.doc_prefix = doc_prefix
        self.query_prefix = query_prefix

    @classmethod
    def from_config(cls, embedding_config: EmbeddingConfig) -> "OpenAIEmbedder":
        """Create an OpenAIEmbedder instance from configuration."""
        return cls(
            model=embedding_config.model,
            max_tokens=embedding_config.max_tokens,
            doc_prefix=embedding_config.doc_prefix,
            query_prefix=embedding_config.query_prefix,
            api_key=str(embedding_config.api_key) if embedding_config.api_key else None,
        )

    @property
    def info(self) -> dict[str, Any]:
        """Return information about the OpenAI embedder."""
        return {
            "provider": Provider.OPENAI,
            "model": self.model,
            "dimensions": self.get_dimensions(),
            "doc_prefix": self.doc_prefix,
            "query_prefix": self.query_prefix,
        }

    @cache
    def get_dimensions(self) -> int:
        """Get the dimensions of the embedding model."""
        response = self.client.embeddings.create(model=self.model, input=["test"])
        return len(response.data[0].embedding)

    def embed(self, text: str | list[str], text_type: TextType | str) -> list[list[float]]:
        """Use OpenAI to embed text into a vector representation."""

        if isinstance(text_type, str):
            text_type = TextType(text_type)

        assert text_type in (TextType.DOC, TextType.QUERY), "text_type must be either 'doc' or 'query'"
        if text_type == TextType.DOC and self.doc_prefix:
            text = append_prefix(text, self.doc_prefix)
        elif text_type == TextType.QUERY and self.query_prefix:
            text = append_prefix(text, self.query_prefix)

        response = self.client.embeddings.create(model=self.model, input=text)
        return [v.embedding for v in response.data]


class TEIEmbedder:
    """Embedder using Text Embedding Inference API (via OpenAI python client)."""

    def __init__(self, model: str, max_tokens: int, base_url: str, api_key: str = "", doc_prefix: str = "", query_prefix: str = "", **kwargs) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.base_url = base_url
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.doc_prefix = doc_prefix
        self.query_prefix = query_prefix
        self._verify_server_match_model()

    @classmethod
    def from_config(cls, embedding_config: EmbeddingConfig) -> "TEIEmbedder":
        """Create a TEIEmbedder instance from configuration."""
        return cls(
            model=embedding_config.model,
            max_tokens=embedding_config.max_tokens,
            base_url=embedding_config.server_url,
            api_key=str(embedding_config.api_key) if embedding_config.api_key else "",
            doc_prefix=embedding_config.doc_prefix,
            query_prefix=embedding_config.query_prefix,
        )

    @property
    def info(self) -> dict[str, Any]:
        """Return information about the TEI embedder."""
        return {
            "provider": Provider.TEXT_EMBEDDING_INFERENCE,
            "model": self.model,
            "max_tokens": self.max_tokens,
            "dimensions": self.get_dimensions(),
            "doc_prefix": self.doc_prefix,
            "query_prefix": self.query_prefix,
        }

    @cache
    def _verify_server_match_model(self) -> None:
        """Verify that the base URL matches the system settings."""

        with httpx.Client(base_url=self.base_url) as client:
            response = client.get("/info")
            response.raise_for_status()
            server_info = response.json()

        if server_info.get("model_id") != self.model:
            raise ValueError(f"Model ID {self.model} does not match server's model ID {server_info.get('model_id')}.")

        if server_info.get("max_input_length") < config.DEFAULT_EMBEDDING_MAX_TOKENS:
            raise ValueError(
                f"Server's max input length {server_info.get('max_input_length')} is less than configured max tokens {config.DEFAULT_EMBEDDING_MAX_TOKENS}."
            )

    @cache
    def get_dimensions(self) -> int:
        """Get the dimensions of the embedding model."""
        response = self.client.embeddings.create(model=self.model, input=["test"])
        return len(response.data[0].embedding)

    def embed(self, text: str | list[str], text_type: TextType | str) -> list[list[float]]:
        """Use Text Embedding Inference to embed text into a vector representation."""

        if isinstance(text_type, str):
            text_type = TextType(text_type)

        if text_type == TextType.DOC and self.doc_prefix:
            text = append_prefix(text, self.doc_prefix)
        elif text_type == TextType.QUERY and self.query_prefix:
            text = append_prefix(text, self.query_prefix)

        response = self.client.embeddings.create(model=self.model, input=text)
        return [v.embedding for v in response.data]


def get_embedder(embedding_config: EmbeddingConfig = config.embedding_config) -> Embedder:
    """Get the embedder instance based on configuration."""
    if embedding_config.provider == "openai":
        return OpenAIEmbedder.from_config(embedding_config)
    elif embedding_config.provider == "tei":
        return TEIEmbedder.from_config(embedding_config)
    raise ValueError(f"Unknown embedding provider: {embedding_config.provider}")


def embed(
    resources: list[ResourceType], batch_size: int = 256, embedding_config: EmbeddingConfig = config.embedding_config, embedding_field: str = "embedding"
) -> list[ResourceType]:
    """Embed a list of resources in batch."""

    embedder = get_embedder(embedding_config)
    logger.info(f"Using embedder: {embedder.info}")
    for i in range(0, len(resources), batch_size):
        logger.info(f"Embedding resources {i} to {i + batch_size}")
        batch = resources[i : i + batch_size]
        embeddings = embedder.embed(text=[str(resource) for resource in batch], text_type=TextType.DOC)
        for resource, embedding in zip(batch, embeddings):
            setattr(resource, embedding_field, embedding)
    return resources
