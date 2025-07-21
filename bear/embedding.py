from enum import StrEnum
from typing import Protocol

import httpx

from bear.db import Work
from bear.settings import CONFIG, LOGGER


class Embedder(Protocol):
    def embed(self, text: str | list[str]) -> list[list[float]]: ...


class Provider(StrEnum):
    """Embedding providers."""

    OPENAI = "openai"
    TEXT_EMBEDDING_INFERENCE = "text-embedding-inference"


class OpenAIEmbedder:
    """Embedder using OpenAI's API."""

    def __init__(self, model: str, max_tokens: int) -> None:
        try:
            import openai
        except ImportError:
            raise ImportError("OpenAI package is required for OpenAIEmbedder.")

        self.client = openai.OpenAI()
        self.model = model
        self.max_tokens = max_tokens

    def embed(self, text: str | list[str]) -> list[list[float]]:
        """Use OpenAI to embed text into a vector representation."""

        response = self.client.embeddings.create(model=self.model, input=text)
        return [v.embedding for v in response.data]


class TextEmbeddingInferenceEmbedder:
    """Embedder using Text Embedding Inference API."""

    def __init__(self, model_id: str, base_url: str) -> None:
        self.model_id = model_id
        self.client = httpx.Client(base_url=base_url)
        self._verify_server_match_model()

    def _verify_server_match_model(self) -> None:
        """Verify that the base URL matches the system settings."""

        response = self.client.get("/info")
        response.raise_for_status()
        server_info = response.json()
        if server_info.get("model_id") != self.model_id:
            raise ValueError(
                f"Model ID {self.model_id} does not match server's model ID {server_info.get('model_id')}."
            )

    def embed(self, text: str | list[str]) -> list[list[float]]:
        """Use Text Embedding Inference to embed text into a vector representation."""

        response = self.client.post("/embed", json={"inputs": text, "truncate": True})
        response.raise_for_status()
        return response.json()


def get_embedder(provider: str = CONFIG.DEFAULT_EMBEDDING_PROVIDER, **kwargs) -> Embedder:
    """Get the embedder instance based on configuration."""

    provider = Provider(provider.lower())
    if provider == Provider.OPENAI:
        model = kwargs.get("model", CONFIG.DEFAULT_EMBEDDING_MODEL)
        max_tokens = kwargs.get("max_tokens", CONFIG.DEFAULT_EMBEDDING_MAX_TOKENS)
        return OpenAIEmbedder(model=model, max_tokens=max_tokens)
    elif provider == Provider.TEXT_EMBEDDING_INFERENCE:
        model_id = kwargs.get("model_id", CONFIG.DEFAULT_EMBEDDING_MODEL)
        base_url = kwargs.get("base_url", CONFIG.DEFAULT_EMBEDDING_SERVER_URL)
        return TextEmbeddingInferenceEmbedder(model_id=model_id, base_url=base_url)


def embed_works(works: list[Work], batch_size: int = 100) -> list[Work]:
    """Embed a list of works in batch."""

    embedder = get_embedder()

    for i in range(0, len(works), batch_size):
        LOGGER.info(f"Embedding works {i} to {i + batch_size}")
        batch = works[i : i + batch_size]
        embeddings = embedder.embed([str(work) for work in batch])
        for work, embedding in zip(batch, embeddings):
            work.embedding = embedding
    return works
