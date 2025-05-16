import tiktoken
from openai import OpenAI

from bear.settings import CONFIG, LOGGER
from bear.db import Work

client = OpenAI()


def embed(text: str | list[str]) -> list[list[float]]:
    """Use OpenAI to embed text into a vector representation."""
    response = client.embeddings.create(model=CONFIG.EMBEDDING_MODEL, input=text)
    return [v.embedding for v in response.data]


def embed_work(work: Work) -> Work:
    """Embed a work's title and journal into a vector."""
    work.embedding = embed(str(work))[0]
    return work


def trim_text(
    text: str, model: str = CONFIG.EMBEDDING_MODEL, max_tokens: int = 1024
) -> str:
    """Trim text to a maximum number of tokens."""
    encoder = tiktoken.encoding_for_model(model)
    tokens = encoder.encode(text)
    trimmed_tokens = tokens[:max_tokens]
    return encoder.decode(trimmed_tokens)


def embed_works(works: list[Work], batch_size: int = 1024) -> list[Work]:
    """Embed a list of works in batch."""

    for i in range(0, len(works), batch_size):
        LOGGER.info(f"Embedding works {i} to {i + batch_size}")
        batch = works[i : i + batch_size]
        embeddings = embed([trim_text(str(work)) for work in batch])
        for work, embedding in zip(batch, embeddings):
            work.embedding = embedding
    return works
