from openai import OpenAI

from openalex_search.common import CONFIG
from openalex_search.db import Work

client = OpenAI()


def embed(text: str | list[str]) -> list[list[float]]:
    """Use OpenAI to embed text into a vector representation."""
    response = client.embeddings.create(model=CONFIG.EMBEDDING_MODEL, input=text)
    return [v.embedding for v in response.data]


def embed_work(work: Work) -> Work:
    """Embed a work's title and journal into a vector."""
    work.embedding = embed(str(work))[0]
    return work


def embed_works(works: list[Work], batch_size: int = 1024) -> list[Work]:
    """Embed a list of works in batch."""

    for i in range(0, len(works), batch_size):
        batch = works[i : i + batch_size]
        embeddings = embed([str(work) for work in batch])
        for work, embedding in zip(batch, embeddings):
            work.embedding = embedding
    return works
