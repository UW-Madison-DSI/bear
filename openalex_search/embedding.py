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


def embed_works(works: list[Work]) -> list[Work]:
    """Embed a list of works in a single batch."""

    embeddings = embed([str(work) for work in works])
    for work, embedding in zip(works, embeddings):
        work.embedding = embedding
    return works
