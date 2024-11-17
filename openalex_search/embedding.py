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
    text_to_embed = f"title: {work.title} \njournal: {work.journal}"
    work.embedding = embed(text_to_embed)[0]
    return work
