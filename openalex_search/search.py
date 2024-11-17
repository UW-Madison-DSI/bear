from sqlmodel import Session, select

from openalex_search.db import Work, engine
from openalex_search.embedding import embed


def search(query: str, top_k: int = 10) -> list[Work]:
    with Session(engine) as session:
        query_embeddings = embed(query)[0]
        y = session.exec(
            select(Work)
            .order_by(Work.embedding.l2_distance(query_embeddings))
            .limit(top_k)
        )
        return list(y.fetchall())
