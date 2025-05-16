from typing import Any

import pandas as pd
from pydantic import BaseModel
from sqlmodel import Session, select

from bear.db import ENGINE, Work, WorkAuthorship, get_author
from bear.embedding import embed


def _search(query: str, top_k: int) -> list[Any]:
    """Search for works using a query."""
    with Session(ENGINE) as session:
        query_embeddings = embed(query)[0]
        if results := session.exec(
            select(Work, WorkAuthorship)
            .join(WorkAuthorship)
            .order_by(Work.embedding.max_inner_product(query_embeddings))
            .limit(top_k)
        ).all():
            return list(results)
        raise ValueError("No results found.")


class SearchResults(BaseModel):
    """Temporary class to hold search results."""

    works: list[Work]
    work_authorships: list[WorkAuthorship]

    @classmethod
    def from_raw(cls, raw_results):
        works, work_authorships = zip(*raw_results)
        return cls(works=works, work_authorships=work_authorships)

    def _flatten(self) -> pd.DataFrame:
        """Flatten the results by appending entity prefix to the keys."""

        # Cast to dict
        works = [work.model_dump() for work in self.works]

        # Add author_id to each work
        for work, authorship in zip(works, self.work_authorships):
            work["author_id"] = authorship.author_id

        return pd.DataFrame(works)

    def rank(
        self, groupby: str = "author_id", aggregate: str = "count", sortby: str = "id"
    ) -> pd.DataFrame:
        """Prototype ranking function with ugly output."""
        df = self._flatten()
        return (
            df.groupby(groupby)
            .agg(aggregate)
            .sort_values(sortby, ascending=False)
            .reset_index()
        )


def search(query: str, top_k: int = 3, m: int = 1000) -> list[dict]:
    """Search for works using a query.

    Args:
        query (str): The query string.
        top_k (int): The number of authors to return.
        m (int): The number of articles to rank.
    """

    results = SearchResults.from_raw(_search(query, top_k=m)).rank().head(top_k)
    ids = results.author_id.to_list()
    scores = results.id.to_list()

    authors = [
        get_author(id).model_dump() | {"score": score} for id, score in zip(ids, scores)
    ]
    return authors
