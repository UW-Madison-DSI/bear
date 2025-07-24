from typing import Any

import pandas as pd
from pydantic import BaseModel

from bear.db import search_works, Author
from bear.embedding import embed


def _search(query: str, top_k: int) -> list[dict]:
    """Search for works using a query."""
    query_embedding = embed(query)[0]
    results = search_works(query_embedding, top_k)
    return results


class SearchResults(BaseModel):
    """Search results from Milvus."""
    works: list[dict]

    @classmethod
    def from_raw(cls, raw_results):
        return cls(works=raw_results)

    def _flatten(self) -> pd.DataFrame:
        """Convert results to DataFrame."""
        return pd.DataFrame(self.works)

    def rank_by_score(self) -> pd.DataFrame:
        """Sort results by search score."""
        df = self._flatten()
        return df.sort_values("score", ascending=False).reset_index(drop=True)


def search(query: str, top_k: int = 3, m: int = 1000) -> list[dict]:
    """Search for works using a query.

    Args:
        query (str): The query string.
        top_k (int): The number of results to return.
        m (int): The number of works to search (not used in Milvus implementation).
    
    Returns:
        List of work dictionaries with similarity scores.
    """
    
    # Search for works directly in Milvus
    results = _search(query, top_k=top_k)
    
    # Convert to the expected format for the API
    formatted_results = []
    for result in results:
        # Create a simplified author-like response from work data
        # Since we no longer have author aggregation, we'll return work-based results
        formatted_result = {
            "name": result.get("display_name", "Unknown"),
            "open_alex_url": result.get("id", ""),
            "orcid": None,  # Not available in work-only schema
            "score": result.get("score", 0.0),
            # Additional work information
            "doi": result.get("doi", ""),
            "journal": result.get("journal", ""),
            "publication_year": result.get("publication_year", 0),
            "abstract": result.get("abstract", ""),
        }
        formatted_results.append(formatted_result)
    
    return formatted_results