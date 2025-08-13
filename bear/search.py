from typing import Any

from cachetools import TTLCache, cached
from pymilvus import MilvusClient

from bear import model
from bear.config import config, logger
from bear.db import get_milvus_client
from bear.embedding import embed_query
from bear.reranker import Reranker, get_reranker
from bear.utils import strip_oa_prefix


@cached(cache=TTLCache(maxsize=3, ttl=24 * 60 * 60))
def load_institution_author_ids(institution_id: str = config.OPENALEX_INSTITUTION_ID) -> set[str]:
    """Load author IDs associated with a specific institution."""

    client = get_milvus_client()
    iterator = client.query_iterator(collection_name="person", filter=f"institution_id == '{institution_id}'", output_fields=["id"], batch_size=1000)
    results = set()
    while True:
        batch = iterator.next()
        if not batch:
            iterator.close()
            break
        ids = {strip_oa_prefix(item["id"]) for item in batch}
        results.update(ids)
    return results


def filter_institution_authors(institution_ids: list[str], results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Filter authors by institution."""

    logger.info(f"Filtering authors for institutions: {institution_ids}")
    logger.info(f"Total results before filtering: {len(results)}")
    acceptable_author_ids = set()
    for id in institution_ids:
        acceptable_author_ids.update(load_institution_author_ids(id))
    filtered_results = [result for result in results if strip_oa_prefix(result["author_id"]) in acceptable_author_ids]
    logger.info(f"Total results after filtering: {len(filtered_results)}")
    return filtered_results


class SearchEngine:
    """Search engine for vector-based similarity search across resources."""

    def __init__(self, client: MilvusClient | None = None, reranker: Reranker | None = None) -> None:
        self.client = client or get_milvus_client()
        self.reranker = reranker or get_reranker()

    def search_resource(
        self,
        resource_name: str,
        query: str,
        top_k: int = 3,
        min_distance: float | None = None,
        since_year: int | None = None,
        author_ids: list[str] | None = None,
        output_fields: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Search and filter for resource using a query.

        Args:
            resource_name: Name of the resource collection to search
            query: Search query string
            top_k: Maximum number of results to return
            min_distance: Minimum distance threshold for results
            since_year: Filter results from this year onwards
            author_ids: Filter results by specific author IDs
            output_fields: Fields to include in output. If None, all fields except embedding

        Returns:
            List of search results sorted by distance (descending)

        Raises:
            ValueError: If resource class is not found in model
        """
        # Build filter conditions
        filter_conditions = ["ignore == false"]
        if since_year is not None:
            filter_conditions.append(f"publication_year >= {since_year}")
        if author_ids is not None:
            filter_conditions.append(f"array_contains_any(author_ids, {author_ids})")
        filter_expr = " and ".join(filter_conditions)

        # Get resource class and validate
        resource_class = getattr(model, resource_name.capitalize(), None)
        if not resource_class:
            raise ValueError(f"Resource class '{resource_name}' not found in model.")

        # Set output fields if not provided
        if output_fields is None:
            output_fields = [field for field in resource_class.model_fields.keys() if field != "embedding"]

        # Prepare search arguments
        search_args = {
            "collection_name": resource_name,
            "data": [embed_query(query)],
            "limit": top_k,
            "output_fields": output_fields,
            "filter": filter_expr,
        }

        # Execute search
        results = self.client.search(**search_args)[0]

        # Apply distance filter if specified
        if min_distance is not None:
            results = [result for result in results if result["distance"] >= min_distance]

        return sorted(results, key=lambda x: x["distance"], reverse=True)

    def search_author(self, query: str, top_k: int = 1000, institutions: list[str] | None = None, **kwargs) -> list[dict]:
        """Search for authors based on a query string."""

        if not institutions:
            institutions = [config.OPENALEX_INSTITUTION_ID]

        resources_sets = {name: self.search_resource(name, query, top_k, **kwargs) for name in model.ALL_RESOURCES_NAMES}
        results = self.reranker.rerank(resources_sets)
        results = filter_institution_authors(institution_ids=institutions, results=results)
        return results
