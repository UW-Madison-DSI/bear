from bear import model
from bear.db import get_milvus_client
from bear.embedding import embed_query


class SearchEngine:
    """Search engine for vector-based similarity search across resources."""

    def __init__(self, client=None):
        self.client = client or get_milvus_client()

    def search(
        self,
        resource_name: str,
        query: str,
        top_k: int = 3,
        min_distance: float | None = None,
        since_year: int | None = None,
        author_id: int | None = None,
        output_fields: list[str] | None = None,
    ) -> list[dict]:
        """Search and filter for resource using a query.

        Args:
            resource_name: Name of the resource collection to search
            query: Search query string
            top_k: Maximum number of results to return
            min_distance: Minimum distance threshold for results
            since_year: Filter results from this year onwards
            author_id: Filter results by specific author ID
            output_fields: Fields to include in output. If None, all fields except embedding

        Returns:
            List of search results sorted by distance (descending)

        Raises:
            ValueError: If resource class is not found in model
        """
        # Build filter conditions
        filter_conditions = ["ignore == false"]
        if since_year:
            filter_conditions.append(f"year >= {since_year}")
        if author_id:
            filter_conditions.append(f"author_id == {author_id}")
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
            "search_params": {"metric_type": resource_class.embedding_config().metric_type},
        }

        # Execute search
        results = self.client.search(**search_args)[0]

        # Apply distance filter if specified
        if min_distance is not None:
            results = [result for result in results if result["distance"] > min_distance]

        return sorted(results, key=lambda x: x["distance"], reverse=True)
