from datetime import datetime
from itertools import chain
from typing import Any, NewType

import numexpr as ne
import numpy as np
from pydantic import BaseModel, field_validator

from bear.config import logger
from bear.model import Resource

Formula = NewType("Formula", str)


class ResourceScoringConfig(BaseModel):
    """Configuration for scoring resources by author.

    Args:
        resource (Resource): Type of the resource to score.
        formula (Formula): Scoring formula to use for calculating resource scores.
        min_distance (float): Minimum distance threshold for results.
        n_per_author (int): Number of top results to count per author.

    """

    resource: Resource | str
    formula: Formula | str
    min_distance: float = 0.75
    n_per_author: int = 10

    @field_validator("resource", mode="before")
    def convert_str_to_enum(cls, v) -> Resource:
        if isinstance(v, str):
            return Resource(v)
        return v

    @field_validator("formula", mode="before")
    def validate_formula(cls, v) -> Formula:
        if isinstance(v, str):
            return Formula(v)
        return v


class RerankConfig(BaseModel):
    """Configuration for reranking author."""

    configs: list[ResourceScoringConfig]

    def get_scoring_config(self, resource: Resource | str) -> ResourceScoringConfig:
        """Get scoring config for a specific resource type."""
        if isinstance(resource, str):
            resource = Resource(resource)
        for config in self.configs:
            if config.resource == resource:
                return config
        raise ValueError(f"No scoring config found for resource {resource}")


def flatten_result(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten a single result dictionary."""
    flattened = {**result.get("entity", {}), **result}
    flattened.pop("entity", None)

    author_ids = flattened.pop("author_ids", None)
    if not author_ids:
        logger.warning("No author_ids found in the result. Returning flattened result as is.")
        return [flattened]
    return [{**flattened, "author_id": author_id} for author_id in author_ids]


def flatten_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten a list of results."""
    return list(chain.from_iterable(flatten_result(result) for result in results))


def calculate_resource_score(
    results: list[dict[str, Any]],
    config: ResourceScoringConfig,
) -> dict[str, float]:
    """Calculate resource score by each author. Returns: {author_id: score}.

    Args:
        results (list[dict[str, Any]]): List of results to score. Raw results from Milvus search.
        config (ResourceScoringConfig): Configuration containing the scoring formula and parameters.
    """
    if not results:
        logger.info("No results found. Returning empty scores.")
        return {}

    # Filter by min_distance
    flat_results = [r for r in flatten_results(results) if r.get("distance", 0) > config.min_distance]
    if not flat_results:
        logger.info("No results after filtering by min_distance. Returning empty scores.")
        return {}

    # Build arrays for every numeric field in results
    numeric_keys = {k for r in flat_results for k, v in r.items() if isinstance(v, (int, float))}
    arrays = {key: np.array([r.get(key, 0) for r in flat_results], dtype=float) for key in numeric_keys}

    # Compute scores using numexpr safely
    safe_functions = {"log10": np.log10, "sqrt": np.sqrt}
    timing_info = {"current_year": datetime.now().year}
    scores = ne.evaluate(config.formula, local_dict={**arrays, **safe_functions, **timing_info})

    # Sum top-N scores per author
    scores_by_author = {}
    author_ids = np.array([r["author_id"] for r in flat_results])
    for author_id in np.unique(author_ids):
        author_scores = scores[author_ids == author_id]
        top_n = min(config.n_per_author, len(author_scores))
        top_scores = np.partition(author_scores, -top_n)[-top_n:]
        scores_by_author[str(author_id)] = float(np.sum(top_scores))

    return scores_by_author


class Reranker:
    def __init__(self, config: RerankConfig) -> None:
        self.config = config

    def rerank(self, resources_sets: dict[str, list[dict]]) -> list[dict]:
        """Rerank resources by author.

        Args:
            resources_set (dict[str, list[dict]]): Resources set to rerank. e.g. {"work": [{resource}, ...], "grant": [{resource}, ...]}

        Returns:
            dict[str, list[dict]]: Reranked resources. e.g., [{"id": int, "scores": {"total": float, "work": float, ...}}, ...]
        """

        scores = {}
        for name, resources in resources_sets.items():
            scores[name] = calculate_resource_score(resources, config=self.config.get_scoring_config(resource=name))
        return self.group_by_author(scores)

    @staticmethod
    def group_by_author(resource_scores: dict[str, dict[str, float]]) -> list[dict]:
        """Post-process resource scores to group by author."""

        all_author_ids = {author_id for scores in resource_scores.values() for author_id in scores}
        total_scores = {author_id: sum(scores.get(author_id, 0) for scores in resource_scores.values()) for author_id in all_author_ids}
        sorted_authors = sorted(total_scores.items(), key=lambda x: x[1], reverse=True)

        return [
            {
                "author_id": author_id,
                "scores": {
                    "total": total_score,
                    **{resource_type: resource_scores.get(author_id, 0) for resource_type, resource_scores in resource_scores.items()},
                },
            }
            for author_id, total_score in sorted_authors
        ]


def get_reranker(tag: str = "default") -> Reranker:
    """Get default reranker."""

    reranker_settings = {}

    work_scoring = ResourceScoringConfig(
        resource="work",
        formula="distance ** 3 + log10(cited_by_count + 3) + 1 / log10(publication_year + 3)",
        min_distance=0.8,
        n_per_author=10,
    )  # TODO: tune this with new embeddings

    reranker_settings["default"] = RerankConfig(configs=[work_scoring])

    if tag not in reranker_settings:
        raise ValueError(f"Reranker tag {tag} is not supported.")
    config = reranker_settings[tag]
    return Reranker(config)
