import argparse
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tqdm import tqdm

from bear.settings import CONFIG, LOGGER


def strip_oa_prefix(id: str) -> str:
    """Remove the OpenAlex ID prefix."""
    return id.lstrip("https://openalex.org/")


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=1, max=30),
    retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
    reraise=True,
)
def get_openalex_id(entity_type: str, name: str) -> str:
    """
    Get an OpenAlex ID for a given entity type and search name with retry logic.

    Args:
        entity_type: The type of entity to search for. Must be one of "authors" or "institutions".
        name: The name to search for.

    Example:
        get_openalex_id("authors", "Jason Chor Ming Lo")
        get_openalex_id("institutions", "University of Wisconsin-Madison")
    """
    if entity_type not in ("authors", "institutions"):
        raise ValueError("entity_type must be 'authors' or 'institutions'")

    url = f"https://api.openalex.org/{entity_type}?search={name}"
    if CONFIG.OPENALEX_MAILTO_EMAIL:
        url += f"&mailto={CONFIG.OPENALEX_MAILTO_EMAIL}"

    try:
        response = httpx.get(url)
        response.raise_for_status()
        results = response.json().get("results")

        if not results:
            raise ValueError(f"No {entity_type.rstrip('s')} found for query: {name}")

        LOGGER.info(f"Found: {results[0]['display_name']} ({results[0]['id']})")
        return strip_oa_prefix(results[0]["id"])
    except (httpx.HTTPError, httpx.TimeoutException) as e:
        LOGGER.warning(f"Error retrieving {entity_type} ID: {str(e)}. Retrying...")
        raise


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(min=1, max=120),
    retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
    reraise=True,
)
def _get_page_results(endpoint: str, query: str, cursor: str = "*") -> tuple[str, list[dict[str, Any]]]:
    """Get a page of results from the OpenAlex API with retry logic."""

    url = f"https://api.openalex.org/{endpoint}?filter={query}&per-page=100&cursor={cursor}"

    if CONFIG.OPENALEX_MAILTO_EMAIL:
        url += f"&mailto={CONFIG.OPENALEX_MAILTO_EMAIL}"

    try:
        response = httpx.get(url, timeout=30.0)
        response.raise_for_status()

        cursor = response.json()["meta"]["next_cursor"]
        results = response.json()["results"]
        return cursor, results
    except (httpx.HTTPError, httpx.TimeoutException) as e:
        LOGGER.warning(f"Error retrieving results: {str(e)}. Retrying...")
        raise


def query_openalex(endpoint: str, query: str, limit: int = 0) -> list[dict[str, Any]]:
    """Get all results from the OpenAlex API for a given endpoint and query.

    Args:
        endpoint: The API endpoint to query (e.g., "works", "authors").
        query: The filter query for the API.
        limit: The maximum number of pages (round trips) to retrieve.
               If 0 (default), all pages are retrieved.

    Example:
        ```python
        # Get works authored by a specific institution
        query_openalex("works", "authorships.institutions.lineage:I135310074,type:types/article", limit=5)

        # Get authors affiliated with a specific institution
        query_openalex("authors", "last_known_institutions.id:https://openalex.org/I135310074", limit=3)
        ```
    """
    cursor = "*"
    all_results = []
    round_trips = 0
    while True:
        if limit > 0 and round_trips >= limit:
            LOGGER.warning(
                f"Reached API call limit of {limit} for endpoint '{endpoint}' with query: {query}. "
                "Results will be incomplete."
            )
            break

        cursor, results = _get_page_results(endpoint, query, cursor)
        round_trips += 1

        if not results:
            break
        all_results.extend(results)
        LOGGER.info(f"Retrieved {len(all_results)} results so far for query: {query}")
    return all_results


def _dump(data: list[dict], filename: Path) -> None:
    """Dump data to a file."""

    filename.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(data)
    df.to_parquet(filename, index=False)
    LOGGER.info(f"Dumped {len(data)} records to {filename}")


def crawl(
    institution: str,
    save_path: Path = Path("tmp/openalex_data"),
    author_api_call_limit: int = 0,
    authors_limit: int = 0,
    per_author_work_api_call_limit: int = 0,
) -> None:
    """Crawl the OpenAlex API and dump the results to local storage."""

    save_path.mkdir(parents=True, exist_ok=True)

    # Get all authors affiliated with the institution
    institution_id = get_openalex_id("institutions", institution)

    LOGGER.info(f"Fetching authors for institution ID: {institution_id}")
    query_authors = f"last_known_institutions.id:{institution_id}"
    authors = query_openalex(endpoint="authors", query=query_authors, limit=author_api_call_limit)
    _dump(authors, filename=save_path / "authors.parquet")

    # Get all works authored by the institution's authors
    if authors_limit > 0:
        authors = authors[:authors_limit]

    for author in tqdm(authors):
        query_works = f"authorships.author.id:{strip_oa_prefix(author['id'])}"
        works = query_openalex(endpoint="works", query=query_works, limit=per_author_work_api_call_limit)
        _dump(works, filename=save_path / "works" / f"{strip_oa_prefix(author['id'])}.parquet")


def main():
    parser = argparse.ArgumentParser(description="Crawl OpenAlex API.")
    parser.add_argument(
        "institution",
        type=str,
        help="Enter your institution name.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode with limited API calls.",
    )
    args = parser.parse_args()
    crawl(
        institution=args.institution,
        author_api_call_limit=3 if args.test else 0,
        authors_limit=10 if args.test else 0,
        per_author_work_api_call_limit=3 if args.test else 0,
    )


if __name__ == "__main__":
    main()
