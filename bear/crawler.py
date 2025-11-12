import argparse
from contextlib import contextmanager
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

from bear.config import config, logger
from bear.utils import strip_oa_prefix

# Global HTTP client with connection pooling
_http_client: httpx.Client | None = None


def get_http_client() -> httpx.Client:
    """Get or create a shared HTTP client with optimized connection pooling."""
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.Client(
            timeout=httpx.Timeout(30.0, connect=10.0),
            limits=httpx.Limits(
                max_connections=100,
                max_keepalive_connections=20,
                keepalive_expiry=30.0,
            ),
            follow_redirects=True,
        )
    return _http_client


def close_http_client() -> None:
    """Close the shared HTTP client."""
    global _http_client
    if _http_client is not None and not _http_client.is_closed:
        _http_client.close()
        _http_client = None


@contextmanager
def http_client_context():
    """Context manager for HTTP client that ensures cleanup."""
    try:
        yield get_http_client()
    finally:
        pass  # Keep connection alive for reuse


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=1, max=30),
    retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
    reraise=True,
)
def get_openalex_authors(author_ids: list[str]) -> list[dict[str, Any]]:
    """Get author records for a list of OpenAlex author IDs with retry logic."""

    client = get_http_client()
    author_ids = [strip_oa_prefix(a) for a in author_ids]
    # for each 50 author IDs, make a single API call to get the author records

    results = []
    for i in range(0, len(author_ids), 50):
        batch_ids = author_ids[i : i + 50]
        # Combine multiple IDs using '|' as required by OpenAlex filters
        # Ensure a large enough page size to return all requested IDs in one response
        url = f"https://api.openalex.org/authors?filter=id:{'|'.join(batch_ids)}&per-page=50"
        logger.info(
            f"Fetching author records for {len(batch_ids)}/{len(author_ids)} IDs: {batch_ids}"
        )
        if config.OPENALEX_MAILTO_EMAIL:
            url += f"&mailto={config.OPENALEX_MAILTO_EMAIL}"
        # Let errors bubble up so tenacity can retry at the function level
        response = client.get(url)
        response.raise_for_status()
        results.extend(response.json().get("results", []))
    return results


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=1, max=30),
    retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
    reraise=True,
)
def query_openalex_id(entity_type: str, name: str) -> str:
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
    if config.OPENALEX_MAILTO_EMAIL:
        url += f"&mailto={config.OPENALEX_MAILTO_EMAIL}"

    try:
        client = get_http_client()
        response = client.get(url)
        response.raise_for_status()
        results = response.json().get("results")

        if not results:
            raise ValueError(f"No {entity_type.rstrip('s')} found for query: {name}")

        logger.info(f"Found: {results[0]['display_name']} ({results[0]['id']})")
        return strip_oa_prefix(results[0]["id"])
    except (httpx.HTTPError, httpx.TimeoutException) as e:
        logger.warning(f"Error retrieving {entity_type} ID: {str(e)}. Retrying...")
        raise


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(min=1, max=120),
    retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
    reraise=True,
)
def _query_page_results(
    endpoint: str, query: str, cursor: str = "*"
) -> tuple[str, list[dict[str, Any]]]:
    """Get a page of results from the OpenAlex API with retry logic."""

    url = f"https://api.openalex.org/{endpoint}?filter={query}&per-page=100&cursor={cursor}"

    if config.OPENALEX_MAILTO_EMAIL:
        url += f"&mailto={config.OPENALEX_MAILTO_EMAIL}"

    try:
        client = get_http_client()
        response = client.get(url)
        response.raise_for_status()

        cursor = response.json()["meta"]["next_cursor"]
        results = response.json()["results"]
        return cursor, results
    except (httpx.HTTPError, httpx.TimeoutException) as e:
        logger.warning(f"Error retrieving results: {str(e)}. Retrying...")
        raise


def query_openalex(
    endpoint: str, query: str, limit: int = 0, save_folder: Path | None = None
) -> list[dict[str, Any]]:
    """Get all results from the OpenAlex API for a given endpoint and query.

    Args:
        endpoint: The API endpoint to query (e.g., "works", "authors").
        query: The filter query for the API.
        limit: The maximum number of pages (round trips) to retrieve.
               If 0 (default), all pages are retrieved.
        save_folder: Optional folder to save results as a Parquet file.

    Example:
        ```python
        # Get works authored by a specific institution
        query_openalex("works", "authorships.institutions.lineage:I135310074,type:types/article", limit=5)

        # Get authors affiliated with a specific institution
        query_openalex("authors", "last_known_institutions.id:https://openalex.org/I135310074", limit=3)
        ```
    """

    if save_folder is not None:
        save_folder.mkdir(parents=True, exist_ok=True)

    cursor = "*"
    all_results = []
    round_trips = 0
    save_counter = 0
    while True:
        if limit > 0 and round_trips >= limit:
            logger.warning(
                f"Reached API call limit of {limit} for endpoint '{endpoint}' with query: {query}. Results will be incomplete."
            )
            break

        cursor, results = _query_page_results(endpoint, query, cursor)
        round_trips += 1

        if not results:
            break
        all_results.extend(results)

        # Save results to Parquet file if specified
        if save_folder and len(all_results) >= 1000:  # Save every 1000 records
            chunk_file = save_folder / f"chunk_{save_counter}.parquet"
            logger.info(f"Saving {len(all_results)} results to {chunk_file}")
            _dump(all_results, chunk_file)
            save_counter += 1
            all_results = []  # Reset for next chunk

        logger.info(f"Retrieved {len(all_results)} results so far for query: {query}")

    # Save last chunk if there are remaining results
    if save_folder and all_results:
        chunk_file = save_folder / f"chunk_{save_counter}.parquet"
        logger.info(f"Saving final {len(all_results)} results to {chunk_file}")
        _dump(all_results, chunk_file)
    return all_results


def _dump(data: list[dict], filename: Path) -> None:
    """Dump data to a file."""

    filename.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(data)
    df.to_parquet(filename, index=False)
    logger.info(f"Dumped {len(data)} records to {filename}")


def crawl_by_institution_id(
    institution_id: str = config.OPENALEX_INSTITUTION_ID,
    save_path: Path = Path("tmp/openalex_data"),
    author_api_call_limit: int = 0,
    authors_limit: int = 0,
    per_author_work_api_call_limit: int = 0,
    skip_pulling_authors: bool = False,
    skip_existing_works: bool = True,
) -> None:
    """Crawl the OpenAlex API and dump the results to local storage."""

    save_path.mkdir(parents=True, exist_ok=True)

    try:
        # Get existing authors if skip_existing is True
        if skip_existing_works and (save_path / "authors").exists():
            existing_authors = [p.name for p in (save_path / "works").glob("*/")]
        else:
            existing_authors = []

        if not skip_pulling_authors:
            # Get all authors affiliated with the institution
            logger.info(f"Fetching authors for institution ID: {institution_id}")
            query_authors = f"last_known_institutions.id:{institution_id}"
            query_openalex(
                endpoint="authors",
                query=query_authors,
                limit=author_api_call_limit,
                save_folder=save_path / "authors",
            )
            authors = pd.read_parquet(save_path / "authors").to_dict(orient="records")
        else:
            # If skipping pulling authors, use existing authors from previous runs
            if (save_path / "authors").exists():
                authors = pd.read_parquet(save_path / "authors").to_dict(
                    orient="records"
                )
            else:
                logger.warning(
                    "Skipping pulling authors, but no existing authors found."
                )
                authors = []

        # Get all works authored by the institution's authors
        if authors_limit > 0:
            authors = authors[:authors_limit]

        if skip_existing_works:
            authors = [
                a for a in authors if strip_oa_prefix(a["id"]) not in existing_authors
            ]

        for author in tqdm(authors):
            try:
                author_id = strip_oa_prefix(author["id"])

                query_works = f"authorships.author.id:{author_id}"
                query_openalex(
                    endpoint="works",
                    query=query_works,
                    limit=per_author_work_api_call_limit,
                    save_folder=save_path / "works" / author_id,
                )
            except Exception as e:
                logger.error(f"Error processing author {author['id']}: {str(e)}")
                continue
    finally:
        close_http_client()


def crawl_by_author_ids(author_ids: list[str], save_path: Path) -> None:
    """Crawl the OpenAlex API for a list of author IDs and dump the results to local storage."""

    save_path.mkdir(parents=True, exist_ok=True)
    author_ids = [strip_oa_prefix(a) for a in author_ids]

    # Get the author records and save them
    try:
        authors = get_openalex_authors(author_ids)
        if authors:
            _dump(authors, save_path / "authors" / "authors.parquet")
    except Exception as e:
        logger.error(f"Error retrieving author records: {str(e)}")

    # Get the works for each author ID
    try:
        for author_id in tqdm(author_ids):
            try:
                query_works = f"authorships.author.id:{author_id}"
                query_openalex(
                    endpoint="works",
                    query=query_works,
                    save_folder=save_path / "works" / author_id,
                )
            except Exception as e:
                logger.error(f"Error processing author {author_id}: {str(e)}")
                continue
    finally:
        close_http_client()


def crawl(
    institution_id: str | None = None,
    author_ids: list[str] | None = None,
    save_path: Path | None = None,
    test_mode: bool = False,
) -> None:
    """Crawl the OpenAlex API with provided institution id or author ids, only one of which should be specified.

    Args:
        institution_id: The institution ID to crawl (required if crawling by institution).
        author_ids: A list of author IDs to crawl (required if crawling by authors).
        save_path: Directory to save the results. Defaults to "tmp/openalex_data".
        test_mode: If True, limits the number of API calls for testing.
    """
    save_path = save_path or Path("tmp/openalex_data")

    if institution_id and author_ids:
        logger.warning(
            "Both institution_id and author_ids provided; please specify only one."
        )
        return

    if not institution_id and not author_ids:
        institution_id = config.OPENALEX_INSTITUTION_ID
        logger.info(f"No IDs provided. Using default institution ID: {institution_id}")

    if institution_id:
        crawl_by_institution_id(
            institution_id=institution_id,
            save_path=save_path,
            author_api_call_limit=3 if test_mode else 0,
            authors_limit=10 if test_mode else 0,
            per_author_work_api_call_limit=3 if test_mode else 0,
        )

    elif author_ids:
        crawl_by_author_ids(
            author_ids=author_ids[:10] if test_mode else author_ids, save_path=save_path
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Crawl the OpenAlex API by institution ID or author IDs."
    )

    parser.add_argument(
        "--institution-id",
        type=str,
        help="Institution ID to crawl (mutually exclusive with --author-ids-file).",
    )
    parser.add_argument(
        "--author-ids-file",
        type=Path,
        help="Path to a text file containing author IDs (one per line).",
    )
    parser.add_argument(
        "--save-path",
        type=Path,
        default=Path("tmp/openalex_data"),
        help="Directory to save results (default: tmp/openalex_data).",
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Limit API calls for testing.",
    )

    args = parser.parse_args()

    # Load author IDs if a file is provided
    author_ids = None
    if args.author_ids_file:
        with open(args.author_ids_file, "r", encoding="utf-8") as f:
            author_ids = [line.strip() for line in f if line.strip()]

    crawl(
        institution_id=args.institution_id,
        author_ids=author_ids,
        save_path=args.save_path,
        test_mode=args.test_mode,
    )


if __name__ == "__main__":
    main()
