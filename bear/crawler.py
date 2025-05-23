import argparse
from pathlib import Path

import httpx
import pandas as pd
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from bear.settings import CONFIG, LOGGER


def get_openalex_institution_id(name: str) -> str:
    url = f"https://api.openalex.org/institutions?search={name}&mailto={CONFIG.CONTACT_EMAIL}"
    response = httpx.get(url)
    response.raise_for_status()
    results = response.json().get("results")

    if not results:
        raise ValueError(f"No institution found for query: {name}")

    LOGGER.info(f"Found: {results[0]['display_name']} ({results[0]['id']})")
    return results[0]["id"].split("/")[-1]


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(min=1, max=60),
    retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
    reraise=True,
)
def get_page_results(query: str, cursor: str = "*") -> tuple[str, list[dict]]:
    """Get a page of results from the OpenAlex API with retry logic."""

    url = f"https://api.openalex.org/works?filter={query}&per-page=100&cursor={cursor}&mailto={CONFIG.CONTACT_EMAIL}"

    try:
        response = httpx.get(url, timeout=30.0)
        response.raise_for_status()

        cursor = response.json()["meta"]["next_cursor"]
        results = response.json()["results"]
        return cursor, results
    except (httpx.HTTPError, httpx.TimeoutException) as e:
        LOGGER.warning(f"Error retrieving results: {str(e)}. Retrying...")
        raise


def _dump(data: list[dict], filename: Path) -> None:
    """Dump data to a file."""

    filename.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(data)
    df.to_parquet(filename, index=False)
    LOGGER.info(f"Dumped {len(data)} records to {filename}")


def crawl(
    query: str,
    cursor: str = "*",
    output_prefix: str = "works",
    save_path: Path = Path("tmp/openalex_data"),
) -> None:
    """Crawl the OpenAlex API and dump the results to local storage."""

    all_results = []
    part = 0
    total = 0
    LOGGER.info("Crawling OpenAlex API...")
    while True:
        cursor, results = get_page_results(
            query=query,
            cursor=cursor,
        )
        if not results:
            break
        all_results.extend(results)
        total += len(results)
        LOGGER.info(f"Retrieved {total} results...")

        # Checkpointing
        if len(all_results) >= 5000:
            _dump(all_results, save_path / f"{output_prefix}-{part}-{cursor}.parquet")
            part += 1
            all_results = []

    # Dump any remaining results to local storage
    if all_results:
        _dump(all_results, save_path / f"{output_prefix}-{part}-last.parquet")
    LOGGER.info(f"Crawling complete. Total results: {total}")


def main():
    parser = argparse.ArgumentParser(description="Crawl OpenAlex API.")
    parser.add_argument(
        "institution",
        type=str,
        help="Enter your institution name.",
    )
    args = parser.parse_args()
    id = get_openalex_institution_id(args.institution)
    query = f"authorships.institutions.lineage:{id},type:types/article"
    crawl(query=query)


if __name__ == "__main__":
    main()
