import requests

from openalex_search.common import CONFIG, LOGGER
from openalex_search.staging import LocalDumper


def get_page_results(cursor: str, query: str) -> tuple[str, list[dict]]:
    """Get a page of results from the OpenAlex API."""

    url = f"https://api.openalex.org/works?filter={query}&per-page=100&cursor={cursor}&mailto={CONFIG.CONTACT_EMAIL}"

    response = requests.get(url)
    response.raise_for_status()

    cursor = response.json()["meta"]["next_cursor"]
    results = response.json()["results"]
    return cursor, results


def crawl(query: str, cursor: str = "*", output_prefix: str = "uw-works") -> None:
    """Crawl the OpenAlex API and dump the results to local storage."""

    dumper = LocalDumper()
    all_results = []
    part = 0
    LOGGER.info("Crawling OpenAlex API...")
    while True:
        cursor, results = get_page_results(
            cursor=cursor,
            query=query,
        )
        if not results:
            break
        all_results.extend(results)
        LOGGER.info(f"Retrieved {len(all_results)} results...")

        # Checkpointing
        if len(all_results) % 10000 == 0:
            dumper.dump(all_results, f"{output_prefix}-{part}-{cursor}.parquet")
            part += 1
            all_results = []

    # Dump to local storage
    dumper.dump(all_results, f"{output_prefix}-{part}-last.parquet")
    LOGGER.info("Crawling complete.")


if __name__ == "__main__":
    crawl(
        query="authorships.institutions.lineage:i135310074,type:types/article",
    )
