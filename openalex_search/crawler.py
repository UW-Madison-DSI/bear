import requests

from openalex_search.common import CONFIG, LOGGER
from openalex_search.staging import LocalDumper


def get_page_results(cursor: str) -> tuple[str, list[dict]]:
    """Get a page of results from the OpenAlex API."""

    query = "authorships.institutions.lineage:i135310074,type:types/article,primary_topic.id:t10427"  # A small set of UW-Madison articles ~= 300
    url = f"https://api.openalex.org/works?filter={query}&per-page=100&cursor={cursor}&mailto={CONFIG.CONTACT_EMAIL}"

    response = requests.get(url)
    response.raise_for_status()

    cursor = response.json()["meta"]["next_cursor"]
    results = response.json()["results"]
    return cursor, results


def crawl() -> None:
    cursor = "*"
    all_results = []

    LOGGER.info("Crawling OpenAlex API...")
    while True:
        cursor, results = get_page_results(cursor=cursor)
        if not results:
            break
        all_results.extend(results)

    LOGGER.info(f"Found {len(all_results)} results")
    # Dump to local storage
    dumper = LocalDumper()
    dumper.dump(all_results, "test_articles.parquet")
    LOGGER.info("Data dumped to local storage")


if __name__ == "__main__":
    crawl()
