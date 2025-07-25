import argparse
from pathlib import Path

import pandas as pd

from bear.config import logger
from bear.db import push
from bear.embedding import embed_resources
from bear.model import Work


def ingest(path: Path, remove_ingested: bool = False) -> None:
    """Ingest staging file into Milvus."""

    logger.info(f"Loading data from {path}")
    df = pd.read_parquet(path)

    logger.info(f"Data loaded with {len(df)} rows.")
    works = [Work.from_raw(row.to_dict()) for _, row in df.iterrows()]

    works = embed_resources(works)
    push(works)
    logger.info(f"Ingested {len(works)} works from {path} into Milvus.")

    if remove_ingested:
        logger.info(f"Removing file {path} after ingestion.")
        path.unlink()


def main() -> None:
    """Main function to run the ingestion."""
    parser = argparse.ArgumentParser(description="Ingest OpenAlex data into Milvus.")
    parser.add_argument("--path", type=str, default="tmp/openalex_data/works", help="Path to the directory containing parquet files to ingest.")
    parser.add_argument("--test", action="store_true", help="Run in test mode, ingest 10 files.")

    args = parser.parse_args()
    staging_dir = Path(args.path)
    files = list(staging_dir.rglob("*.parquet"))
    files = files[:10] if args.test else files

    for file in files:
        ingest(file, remove_ingested=True)
    logger.info(f"Ingestion complete for directory: {staging_dir}")


if __name__ == "__main__":
    main()
