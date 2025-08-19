import argparse
from pathlib import Path

import pandas as pd

from bear.config import config, logger
from bear.db import push
from bear.embedding import embed_resources
from bear.model import Person, Work


def ingest_work(path: Path, remove_ingested: bool = False) -> None:
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


def ingest_person(path: Path, remove_ingested: bool = False) -> None:
    """Ingest staging person data from a Parquet file to Milvus."""

    logger.info(f"Loading data from {path}")
    df = pd.read_parquet(path)

    logger.info(f"Data loaded with {len(df)} rows.")
    persons = []
    for _, row in df.iterrows():
        try:
            person = Person.from_raw(row.to_dict(), institution_id=config.OPENALEX_INSTITUTION_ID)
            person.embedding = [0, 0]  # Dummy embedding workaround, Milvus must have vector field
            persons.append(person)
        except Exception as e:
            logger.error(f"Error processing row {_}: {e}")

    push(persons)
    logger.info(f"Ingested {len(persons)} persons from {path} into Milvus.")

    if remove_ingested:
        logger.info(f"Removing file {path} after ingestion.")
        path.unlink()


def main() -> None:
    """Main function to run the ingestion."""
    parser = argparse.ArgumentParser(description="Ingest OpenAlex data into Milvus.")
    parser.add_argument("--type", type=str, choices=["work", "person", "all"], default="all", help="Type of data to ingest.")
    parser.add_argument(
        "--path",
        type=str,
        default="",
        help="Path to the directory containing parquet files to ingest. (e.g. tmp/openalex_data/works for --type work, tmp/openalex_data/authors for --type person)",
    )
    parser.add_argument("--test", action="store_true", help="Run in test mode, ingest 10 files.")

    args = parser.parse_args()

    # Default everything
    if not args.path and args.type == "all":
        logger.info("Ingesting works")
        [ingest_work(f, remove_ingested=True) for f in Path("tmp/openalex_data/works").rglob("*.parquet")]
        Path("tmp/openalex_data/works").unlink()  # Wipe parent folder
        logger.info("Ingesting persons")
        [ingest_person(f, remove_ingested=True) for f in Path("tmp/openalex_data/authors").rglob("*.parquet")]
        logger.info("Ingestion complete for all types and removed all intermediate files.")
        return

    # Advanced ingestion
    staging_dir = Path(args.path)
    files = list(staging_dir.rglob("*.parquet"))
    files = files[:10] if args.test else files

    for file in files:
        if args.type == "work":
            ingest_work(file, remove_ingested=True)
        elif args.type == "person":
            ingest_person(file)
        else:
            logger.warning(f"Unknown data type: {args.type}")
    logger.info(f"Ingestion complete for directory: {staging_dir}")


if __name__ == "__main__":
    main()
