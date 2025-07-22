from datetime import datetime
from typing import Any, TypeVar
import json

import httpx
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pymilvus import MilvusClient, DataType
from tqdm import tqdm

from bear.settings import CONFIG, LOGGER

load_dotenv()

# Global Milvus client
_milvus_client = None


def get_milvus_client():
    """Get or create Milvus client."""
    global _milvus_client
    if _milvus_client is None:
        try:
            uri = f"http://{CONFIG.MILVUS_HOST}:{CONFIG.MILVUS_PORT}"
            _milvus_client = MilvusClient(
                uri=uri,
                token=CONFIG.MILVUS_TOKEN if CONFIG.MILVUS_TOKEN else None,
            )
            LOGGER.info(f"Connected to Milvus at {uri}")
        except Exception as e:
            LOGGER.error(f"Failed to connect to Milvus: {e}")
            raise
    return _milvus_client


def create_collection_schema():
    """Create Milvus collection schema for academic works using MilvusClient API."""
    client = get_milvus_client()

    # Create schema with dynamic fields enabled
    schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=False)

    # Add fields to schema
    schema.add_field(field_name="pk", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="id", datatype=DataType.VARCHAR, max_length=512)
    schema.add_field(field_name="doi", datatype=DataType.VARCHAR, max_length=512)
    schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=2048)
    schema.add_field(field_name="display_name", datatype=DataType.VARCHAR, max_length=2048)
    schema.add_field(field_name="journal", datatype=DataType.VARCHAR, max_length=1024)
    schema.add_field(field_name="publication_year", datatype=DataType.INT64)
    schema.add_field(field_name="publication_date", datatype=DataType.VARCHAR, max_length=32)
    schema.add_field(field_name="type", datatype=DataType.VARCHAR, max_length=256)
    schema.add_field(field_name="cited_by_count", datatype=DataType.INT64)
    schema.add_field(field_name="is_retracted", datatype=DataType.BOOL)
    schema.add_field(field_name="is_paratext", datatype=DataType.BOOL)
    schema.add_field(field_name="is_oa", datatype=DataType.BOOL)
    schema.add_field(field_name="pdf_url", datatype=DataType.VARCHAR, max_length=2048)
    schema.add_field(field_name="landing_page_url", datatype=DataType.VARCHAR, max_length=2048)
    schema.add_field(field_name="abstract", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=CONFIG.DEFAULT_EMBEDDING_DIMS)

    return schema


def create_index_params():
    """Create index parameters for the collection."""
    client = get_milvus_client()

    index_params = client.prepare_index_params()

    # Add index for primary key
    index_params.add_index(field_name="pk", index_type="AUTOINDEX")

    # Add index for embedding field
    index_params.add_index(
        field_name="embedding",
        index_type=CONFIG.DEFAULT_INDEX_TYPE,
        metric_type=CONFIG.DEFAULT_METRIC_TYPE,
        params={
            "M": CONFIG.DEFAULT_HNSW_M,
            "efConstruction": CONFIG.DEFAULT_HNSW_EF_CONSTRUCTION,
        },
    )

    return index_params


def init(wipe: bool = False) -> None:
    """Initialize Milvus collection."""
    client = get_milvus_client()
    collection_name = CONFIG.MILVUS_COLLECTION_NAME

    # Drop collection if wipe is requested
    if wipe and client.has_collection(collection_name):
        client.drop_collection(collection_name)
        LOGGER.info(f"Dropped existing collection: {collection_name}")

    # Create collection if it doesn't exist
    if not client.has_collection(collection_name):
        schema = create_collection_schema()
        index_params = create_index_params()

        # Create collection
        client.create_collection(collection_name=collection_name, schema=schema, index_params=index_params)
        LOGGER.info(f"Created collection: {collection_name}")

        # Verify collection is loaded
        load_state = client.get_load_state(collection_name)
        LOGGER.info(f"Collection {collection_name} load state: {load_state}")

    else:
        LOGGER.info(f"Collection {collection_name} already exists")
        # Ensure collection is loaded
        load_state = client.get_load_state(collection_name)
        if load_state["state"] != "Loaded":
            client.load_collection(collection_name)
            LOGGER.info(f"Loaded collection: {collection_name}")


Record = TypeVar("Record", Work, WorkAuthorship, Author)


def push(records: list[Record]) -> None:
    """Push records to Milvus (only Works are supported in vector store)."""
    if not records:
        return

    # Filter only Work records for Milvus storage
    works = [r for r in records if isinstance(r, Work)]
    if not works:
        LOGGER.info("No Work records to push to Milvus")
        return

    client = get_milvus_client()
    collection_name = CONFIG.MILVUS_COLLECTION_NAME

    # Convert works to Milvus format
    entities = []
    for work in tqdm(works, desc="Preparing works for insertion"):
        work_dict = work.to_milvus_dict()
        # Skip works without embeddings
        if not work_dict["embedding"]:
            continue
        entities.append(work_dict)

    if entities:
        try:
            result = client.insert(collection_name=collection_name, data=entities)
            LOGGER.info(f"Inserted {len(entities)} works into Milvus. Insert count: {result['insert_count']}")
        except Exception as e:
            LOGGER.error(f"Failed to insert works into Milvus: {e}")
            raise


def get_author(id: str) -> Author:
    """Get an author by id. Note: Authors are not stored in Milvus in this implementation."""
    # This is a simplified implementation - in a real system you might want to
    # store authors in a separate system or include them in the Milvus schema
    raise NotImplementedError("Author retrieval not implemented in Milvus-only backend")


def search_works(query_embedding: list[float], top_k: int = 10) -> list[dict]:
    """Search for works using vector similarity."""
    client = get_milvus_client()
    collection_name = CONFIG.MILVUS_COLLECTION_NAME

    search_params = {
        "metric_type": CONFIG.DEFAULT_METRIC_TYPE,
        "params": {"ef": max(top_k * 2, 64)},  # ef should be larger than top_k
    }

    results = client.search(
        collection_name=collection_name,
        data=[query_embedding],
        anns_field="embedding",
        search_params=search_params,
        limit=top_k,
        output_fields=[
            "id",
            "doi",
            "title",
            "display_name",
            "journal",
            "publication_year",
            "publication_date",
            "type",
            "cited_by_count",
            "is_retracted",
            "is_paratext",
            "is_oa",
            "pdf_url",
            "landing_page_url",
            "abstract",
        ],
    )

    # Convert results to list of dictionaries
    search_results = []
    for result_set in results:
        for result in result_set:
            result_dict = result["entity"]
            result_dict["score"] = result["distance"]
            search_results.append(result_dict)

    return search_results
