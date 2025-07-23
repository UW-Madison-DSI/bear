from typing import Any

from dotenv import load_dotenv
from pymilvus import MilvusClient

from bear.config import config, logger
from bear.model import ALL_MODELS

load_dotenv()


def get_milvus_client():
    """Get or create Milvus client."""
    uri = f"http://{config.MILVUS_HOST}:{config.MILVUS_PORT}"
    token = config.MILVUS_TOKEN if config.MILVUS_TOKEN else ""
    return MilvusClient(uri=uri, token=token)


def create_milvus_collection(client: MilvusClient, model: Any) -> None:  # TODO: Better typing with ABC
    """Create a Milvus collection for the given model."""

    if model not in ALL_MODELS:
        raise ValueError(f"Model {model} is not registered in bear.model.ALL_MODELS.")

    collection_name = model.__name__.lower()

    if client.has_collection(collection_name):
        logger.info(f"Collection '{collection_name}' already exists. Skipping creation.")
        return

    # Initialize collection with schema and index parameters
    schema = client.create_schema(auto_id=True, enable_dynamic_field=True)
    index_params = client.prepare_index_params()
    for field_name, field_info in model.model_fields.items():
        assert len(field_info.metadata) == 1, f"Field {field_name} should have exactly one metadata entry."
        milvus_metadata = field_info.metadata[0].json_schema

        if "index_configs" in milvus_metadata:
            index_config = milvus_metadata.pop("index_configs")
            logger.info(f"Adding index for field {field_name} with config {index_config}")
            index_params.add_index(field_name=field_name, **index_config)

        logger.info(f"Adding field {field_name} with schema {milvus_metadata}")
        schema.add_field(field_name=field_name, **milvus_metadata)

    logger.info(f"Creating collection '{collection_name}' with schema and index parameters.")
    client.create_collection(collection_name=collection_name, schema=schema, index_params=index_params)


def init(db_name: str = config.MILVUS_DB_NAME) -> None:
    """Initialize Milvus collection."""

    client = get_milvus_client()

    if db_name not in client.list_databases():
        logger.info(f"Creating database: {db_name}")
        client.create_database(db_name=db_name)

    client.use_database(db_name)

    for model in ALL_MODELS:
        create_milvus_collection(client=client, model=model)


if __name__ == "__main__":
    init()
    logger.info("Milvus collections initialized successfully.")
