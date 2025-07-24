from dotenv import load_dotenv
from pymilvus import MilvusClient

from bear import ALL_RESOURCES, Resource, ResourceType
from bear.config import config, logger

load_dotenv()


def get_milvus_client(db_name: str = config.MILVUS_DB_NAME) -> MilvusClient:
    """Get or create Milvus client."""
    uri = f"http://{config.MILVUS_HOST}:{config.MILVUS_PORT}"
    token = config.MILVUS_TOKEN if config.MILVUS_TOKEN else ""
    client = MilvusClient(uri=uri, token=str(token))
    client.use_database(db_name)
    return client


def create_milvus_collection(client: MilvusClient, model: type[Resource], auto_id: bool = False, enable_dynamic_field: bool = True) -> None:
    """Create a Milvus collection for the given model."""

    if model not in ALL_RESOURCES:
        raise ValueError(f"Model {model} is not registered in bear.model.ALL_MODELS.")

    collection_name = model.__name__.lower()  # Not instantiated, just use the class name

    if client.has_collection(collection_name):
        logger.info(f"Collection '{collection_name}' already exists. Skipping creation.")
        return

    # Initialize collection with schema and index parameters
    schema = client.create_schema(auto_id=auto_id, enable_dynamic_field=enable_dynamic_field)
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


def init(db_name: str = config.MILVUS_DB_NAME, wipe: bool = False) -> None:
    """Initialize Milvus collection."""

    client = get_milvus_client()

    if wipe and db_name in client.list_databases():
        logger.info(f"Wiping database: {db_name}")
        client.use_database(db_name)
        [client.drop_collection(x) for x in client.list_collections()]
        client.drop_database(db_name=db_name)

    if db_name not in client.list_databases():
        logger.info(f"Creating database: {db_name}")
        client.create_database(db_name=db_name)

    client.use_database(db_name)

    for model in ALL_RESOURCES:
        create_milvus_collection(client=client, model=model)


def push(resources: list[ResourceType], db_name: str = config.MILVUS_DB_NAME) -> None:
    """Upsert resources into Milvus. This method is slower but ensures no duplicate IDs."""
    client = get_milvus_client()
    client.use_database(db_name)
    collection_name = resources[0]._name

    if not client.has_collection(collection_name):
        raise ValueError(f"Collection '{collection_name}' does not exist. Please create it first.")

    data = [resource.to_milvus() for resource in resources]
    client.insert(collection_name=collection_name, data=data)
    logger.info(f"Inserted {len(resources)} resources into collection '{collection_name}'.")


if __name__ == "__main__":
    init(wipe=True)  # TODO: Make this configurable
    logger.info("Milvus collections initialized successfully.")
