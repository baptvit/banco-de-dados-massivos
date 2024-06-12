import logging
import sys
import time
from typing import Any, Dict
from milvus_python.setup.setup_insert_milvus import DeltaToMilvus
from pymilvus import (
    Collection,
    connections,
    CollectionSchema,
    FieldSchema,
    DataType,
    MilvusException,
)

from milvus_python.setup.setup_params_milvus import (
    CONSISTENCY_LEVEL,
    DIM,
    EMBEDDING_COLUMN,
    SHARDS_NUM,
    VARCHAR_LENGTH,
)
from milvus_python.benchmark.benchmarking import all_in_one_profile


class SetUpMilvusResources:
    def __init__(
        self,
        collection_name: str,
        index_name: str,
        index_params: Dict[str, Any],
        read_delta_path: str,
        consistency_level=CONSISTENCY_LEVEL,
        shards_num=SHARDS_NUM,
        sentence_embedding=EMBEDDING_COLUMN,
        dimension=DIM,
    ) -> None:
        self.collection_name = collection_name
        self.index_name = index_name
        self.index_params = index_params
        self.read_delta_path = read_delta_path

        self.sentence_embedding = sentence_embedding
        self.dimension = dimension
        self.shards_num = shards_num
        self.consistency_level = consistency_level

        self.setup_log(self.index_name)

        logging.info("start connecting to Milvus")
        connections.connect("default", host="localhost", port="19530")

    def create_med_qa_schema(self) -> CollectionSchema:
        """
        This function defines the schema for the MedQA collection in Milvus.

        Returns:
            A CollectionSchema object representing the schema of the MedQA collection.
        """

        # Field for MedQA ID
        med_qa_id = FieldSchema(
            name="med_qa_id",
            dtype=DataType.INT64,
            description="Unique identifier for each MedQA entry (primary key)",
            is_primary=True,
            auto_id=True,
        )

        # Field for sentence text
        sentence = FieldSchema(
            name="sentence",
            dtype=DataType.VARCHAR,
            max_length=VARCHAR_LENGTH,
            description="The actual text of the sentence in the MedQA entry",
        )

        # Field for sentence embedding vector
        sentence_embedding = FieldSchema(
            name=self.sentence_embedding,
            dtype=DataType.FLOAT_VECTOR,
            dim=DIM,
            description="Dense vector representation of the sentence for similarity search",
        )

        # Combine fields into the collection schema
        return CollectionSchema(
            fields=[med_qa_id, sentence, sentence_embedding],
            description="Schema for MedQA collection containing sentence text and embeddings",
        )

    @all_in_one_profile
    def create_med_qa_collection(
        self, collection_name: str, schema: CollectionSchema
    ) -> Collection:
        """
        Creates a Milvus collection with the specified name and schema.

        Args:
            collection_name (str): The name of the collection to create.

        Returns:
            A Collection object representing the created Milvus collection.

        Raises:
            Exception: If an error occurs during collection creation.
        """
        try:
            return Collection(
                name=collection_name,
                schema=schema,
                shards_num=self.shards_num,
                consistency_level=self.consistency_level,
            )
        except MilvusException as e:  # Catch specific Milvus exception
            raise MilvusException(
                f"Error creating collection from {collection_name}: {e}"
            ) from e

    @all_in_one_profile
    def create_med_qa_indexs(
        self, collection: Collection, index_name: str, index_params: Dict[str, Any]
    ) -> None:
        """
        Creates an index on the specified field in the provided Milvus collection.

        Args:
            collection (Collection): The Milvus collection to create the index on.
            index_name (str): The name of the index to create.
            index_params (Dict[str, Any]): A dictionary containing index creation parameters.

        Raises:
            Exception: If an error occurs during index creation.
        """

        try:
            collection.create_index(
                index_name=index_name,
                field_name=self.sentence_embedding,
                index_params=index_params,
            )
        except MilvusException as e:  # Catch specific Milvus exception
            raise MilvusException(
                f"Error creating index {index_name} on collection {collection.name}: {e}"
            ) from e

    def setup_log(self, index_name: str) -> None:
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(logging.DEBUG)
        stdout_handler.setFormatter(formatter)

        file_handler = logging.FileHandler(
            f"./milvus_python/logs/setup/{index_name}.log"
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(stdout_handler)

    def setup(self) -> None:

        self.logger.info("===========================")
        self.logger.info("")
        self.logger.info(f"Setuping collection name: {self.collection_name}")
        self.logger.info(f"Index name: {self.index_name}")
        self.logger.info(f"With index parameters: {self.index_params}")

        schema: CollectionSchema = self.create_med_qa_schema()
        collection: Collection = self.create_med_qa_collection(
            self.collection_name, schema
        )
        # TO DO: Add collection data step
        DeltaToMilvus(collection, self.read_delta_path).insert_from_delta()
        self.create_med_qa_indexs(collection, self.index_name, self.index_params)
        self.logger.info("")
        self.logger.info("===========================")
        time.sleep(1)
