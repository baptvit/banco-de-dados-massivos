import datetime
import time
import pandas as pd
from typing import Any, Dict
from milvus_python.setup.setup_insert_milvus import DeltaToMilvus
from pymilvus import (
    Collection,
    connections,
    CollectionSchema,
    FieldSchema,
    DataType,
    MilvusException,
    utility,
)

from milvus_python.setup.setup_params_milvus import (
    CONSISTENCY_LEVEL,
    DIM,
    EMBEDDING_COLUMN,
    SHARDS_NUM,
    VARCHAR_LENGTH,
)


class SetUpMilvusResources:
    def __init__(
        self,
        collection_name: str,
        index_name: str,
        index_params: Dict[str, Any],
        read_delta_path: str,
        logger,
        consistency_level=CONSISTENCY_LEVEL,
        shards_num=SHARDS_NUM,
        sentence_embedding=EMBEDDING_COLUMN,
        dimension=DIM,
    ) -> None:
        self.collection_name = collection_name
        self.index_name = index_name
        self.index_params = index_params
        self.read_delta_path = read_delta_path
        self.logger = logger

        self.sentence_embedding = sentence_embedding
        self.dimension = dimension
        self.shards_num = shards_num
        self.consistency_level = consistency_level

        self.logger.info("start connecting to Milvus")
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

        token_sentence = FieldSchema(
            name="token_sentence",
            dtype=DataType.INT64,
            description="Count token for each sentence",
        )

        metadata = FieldSchema(
            name="metadata",
            dtype=DataType.VARCHAR,
            max_length=VARCHAR_LENGTH,
            description="Metadata about the text sentence",
        )

        # Combine fields into the collection schema
        return CollectionSchema(
            fields=[med_qa_id, sentence, sentence_embedding, token_sentence, metadata],
            description="Schema for MedQA collection containing sentence text and embeddings",
        )

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

    def create_med_qa_indexs(
        self, collection: Collection, index_name: str, index_params: Dict[str, Any]
    ) -> int:
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
            start_time = time.time()
            status = collection.create_index(
                index_name=index_name,
                field_name=self.sentence_embedding,
                index_params=index_params,
            )
            end_time = time.time()
            if status.code == 0:
                return end_time - start_time
            else:
                raise
        except MilvusException as e:  # Catch specific Milvus exception
            raise MilvusException(
                f"Error creating index {index_name} on collection {collection.name}: {e}"
            ) from e

    def setup(self) -> None:

        self.logger.info("===========================")
        self.logger.info("")
        self.logger.info(f"Setuping collection name: {self.collection_name}")
        self.logger.info(f"Index name: {self.index_name}")
        self.logger.info(f"With index parameters: {self.index_params}")

        self.timestamp = datetime.datetime.now()
        
        schema: CollectionSchema = self.create_med_qa_schema()
        collection: Collection = self.create_med_qa_collection(
            self.collection_name, schema
        )
        self.time_load = 0
        if collection.is_empty:
            self.logger.info(f"Load data from the path: {self.read_delta_path}")
            self.time_load = DeltaToMilvus(
                collection, self.read_delta_path, self.logger
            ).insert_from_delta()

        self.time_index = 0
        if not collection.has_index():
            self.time_index = self.create_med_qa_indexs(
                collection, self.index_name, self.index_params
            )

        self.logger.info("")
        self.logger.info("===========================")
        msg = f"Load time: {self.time_load}, build index time: {self.time_index}"
        self.logger.info(msg)

        self.to_dataframe()

        time.sleep(1)

    def to_dataframe(self):
        metrics = [{
            "timestamp": self.timestamp,
            "index_name": self.index_name,
            "load_datafreme_duration_s": self.time_load,
            "build_index_duration_s": self.time_index,
        }]
        df = pd.DataFrame(metrics)
        df.to_csv(f"/home/baptvit/Documents/github/banco-de-dados-massivos/milvus/milvus-python/milvus_python/results/setup/{self.index_name}.csv")

    def teardown(self) -> None:
        self.logger.info("===========================")
        self.logger.info("")
        self.logger.info(f"Dropping collection name: {self.collection_name}")
        utility.drop_collection(self.collection_name)
        time.sleep(1)
