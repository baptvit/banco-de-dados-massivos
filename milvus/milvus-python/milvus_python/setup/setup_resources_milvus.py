from typing import Any, Dict
from pymilvus import (
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    MilvusException,
)
from milvus_python.setup.setup_params_milvus import CONSISTENCY_LEVEL, DIM, SHARDS_NUM


class SetUpMilvusResources:
    def __init__(
        self,
        consistency_level=CONSISTENCY_LEVEL,
        shards_num=SHARDS_NUM,
        sentence_embedding="sentence_embedding",
        dimension=DIM,
    ) -> None:
        self.sentence_embedding = sentence_embedding
        self.dimension = dimension
        self.shards_num = shards_num
        self.consistency_level = consistency_level

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
            dtype=DataType.STRING,
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
                using=collection_name,
                shards_num=self.shards_num,
                consistency_level=self.consistency_level,
            )
        except MilvusException as e:  # Catch specific Milvus exception
            raise MilvusException(
                f"Error creating collection from {collection_name}: {e}"
            ) from e

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

    def setup(
        self, collection_name: str, index_name: str, index_params: Dict[str, Any]
    ) -> None:
        print("===========================")
        print("")
        print("Setuping collection name: ", collection_name)
        print("Index name: ", index_name)
        print("With index parameters: ", index_params)
        schema = self.create_med_qa_schema()
        collection = self.create_med_qa_collection(collection_name, schema)
        self.create_med_qa_indexs(collection, index_name, index_params)
        print("")
        print("===========================")
