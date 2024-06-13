import time
from milvus_python.setup.setup_params_milvus import (
    DIM,
    EMBEDDING_COLUMN,
    VARCHAR_LENGTH,
)
import pandas as pd
from pymilvus import Collection, CollectionSchema, DataType, FieldSchema
from milvus_python.benchmark.benchmarking import all_in_one_profile
from pymilvus.bulk_writer import LocalBulkWriter, BulkFileType


class DeltaToMilvus:
    """
    A class to insert data from Delta Lake format files into Milvus.
    """

    def __init__(self, collection: Collection, read_path: str, logger):
        """
        Args:
            collection: Collection: An instance of the Milvus Collection to add on.
            read_path: str): Path to read from de delta.
        """
        self.collection = collection
        self.read_path = read_path
        self.logger = logger

    def insert_from_delta(self) -> None:
        """
        Inserts data from a Delta Lake file into the Milvus collection.

        Args:
            delta_path (str): Path to the Delta Lake file.
        """
        delta_df = pd.read_parquet(self.read_path)
        start_time = time.time()
        res = self.collection.insert(delta_df)
        total_time = time.time() - start_time
        self.logger.info(f"Took: {total_time}s to load {res.insert_count} records on colletion")
        return total_time

    # def create_med_qa_schema(self) -> CollectionSchema:
    #     """
    #     This function defines the schema for the MedQA collection in Milvus.

    #     Returns:
    #         A CollectionSchema object representing the schema of the MedQA collection.
    #     """

    #     # Field for MedQA ID
    #     med_qa_id = FieldSchema(
    #         name="med_qa_id",
    #         dtype=DataType.INT64,
    #         description="Unique identifier for each MedQA entry (primary key)",
    #         is_primary=True,
    #         auto_id=True,
    #     )

    #     # Field for sentence text
    #     sentence = FieldSchema(
    #         name="sentence",
    #         dtype=DataType.VARCHAR,
    #         max_length=VARCHAR_LENGTH,
    #         description="The actual text of the sentence in the MedQA entry",
    #     )

    #     # Field for sentence embedding vector
    #     sentence_embedding = FieldSchema(
    #         name=EMBEDDING_COLUMN,
    #         dtype=DataType.FLOAT_VECTOR,
    #         dim=DIM,
    #         description="Dense vector representation of the sentence for similarity search",
    #     )

    #     metadata = FieldSchema(
    #         name="metadata",
    #         dtype=DataType.VARCHAR,
    #         max_length=VARCHAR_LENGTH,
    #         description="Metadata about the text sentence",
    #     )

    #     # Combine fields into the collection schema
    #     return CollectionSchema(
    #         fields=[med_qa_id, sentence, sentence_embedding, metadata],
    #         description="Schema for MedQA collection containing sentence text and embeddings",
    #     )

    # def insert_bulk_data(self) -> None:
    #     # Use `from pymilvus import LocalBulkWriter, BulkFileType`
    #     # when you use pymilvus earlier than 2.4.2

    #     writer = LocalBulkWriter(
    #         schema=self.create_med_qa_schema(),
    #         local_path=self.read_path,
    #         segment_size=512 * 1024 * 1024, # Default value
    #         file_type=BulkFileType.PARQUET
    #     )
    #     writer.commit()
