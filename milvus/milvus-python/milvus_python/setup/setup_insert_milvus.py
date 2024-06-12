import pandas as pd
from pymilvus import Collection
from milvus_python.benchmark.benchmarking import all_in_one_profile


class DeltaToMilvus:
    """
    A class to insert data from Delta Lake format files into Milvus.
    """

    def __init__(self, collection: Collection, read_path: str):
        """
        Args:
            collection: Collection: An instance of the Milvus Collection to add on.
            read_path: str): Path to read from de delta.
        """
        self.collection = collection
        self.read_path = read_path

    @all_in_one_profile
    def insert_from_delta(self) -> None:
        """
        Inserts data from a Delta Lake file into the Milvus collection.

        Args:
            delta_path (str): Path to the Delta Lake file.
        """
        delta_df = pd.read_parquet(self.read_path)

        self.collection.insert(delta_df)
