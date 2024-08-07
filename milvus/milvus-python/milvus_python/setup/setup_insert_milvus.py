import time
import glob
import pandas as pd
from pymilvus import Collection
from multiprocessing import Pool

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
        print(f"Starting inserting in collection: {self.collection}")

        parquet_files = glob.glob(f"{self.read_path}/*.parquet")

        # files = list(map(lambda x: os.path.join(os.path.abspath(self.read_path), x),os.listdir(self.read_path)))
        final_time = 0
        for index, file in enumerate(parquet_files):
            if ".parquet" in file:
                delta_df = pd.read_parquet(file)
                delta_df = delta_df.loc[delta_df["token_sentence"] < 500]
                start_time = time.time()
                res = self.collection.insert(delta_df)
                total_time = time.time() - start_time
                self.logger.info(
                    f"Took: {total_time}s to load {res.insert_count} records on colletion for batch {index}"
                )
                final_time += total_time

        self.logger.info(f"Final Took: {final_time}s to load on colletion")
        print(f"Final inserting in collection: {self.collection}")
        return final_time
    
    # def insert_data(self, file):
    #     if ".parquet" in file:
    #         delta_df = pd.read_parquet(file)
    #         delta_df = delta_df.loc[delta_df["token_sentence"] < 500]
    #         start_time = time.time()
    #         res = self.collection.insert(delta_df)
    #         total_time = time.time() - start_time
    #         self.logger.info(
    #             f"Took: {total_time}s to load {res.insert_count} records on colletion for batch {self.index}"
    #         )
    #         self.final_time += total_time
    #         self.index += 1

    # def insert_from_delta(self) -> None:
    #     """
    #     Inserts data from a Delta Lake file into the Milvus collection.

    #     Args:
    #         delta_path (str): Path to the Delta Lake file.
    #     """
    #     print(f"Starting inserting in collection: {self.collection}")

    #     parquet_files = glob.glob(f"{self.read_path}/*.parquet")

    #     # files = list(map(lambda x: os.path.join(os.path.abspath(self.read_path), x),os.listdir(self.read_path)))
    #     self.final_time = 0
    #     self.index = 0
    #     with Pool(3) as p:
    #         p.map(self.insert_data, parquet_files)
        
    #     self.logger.info(f"Final Took: {self.final_time}s to load on colletion")
    #     print(f"Final inserting in collection: {self.collection}")
    #     return self.final_time
