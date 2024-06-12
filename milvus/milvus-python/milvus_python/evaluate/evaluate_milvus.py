import logging
import sys
from typing import Dict
import numpy as np
from milvus_python.evaluate.evaluate_parms import EXPECT_OUTPUT, FIX_VECTOR, NEW_VECTOR_TEST
from pymilvus import Collection, connections, SearchResult, Hits, Hit
from milvus_python.benchmark.benchmarking import all_in_one_profile


class MilvusEvaluator:
    """
    A class to evaluate a Milvus collection using vector search.
    """

    def __init__(self, collection_name: str, index_name: str, params: Dict):
        """
        Initialize the class with Milvus client and collection name.

        Args:
            milvus_client (Milvus): An instance of the Milvus client.
            collection_name (str): Name of the Milvus collection.
        """
        self.collection_name = collection_name
        self.collection = Collection(collection_name)
        self.index_name = index_name
        self.params = params

        self.setup_log(index_name)

        logging.info("start connecting to Milvus")
        connections.connect("default", host="localhost", port="19530")

    def setup_log(self, index_name: str) -> None:
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(logging.DEBUG)
        stdout_handler.setFormatter(formatter)

        file_handler = logging.FileHandler(
            f"./milvus_python/logs/evaluate/{index_name}.log"
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(stdout_handler)

    @all_in_one_profile
    def collection_load(self) -> None:
        self.collection.load()

    @all_in_one_profile
    def search(self, top_k=1):
        """
        Performs a vector search in the collection based on a query vector.

        Args:
            query_vector (numpy.ndarray): The query vector to search with.
            top_k (int, optional): Number of top results to return (default: 10).
            params (SearchParameters, optional): Additional search parameters
                (e.g., metric, partition_names). Defaults to None.

        Returns:
            tuple: A tuple containing the following elements:
                - search_results (list): List of search results
                    (each element is a dictionary with "id" and "distance" keys).
                - elapsed_time (float): Time taken for the search in milliseconds.
        """
        self.collection_load()

        search_param = {
            "data": [np.array(NEW_VECTOR_TEST, dtype="float64")],
            "anns_field": "embedding_sentence",
            "param": self.params,
            "limit": top_k,
        }

        return self.collection.search(output_fields=["sentence"], **search_param)

    def evaluate(self):
        search_results: SearchResult = self.search()
        hit_zero: Hits = search_results[0]
        hit: Hit = hit_zero[0]
        assert hit.to_dict()["entity"]["sentence"] == EXPECT_OUTPUT
        breakpoint()
