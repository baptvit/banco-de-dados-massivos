import time
import re
import numpy as np
from typing import Dict
from milvus_python.evaluate.evaluate_parms import EXPECT_OUTPUT, NEW_VECTOR_TEST
from pymilvus import Collection, connections


class MilvusEvaluator:
    """
    A class to evaluate a Milvus collection using vector search.
    """

    def __init__(self, collection_name: str, index_name: str, params: Dict, logger):
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
        self.logger = logger

        self.logger.info("start connecting to Milvus")
        connections.connect("default", host="localhost", port="19530")

    def collection_load(self) -> int:
        start_time = time.time()
        self.collection.load()
        return time.time() - start_time

    def search(self, top_k=5):
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
        self.search_param = {
            "data": [np.array(NEW_VECTOR_TEST, dtype="float64")],
            "anns_field": "embedding_sentence",
            "param": self.params,
            "limit": top_k,
        }
        start_time = time.time()
        output = self.collection.search(output_fields=["sentence"], **self.search_param)
        return output, (time.time() - start_time)

    def evaluate(self):
        load_time = self.collection_load()
        times = []
        top_1_result = []  # If true, the model is write on top 1 else false
        top_5_result = []  # If true, the model is write on top 5 results else false
        costs = []
        distances = []
        for attemp in range(
            10
        ):  # level of diferent types of testing (vector as input and expect value match)
            search_results, time = self.search()
            times.append(time)
            match = re.search(r"cost: (\d+)", str(search_results))
            cost = match.group(1)
            costs.append(cost)
            retrived_distance = 1
            for num, hits in enumerate(search_results[0]):
                # If is first hit
                top_1 = False
                top_5 = False
                retrived_sentence = hits.to_dict()["entity"]["sentence"]
                retrived_distance = hits.to_dict()["distance"]
                if retrived_sentence == EXPECT_OUTPUT and num == 0:
                    top_1 = True
                    top_5 = True
                    distances.append(retrived_distance)
                    break
                elif retrived_sentence == EXPECT_OUTPUT and num != 0:
                    top_1 = False
                    top_5 = True
                    distances.append(retrived_distance)
                    break
                else:
                    top_1 = False
                    top_5 = False
                    distances.append(retrived_distance)

            top_1_result.append(top_1)
            top_5_result.append(top_5)

            msg = f"Test number {attemp}, Load into memory time {load_time}, Top 1: {top_1}, Within Top 5 {top_5}, Query time: {time}s, Distance {retrived_distance} and with cost {cost}"
            self.logger.info(msg)
