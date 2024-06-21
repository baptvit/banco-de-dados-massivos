import datetime
import time
import re
import numpy as np
import pandas as pd
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

    def search(self, top_k=10):
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
        self.load_time = self.collection_load()
        self.indexs_name = []
        self.times = []
        self.timestamps = []
        self.top_1_result = []  # If true, the model is write on top 1 else false
        self.top_5_result = []  # If true, the model is write on top 5 results else false
        self.top_10_result = [] # If true, the model is write on top 10 results else false
        self.costs = []
        self.distances = []
        self.query_id_result = []
        for query_id in range(
            10
        ):  # level of diferent types of testing (vector as input and expect value match)
            self.timestamps.append(datetime.datetime.now())
            search_results, time = self.search()
            self.times.append(time)
            match = re.search(r"cost: (\d+)", str(search_results))
            cost = match.group(1)
            self.costs.append(cost)
            self.indexs_name.append(self.index_name)
            retrived_distance = 1
            for num, hits in enumerate(search_results[0]): #top 10 results
                # If is first hit
                top_1 = False
                top_5 = False
                top_10 = False
                retrived_sentence = hits.to_dict()["entity"]["sentence"]
                retrived_distance = hits.to_dict()["distance"]
                if retrived_sentence == EXPECT_OUTPUT and num == 0:
                    top_1 = True
                    top_5 = True
                    top_10 = True
                    distance = retrived_distance
                    break
                elif retrived_sentence == EXPECT_OUTPUT and num <= 5:
                    top_1 = False
                    top_5 = True
                    top_10 = True
                    distance = retrived_distance
                    break
                elif retrived_sentence == EXPECT_OUTPUT and num <= 10:
                    top_1 = False
                    top_5 = False
                    top_10 = True
                    distance = retrived_distance
                    break
                else:
                    top_1 = False
                    top_5 = False
                    top_10 = False
                    distance = 1
                
            self.distances.append(distance)

            self.top_1_result.append(top_1)
            self.top_5_result.append(top_5)
            self.top_10_result.append(top_10)
            self.query_id_result.append(query_id)

            msg = f"Query id {query_id}, Load into memory time {self.load_time}, Top 1: {top_1}, Within Top 5 {top_5}, Within Top 10 {top_10} Query time: {time}s, Distance {retrived_distance} and with cost {cost}"
            self.logger.info(msg)
        self.generate_csv()

    def generate_csv(self) -> None:
        metrics = {
            "timestamp": self.timestamps,
            "index_name": self.indexs_name,
            "memory_load_duration": self.load_time,
            "query_id": self.query_id_result,
            "search_duration": self.times,
            "top1": self.top_1_result,
            "top5": self.top_5_result,
            "top10": self.top_10_result,
            "distance": self.distances,
            "query_cost": self.costs,
        }
        df = pd.DataFrame(metrics)
        df.to_csv(f"./milvus_python/results/evaluation/{self.index_name}/.csv")
