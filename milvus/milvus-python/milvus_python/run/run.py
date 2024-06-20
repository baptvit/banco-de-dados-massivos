import sys
import logging
from milvus_python.evaluate.evaluate_milvus import MilvusEvaluator
from milvus_python.setup.setup_params_milvus import setup_parameters_milvus
from milvus_python.setup.setup_resources_milvus import SetUpMilvusResources


def set_up_log(index_name) -> None:
    logger = logging.getLogger(index_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    # stdout_handler = logging.StreamHandler(sys.stdout)
    # stdout_handler.setLevel(logging.DEBUG)
    # stdout_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(f"./milvus_python/logs/setup/{index_name}.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    # logger.addHandler(stdout_handler)
    return logger


if __name__ == "__main__":
    # Replace with the path to your Delta Lake file
    #read_delta_path = "/home/baptvit/Documents/github/banco-de-dados-massivos/milvus/milvus-python/tmp/tmp/transformed_dataset_path/"
    read_delta_path = "/home/baptvit/Documents/github/mineracao-dados-massivos/data/med-qa-dataset/textbook_transfomed_parquet_partition_64"
    list_params = setup_parameters_milvus()

    for params in list_params:
        # TO DO: check if /volumns from milvus database is empty
        # TO DO: up milvus docker-compose up
        # TO DO: run setup
        # TO DO: run queries
        # TO DO: evalutute things
        # TO DO: delete /volumns
        collection_name = params["collection_name"]
        index_name = params["index_name"]
        index_params = params["index_params"]
        logger = set_up_log(index_name)

        setup_milvus_resources = SetUpMilvusResources(
            collection_name, index_name, index_params, read_delta_path, logger
        )
        setup_milvus_resources.setup()

        evalutor_milvus = MilvusEvaluator(
            collection_name, index_name, index_params, logger
        )
        evalutor_milvus.evaluate()

        setup_milvus_resources.teardown()
