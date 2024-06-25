import os
import time
from milvus_python.evaluate.evaluate_milvus import MilvusEvaluator
from milvus_python.setup.setup_params_milvus import setup_parameters_milvus
from milvus_python.setup.setup_resources_milvus import SetUpMilvusResources
from milvus_python.setup.utils import remove_volume_folder, set_up_log

MILVUS_DOCKER_COMPOSE = "/home/baptvit/Documents/github/banco-de-dados-massivos/milvus/milvus-standalone-docker-compose-gpu.yml"

if __name__ == "__main__":
    # Replace with the path to your Delta Lake file
    medqa_kb = "/home/baptvit/Documents/github/mineracao-dados-massivos/data/med-qa-dataset/medqa_kb"
    #test_dataset_textbook = "/home/baptvit/Documents/github/banco-de-dados-massivos/milvus/milvus-python/milvus_python/test/data/pd_textbook_test_embedding_1000.csv"
    test_dataset_question = "/home/baptvit/Documents/github/banco-de-dados-massivos/milvus/milvus-python/milvus_python/test/data/pd_question_test_embedding_1000.csv"

    list_params = setup_parameters_milvus()

    for params in list_params:
        # Check if /volumns from milvus database is empty
        remove_volume_folder()

        # Up milvus docker-compose up
        os.system(
            f"docker-compose -f {MILVUS_DOCKER_COMPOSE} up -d"
        )

        time.sleep(10)

        # Unpacking the params
        collection_name = params["collection_name"]
        index_name = params["index_name"]
        index_params = params["index_params"]

        # Setup logging for all tests runned
        logger = set_up_log(index_name)

        setup_milvus_resources = SetUpMilvusResources(
            collection_name, index_name, index_params, medqa_kb, logger
        )
        # Run setup - Create collection, Indexs and so on.
        setup_milvus_resources.setup()

        evalutor_milvus = MilvusEvaluator(
            collection_name, index_name, index_params, logger
        )

        # Run queries
        # Evalutute things
        evalutor_milvus.evaluate(test_dataset_question)

        # Delete collection and indexing
        setup_milvus_resources.teardown()

        # Docker compose down
        os.system(
            f"docker-compose -f {MILVUS_DOCKER_COMPOSE} down"
        )
