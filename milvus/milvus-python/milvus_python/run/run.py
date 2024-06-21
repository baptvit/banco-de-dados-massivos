import os
from milvus_python.evaluate.evaluate_milvus import MilvusEvaluator
from milvus_python.setup.setup_params_milvus import setup_parameters_milvus
from milvus_python.setup.setup_resources_milvus import SetUpMilvusResources
from milvus_python.setup.utils import remove_volume_folder, set_up_log

MILVUS_DOCKER_COMPOSE = "/Users/joaobaptista/Documents/personal/banco-de-dados-massivos/milvus/milvus-standalone-docker-compose-gpu.yml"

if __name__ == "__main__":
    # Replace with the path to your Delta Lake file
    # read_delta_path = "/home/baptvit/Documents/github/banco-de-dados-massivos/milvus/milvus-python/tmp/tmp/transformed_dataset_path/"
    read_delta_path = "/home/baptvit/Documents/github/mineracao-dados-massivos/data/med-qa-dataset/textbook_transfomed_parquet_partition_64"
    list_params = setup_parameters_milvus()

    for params in list_params:
        # Check if /volumns from milvus database is empty
        remove_volume_folder()

        # Up milvus docker-compose up
        os.system(
            f"docker-compose -f {MILVUS_DOCKER_COMPOSE} up -d"
        )

        # Unpacking the params
        collection_name = params["collection_name"]
        index_name = params["index_name"]
        index_params = params["index_params"]

        # # Setup logging for all tests runned
        # logger = set_up_log(index_name)

        # setup_milvus_resources = SetUpMilvusResources(
        #     collection_name, index_name, index_params, read_delta_path, logger
        # )
        # # Run setup - Create collection, Indexs and so on.
        # setup_milvus_resources.setup()

        # evalutor_milvus = MilvusEvaluator(
        #     collection_name, index_name, index_params, logger
        # )

        # # Run queries
        # # Evalutute things
        # evalutor_milvus.evaluate()

        # # Delete collection and indexing
        # setup_milvus_resources.teardown()

        # Docker compose down
        os.system(
            f"docker-compose -f {MILVUS_DOCKER_COMPOSE} down"
        )
