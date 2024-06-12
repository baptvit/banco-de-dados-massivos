from milvus_python.evaluate.evaluate_milvus import MilvusEvaluator
from milvus_python.setup.setup_params_milvus import setup_parameters_milvus
from milvus_python.setup.setup_resources_milvus import SetUpMilvusResources

if __name__ == "__main__":
    # Replace with the path to your Delta Lake file
    read_delta_path = "/home/baptvit/Documents/github/mineracao-dados-massivos/apps/tmp/transformed_dataset_path"

    list_params = setup_parameters_milvus()

    for params in list_params:
        # TO DO: check if /volumns from milvus database is empty
        # TO DO: up milvus docker-compose up
        # TO DO: run setup
        # TO DO: run queries
        # TO DO: evalutute things
        # TO DO: delete /volumns

        setup_milvus_resources = SetUpMilvusResources(
            params["collection_name"],
            params["index_name"],
            params["index_params"],
            read_delta_path,
        )
        setup_milvus_resources.setup()

        evalutor_milvus = MilvusEvaluator(
            params["collection_name"], params["index_name"], params["index_params"]
        )
        evalutor_milvus.evaluate()
        breakpoint()
