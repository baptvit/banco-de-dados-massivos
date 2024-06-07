from milvus_python.setup.setup_params_milvus import  setup_parameters_milvus
from milvus_python.setup.setup_resources_milvus import SetUpMilvusResources

if __name__ == "__main__":
    setup_milvus_resources = SetUpMilvusResources()

    list_params = setup_parameters_milvus()

    
    for params in list_params:
        print("Index name: ", params["index_name"])
        print("Collection name: ", params["collection_name"])
        print("Parameters index: ", params["index_params"])

        # setup_milvus_resources.setup(
        #     params["collection_name"], params["index_name"], params["index_params"]
        # )
