from typing import Any, Dict, List

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"

EMBEDDING_COLUMN = "embedding_sentence"
NLIST = 4096  # Number of cluster units MAX 65536
NPROBE = 32  # nprobe: Number of units to query
m = 8  # Number of factors of product quantization
NBITS = 4  # [Optional] Number of bits in which each low-dimensional vector is stored. 8 by default
WITH_RAW_DATA = True

M = 32  # Maximum degree of the node, [4, 64]
EFCONSTRUCTION = 256  # 	Search scope	[8, 512]
EF = 2048  # Search scope	[top_k, 32768]
PQM = 8  # Number of factors of product quantization	dim ≡ 0 (mod PQM)
N_TRESS = 512  # The number of methods of space division.	[1, 1024]
SEARCH_K = 10  # The number of nodes to search. -1 means 5% of the whole data.	{-1} ∪ [top_k, n × n_trees]

# GPU
INTERMEDIATE_GRAPH_DEGREE = 32
GRAPH_DEGREE = 32

VARCHAR_LENGTH = 65535
DIM = 768  # Dimension of the vector.	[1, 32,768]
SHARDS_NUM = 1  # Number of the shards for the collection to create.	[1,256]
CONSISTENCY_LEVEL = "Strong"  # Consistency level of the collection to create.
# Strong
# Bounded
# Session
# Eventually
# Customized

metric_type_dict = {
    "Euclidean_distance": "L2",
    "Inner_product": "IP",
    "Cosine_similarity": "COSINE",
}

index_type_dict = {
    "FLAT": {},
    "IVF_FLAT": {"nlist": NLIST, "nprobe": NPROBE},
    "IVF_SQ8": {"nlist": NLIST, "nprobe": NPROBE},
    "IVF_PQ": {"nlist": NLIST, "m": m, "nbits": NBITS, "nprobe": NPROBE},
    "HNSW": {"M": M, "efConstruction": EFCONSTRUCTION, "ef": EF},
    "SCANN": {"nlist": NLIST, "with_raw_data": WITH_RAW_DATA},
}

index_type_dict_gpu = {
    "GPU_CAGRA": {
        "intermediate_graph_degree": INTERMEDIATE_GRAPH_DEGREE,
        "graph_degree": GRAPH_DEGREE,
        "build_algo": "IVF_PQ",
        "cache_dataset_on_device": True,
    },
    "GPU_IVF_FLAT": {"nlist": NLIST, "nprobe": NPROBE},
    "GPU_IVF_PQ": {"nlist": NLIST, "m": m, "nbits": NBITS, "nprobe": NPROBE},
    "GPU_BRUTE_FORCE": {},
}


def setup_parameters_milvus() -> List[Dict[str, Any]]:
    """ "index_name, collection_name and index_name"""
    list_parms = []
    for name_index, params_index in index_type_dict.items():
        for name_metric, value_metric in metric_type_dict.items():
            index_name = f"index_{name_index}__{name_metric}"
            collection_name = f"collection_{name_index}__{name_metric}"
            index_params = {
                "metric_type": value_metric,
                "index_type": name_index,
                "params": params_index,
            }

            list_parms.append(
                {
                    "index_name": index_name.lower(),
                    "collection_name": collection_name.lower(),
                    "index_params": index_params,
                }
            )

    return list_parms
