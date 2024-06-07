from typing import Any, Dict, List
from memory_profiler import profile

NLIST = 2048  # Number of cluster units MAX 65536
CPU = 1024  # nprobe: Number of units to query
GPU = 1024  # nprob: Number of units to query
m = 0  # Number of factors of product quantization
NBITS = 8  # [Optional] Number of bits in which each low-dimensional vector is stored. 8 by default

M = 4  # Maximum degree of the node, [4, 64]
EFCONSTRUCTION = 8  # 	Search scope	[8, 512]
EF = 32768  # Search scope	[top_k, 32768]
PQM = 0  # Number of factors of product quantization	dim ≡ 0 (mod PQM)
N_TRESS = 1024  # The number of methods of space division.	[1, 1024]
SEARCH_K = 1  # The number of nodes to search. -1 means 5% of the whole data.	{-1} ∪ [top_k, n × n_trees]

DIM = 768  # Dimension of the vector.	[1, 32,768]
SHARDS_NUM = 1  # Number of the shards for the collection to create.	[1,256]
CONSISTENCY_LEVEL = "Strong"  # Consistency level of the collection to create.
# Strong
# Bounded
# Session
# Eventually
# Customized

metric_type_dict = {"Euclidean_distance": "L2", "Inner_product": "IP"}

index_type_dict = {
    "FLAT": {},
    "IVF_FLAT": {"nlist": NLIST, "nprobe": CPU},
    "IVF_SQ8": {"nlist": NLIST, "nprobe": CPU},
    "IVF_PQ": {"nlist": NLIST, "m": m, "nbits": NBITS, "nprobe": CPU},
    "HNSW": {"M": M, "efConstruction": EFCONSTRUCTION, "ef": EF},
    "ANNOY": {"n_trees": N_TRESS, "search_k": SEARCH_K},
    "RHNSW_FLAT": {"M": M, "efConstruction": EFCONSTRUCTION, "PQM": PQM, "ef": EF},
    "RHNSW_PQ": {"M": M, "efConstruction": EFCONSTRUCTION, "PQM": PQM, "ef": EF},
    "RHNSW_SQ": {"M": M, "efConstruction": EFCONSTRUCTION, "ef": EF},
}



@profile
def setup_parameters_milvus() -> List[Dict[str, Any]]:
    """ "index_name, collection_name and index_name"""
    list_parms = []
    for name_index, params_index in index_type_dict.items():
        for name_metric, value_metric in metric_type_dict.items():
            index_name = f"index_{name_index}_{name_metric}"
            collection_name = f"collection_{name_index}_{name_metric}"
            index_params = {
                "metric_type": value_metric,
                "index_type": name_index,
                "params": params_index,
            }

            list_parms.append(
                {
                    "index_name": index_name,
                    "collection_name": collection_name,
                    "index_params": index_params,
                }
            )

    return list_parms


