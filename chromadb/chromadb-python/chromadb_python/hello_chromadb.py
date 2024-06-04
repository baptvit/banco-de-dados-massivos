# hello_chromadb.py demonstrates the basic operations of PyMilvus, a Python SDK of Milvus.
# 1. connect to Chromadb
# 2. create collection
# 3. insert data
# 4. create index
# 5. search, query, and hybrid search on entities
# 6. delete entities by PK
# 7. drop collection
import time

import numpy as np
from chromadb import HttpClient

fmt = "\n=== {:30} ===\n"
search_latency_fmt = "search latency = {:.4f}s"
num_entities, dim = 3000, 8

#################################################################################
# 1. connect to ChromaDB
try:
  client = HttpClient(host="localhost", port="8001")
  print("Connected to ChromaDB successfully!")
except Exception as e:
  print(f"Connection failed: {e}")
  exit(1)

breakpoint()

#################################################################################
# 2. Create collection (replace 'hello_chormadb' with your desired name)
collection_name = "hello_chormadb"

try:
  client.create_collection(collection_name)
  print(f"Collection '{collection_name}' created successfully!")
except Exception as e:
  print(f"Collection creation failed: {e}")
breakpoint()

#################################################################################
# 3. Sample data (replace with your data structure)
print(fmt.format("Start inserting entities"))
rng = np.random.default_rng(seed=19530)
entities = [
    # provide the pk field because `auto_id` is set to False
    [str(i) for i in range(num_entities)],
    rng.random(num_entities).tolist(),  # field random, only supports list
    rng.random((num_entities, dim)),    # field embeddings, supports numpy.ndarray and list
]

try:
  hello_chromadb_collection = client.get_collection(collection_name)
  hello_chromadb_collection.add(ids=entities[0], embeddings=entities[2], documents=list(map(str, entities[1])))
  print(f"Data inserted into collection '{collection_name}' successfully!")
except Exception as e:
  print(f"Data insertion failed: {e}")
breakpoint()
# -----------------------------------------------------------------------------
# search based on vector similarity
print(fmt.format("Start searching based on vector similarity"))
start_time = time.time()
results = hello_chromadb_collection.query(
    query_texts=[entities[-1][-1:]], # Chroma will embed this for you
    n_results=3 # how many results to return
)
end_time = time.time()
print(results)
print(search_latency_fmt.format(end_time - start_time))


# -----------------------------------------------------------------------------
# query based on scalar filtering(boolean, int, etc.)
print(fmt.format("Start querying with `random > 0.5`"))

start_time = time.time()
results = hello_chromadb_collection.query(
    where_document={"$gt": 0.5},
    n_results=3 # how many results to return
)

end_time = time.time()