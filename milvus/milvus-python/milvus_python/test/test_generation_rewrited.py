import json
import pprint
from google.generativeai import GenerativeModel, configure, list_models
from milvus_python.embedding_model.sentence_embedding_bert_model import (
    BertSentenceEmbedding,
)

# 1 - Read a input json in the data_test_contract_format
# 2 - Vecotrize the rewrited text into vector embbeding using the model
# 3 - Create a output data_test_output.json file with the vector embeeding in json


def main() -> None:
    configure(api_key="")
    with open("./milvus_python/test/data/data_test_example.json") as f:
        d = json.load(f)

    model = GenerativeModel("models/gemini-1.0-pro")

    new_output_list = []
    for line in d:
        prompt = line["model_rewrited_prompt"].replace(
            "<original_sentence>", line["original_sentence"]
        )
        response = model.generate_content(prompt)
        line["rewrited_sentence"] = response.text
        new_output_list.append(line)

    with open(
        "./milvus_python/test/data/data_test_rewrited.json", "w", encoding="utf-8"
    ) as f:
        json.dump(new_output_list, f, indent=4)
    return None


if __name__ == "__main__":
    main()
