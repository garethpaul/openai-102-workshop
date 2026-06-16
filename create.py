import json


sample_saved_embeddings = [
    [1, [0.1, 0.2], {"text": "sample text 1"}],
    [2, [0.2, 0.3], {"text": "sample text 2"}],
]

with open("test_embeddings.json", "w", encoding="utf-8") as file:
    json.dump(sample_saved_embeddings, file, indent=2)
    file.write("\n")
