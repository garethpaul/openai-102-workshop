import os

import openai

from utils.generate import load_embeddings_and_train_model

# Goal: Provide users with an interface to query our developer docs.
query = "what are the params for scheduling messages?"

# Load and validate the explicit local fixture before making a paid API call.
embedding_file_path = os.environ.get("EMBEDDINGS_FILE_PATH", "embeddings.json")
nn_model, metadata = load_embeddings_and_train_model(embedding_file_path)

# 1. We need to convert our query into an embedding
res = openai.Embedding.create(
    input=[query],
    engine="text-embedding-ada-002"
)
query_embedding = res['data'][0]['embedding']

# 2. Next we need to use the Twilio Docs Data to use the NearestNeighbors (similarity) model
#    to find the top-k closest metadata entries to the query embedding
#    See https://colab.research.google.com/drive/1ehvjQylrTece3JQkUxd1-3ncGw9Nh_ra#scrollTo=8ftM8KfU3OZs for more details on crawling and embedding data.
#
# 3. We need to use the metadata to get the Twilio params
distances, indices = nn_model.kneighbors([query_embedding])
neighours = [metadata[i] for i in indices[0]]

# 4.contexts = [item['text'] for item in top_k_metadata]
contexts = [item['text'] for item in neighours]
augmented_query = "\n\n---\n\n".join(contexts) + "\n\n-----\n\n" + query

primer = f"""You are Q&A bot. A highly intelligent system that answers
    user questions based on the information provided by the user above
    each question. If the information cannot be found in the information
    provided by the user, you truthfully say "I don't know".
    """

res = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": primer},
        {"role": "user", "content": augmented_query}
    ]
)

print(res['choices'][0])
