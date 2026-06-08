import requests
import numpy as np
import pickle
from sklearn.neighbors import NearestNeighbors
import openai
import os
import hashlib

EMBEDDINGS_URL = 'https://storage.googleapis.com/artifacts.gjones-webinar.appspot.com/embeddings.pkl'
EMBEDDINGS_SHA256 = '0331e16d863953ab90d26fa3a2a16fe963990553216fd465d5a0d08f4e002c58'


def verify_embeddings_pickle(data):
    digest = hashlib.sha256(data).hexdigest()
    if digest != EMBEDDINGS_SHA256:
        raise RuntimeError(
            'embeddings.pkl checksum mismatch; refusing to load untrusted pickle'
        )


def download_embeddings_pickle(path):
    response = requests.get(EMBEDDINGS_URL, timeout=30)
    response.raise_for_status()
    verify_embeddings_pickle(response.content)

    with open(path, 'wb') as file:
        file.write(response.content)


def load_embeddings_pickle(path):
    with open(path, 'rb') as file:
        data = file.read()

    verify_embeddings_pickle(data)
    return pickle.loads(data)

# Goal: Provide users with an interface to query our developer docs.
query = "what are the params for scheduling messages?"

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
# To save sometime I've already crawled the Twilio Docs and put the embeddings in this pickle file.
# PICKLE_FILE_PATH = 'embeddings.pkl'
# save the pickle file
pickle_file_path = 'embeddings.pkl'

# Check if the file already exists
if not os.path.exists(pickle_file_path):
    download_embeddings_pickle(pickle_file_path)

saved_embeddings = load_embeddings_pickle(pickle_file_path)
ids, embeddings, metadata = zip(*saved_embeddings)
embeddings_array = np.stack(embeddings)
nn_model = NearestNeighbors(n_neighbors=5, algorithm='ball_tree')
nn_model.fit(embeddings_array)

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
