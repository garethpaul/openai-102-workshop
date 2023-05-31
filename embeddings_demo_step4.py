import requests
import numpy as np
import pickle
from sklearn.neighbors import NearestNeighbors
import openai
import os

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
    # Download the pickle file
    pkl_file_download = requests.get(
        'https://storage.googleapis.com/artifacts.gjones-webinar.appspot.com/embeddings.pkl')

    # Save the pickle file
    with open(pickle_file_path, 'wb') as file:
        file.write(pkl_file_download.content)

# Load the pickle file
with open(pickle_file_path, 'rb') as file:
    saved_embeddings = pickle.load(file)
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
