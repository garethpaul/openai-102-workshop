import openai
import os
import json
import math
from uuid import uuid4
import streamlit as st
import pickle
from sklearn.neighbors import NearestNeighbors
import numpy as np
import requests


@st.cache_resource
def load_embeddings_and_train_model(pickle_file_path):
    """
    Load embeddings from the specified pickle file and train a
    NearestNeighbors model.

    Args:
        pickle_file_path (str): The path to the pickle file containing the
        embeddings.

    Returns:
        nn_model (NearestNeighbors): The trained NearestNeighbors model.
        metadata (list): The metadata associated with the embeddings.
    """
    """
    # Check if the file already exists
    if not os.path.exists(pickle_file_path):
        # Download the pickle file
        pkl_file_download = requests.get(
            'https://storage.googleapis.com/artifacts.gjones-webinar.appspot.com/embeddings.pkl')

        # Save the pickle file
        with open(pickle_file_path, 'wb') as file:
            file.write(pkl_file_download.content)
    """
    with open(pickle_file_path, 'rb') as file:
        saved_embeddings = pickle.load(file)
    ids, embeddings, metadata = zip(*saved_embeddings)
    embeddings_array = np.stack(embeddings)
    nn_model = NearestNeighbors(n_neighbors=5, algorithm='ball_tree')
    nn_model.fit(embeddings_array)
    return nn_model, metadata


def cosine_similarity(vector1, vector2):
    dot_product = sum(val1 * val2 for val1, val2 in zip(vector1, vector2))
    norm_vector1 = math.sqrt(sum(val * val for val in vector1))
    norm_vector2 = math.sqrt(sum(val * val for val in vector2))
    similarity = dot_product / (norm_vector1 * norm_vector2)
    return similarity


def euclidean_distance(vector1, vector2):
    if len(vector1) != len(vector2):
        raise ValueError("Vectors must have the same dimensionality.")

    squared_diffs = [(val1 - val2) ** 2 for val1,
                     val2 in zip(vector1, vector2)]
    sum_squared_diffs = sum(squared_diffs)
    distance = math.sqrt(sum_squared_diffs)
    return distance


def manhattan_distance(vector1, vector2):
    if len(vector1) != len(vector2):
        raise ValueError("Vectors must have the same dimensionality.")

    distance = sum(abs(val1 - val2) for val1, val2 in zip(vector1, vector2))
    return distance


def get_embeddings(query, embedding_type='text'):
    """
    Retrieve embeddings for the specified query using OpenAI's Embedding API.

    Args:
        query (str): The text query to get embeddings for.

    Returns:
        np.ndarray: The embeddings of the query.
    """
    # if the query is long then base64 encode it for the filename
    if embedding_type == 'url':
        cache_folder = "url_cache"
        query_id = str(uuid4())
        cache_file = os.path.join(cache_folder, f"{query_id}.json")
    else:
        cache_folder = "cache"
        cache_file = os.path.join(cache_folder, f"{query}.json")

    # Check if the cache file exists
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            cached_response = json.load(f)
        return cached_response

    res = openai.Embedding.create(
        input=[query],
        engine="text-embedding-ada-002"
    )
    # add to total cost
    num_tokens = res['usage']['total_tokens']
    print(f"Number of tokens: {num_tokens}")
    cost_per_1k_token = 0.0004
    cost_per_token = cost_per_1k_token / 1000
    estimated_cost = num_tokens * cost_per_token

    # add the estimated cost to the total cost
    # get the current st_state cost
    st_state = st.session_state
    if 'cost' not in st_state:
        st_state['cost'] = f"${0:.10f}"
    else:
        current_cost = float(st_state['cost'].replace("$", ""))
        updated_cost = current_cost + estimated_cost
        st_state['cost'] = f"${updated_cost:.10f}"

    # Create the cache folder if it doesn't exist
    if not os.path.exists(cache_folder):
        os.makedirs(cache_folder)

    with open(cache_file, "w") as f:
        json.dump(res['data'], f)

    return res['data']


def get_top_k_metadata(embedding, nn_model, metadata):
    """
    Get the top-k metadata entries that are closest to the given embedding.
    Args:
        embedding (np.ndarray): The query embedding.
        nn_model (NearestNeighbors): The trained NearestNeighbors model.
        metadata (list): The metadata associated with the embeddings.
    Returns:
        list: The top-k metadata entries.
    """
    distances, indices = nn_model.kneighbors([embedding])
    return [metadata[i] for i in indices[0]]


def create_augmented_query(top_k_metadata, query):
    """
    Create an augmented query by combining the top-k metadata with the
    original query.

    Args:
        top_k_metadata (list): The top-k metadata entries.
        query (str): The original query text.

    Returns:
        str: The augmented query text.
    """
    contexts = [item['text'] for item in top_k_metadata]
    return "\n\n---\n\n".join(contexts) + "\n\n-----\n\n" + query


def get_model_response(augmented_query):
    """
    Get the model response for the augmented query using OpenAI's
    ChatCompletion API.

    Args:
        augmented_query (str): The augmented query text.

    Returns:
        dict: The model's response.
    """
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
    # add to total cost
    num_tokens = res['usage']['total_tokens']
    cost_per_1k_token = 0.002
    cost_per_token = cost_per_1k_token / 1000
    estimated_cost = num_tokens * cost_per_token
    # add the estimated cost to the total cost
    # get the current st_state cost
    st_state = st.session_state
    if 'cost' not in st_state:
        st_state['cost'] = f"${0:.10f}"
    else:
        current_cost = float(st_state['cost'].replace("$", ""))
        updated_cost = current_cost + estimated_cost
        st_state['cost'] = f"${updated_cost:.10f}"

    return res['choices'][0]


def get_generic_response(query):
    """
    Get a generic response for the query using OpenAI's Completion API.
    Args:
        query (str): The text query to get a response for.
    Returns:
        dict: The model's response.
    """
    res = openai.Completion.create(
        model="text-davinci-003",
        prompt=query,
        temperature=0.6,
        max_tokens=1000,
    )
    # add to total cost
    num_tokens = res['usage']['total_tokens']
    cost_per_1k_token = 0.02
    cost_per_token = cost_per_1k_token / 1000
    estimated_cost = num_tokens * cost_per_token
    # add the estimated cost to the total cost
    # get the current st_state cost
    st_state = st.session_state
    if 'cost' not in st_state:
        st_state['cost'] = f"${0:.10f}"
    else:
        current_cost = float(st_state['cost'].replace("$", ""))
        updated_cost = current_cost + estimated_cost
        st_state['cost'] = f"${updated_cost:.10f}"
    # return the response
    return res['choices'][0]
