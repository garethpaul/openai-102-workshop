import openai
import os
import json
import math
import hashlib
from uuid import uuid4
import streamlit as st
from sklearn.neighbors import NearestNeighbors
import numpy as np
import requests


def get_cache_file(cache_folder, query):
    """
    Return a cache file path that cannot escape the cache folder.

    Existing simple cache files such as ``cache/pizza.json`` are still read
    when present, but new writes use a stable SHA-256 filename.
    """
    cache_root = os.path.abspath(cache_folder)
    legacy_name = f"{query}.json"
    if os.path.isdir(cache_root):
        with os.scandir(cache_root) as entries:
            for entry in entries:
                entry_path = os.path.abspath(entry.path)
                if (
                    entry.name == legacy_name
                    and os.path.commonpath([cache_root, entry_path]) == cache_root
                    and entry.is_file()
                ):
                    return entry_path

    digest = hashlib.sha256(query.encode("utf-8")).hexdigest()
    return os.path.join(cache_root, f"{digest}.json")


@st.cache_resource
def load_embeddings_and_train_model(json_file_path):
    """
    Load embeddings from the specified JSON file and train a
    NearestNeighbors model.

    Args:
        json_file_path (str): The path to the JSON file containing the
        embeddings.

    Returns:
        nn_model (NearestNeighbors): The trained NearestNeighbors model.
        metadata (list): The metadata associated with the embeddings.
    """
    saved_embeddings = load_embedding_fixture(json_file_path)
    validate_saved_embeddings(saved_embeddings)
    _, embeddings, metadata = zip(*saved_embeddings)
    embeddings_array = np.stack(embeddings)
    n_neighbors = min(5, len(embeddings_array))
    nn_model = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree')
    nn_model.fit(embeddings_array)
    return nn_model, metadata


def load_embedding_fixture(json_file_path):
    try:
        with open(json_file_path, encoding="utf-8") as file:
            saved_embeddings = json.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Embedding fixture not found: {json_file_path}"
        ) from None
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(
            "Embedding fixture must be valid UTF-8 JSON."
        ) from error

    if not isinstance(saved_embeddings, list):
        raise ValueError("Embedding fixture must be a JSON array.")
    return saved_embeddings


def validate_saved_embeddings(saved_embeddings):
    if not saved_embeddings:
        raise ValueError("At least one embedding fixture row is required.")

    expected_dimensions = None
    for index, row in enumerate(saved_embeddings):
        if not isinstance(row, (list, tuple)) or len(row) != 3:
            raise ValueError(
                f"Embedding fixture row {index} must contain id, embedding, and metadata."
            )

        embedding = row[1]
        try:
            dimensions = len(embedding)
        except TypeError:
            raise ValueError(
                f"Embedding fixture row {index} must include a sequence embedding."
            )

        if dimensions == 0:
            raise ValueError(
                f"Embedding fixture row {index} must include at least one dimension."
            )
        for value in embedding:
            if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
                raise ValueError(
                    f"Embedding fixture row {index} embedding values must be numeric finite numbers."
                )
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                raise ValueError(
                    f"Embedding fixture row {index} embedding values must be numeric finite numbers."
                )
            if not math.isfinite(numeric_value):
                raise ValueError(
                    f"Embedding fixture row {index} embedding values must be numeric finite numbers."
                )

        metadata = row[2]
        if (
            not isinstance(metadata, dict)
            or not isinstance(metadata.get("text"), str)
            or not metadata["text"].strip()
        ):
            raise ValueError(
                f"Embedding fixture row {index} metadata must include text."
            )

        if expected_dimensions is None:
            expected_dimensions = dimensions
        elif dimensions != expected_dimensions:
            raise ValueError("Embedding fixture rows must have the same dimensionality.")


def _validate_vector_pair(vector1, vector2):
    try:
        vector_lengths = (len(vector1), len(vector2))
    except TypeError:
        raise ValueError("Vectors must be non-empty numeric finite sequences.")

    if 0 in vector_lengths:
        raise ValueError("Vectors must be non-empty numeric finite sequences.")
    if vector_lengths[0] != vector_lengths[1]:
        raise ValueError("Vectors must have the same dimensionality.")

    for vector in (vector1, vector2):
        for value in vector:
            if (
                isinstance(value, (bool, complex, np.complexfloating))
                or not isinstance(value, (int, float, np.number))
            ):
                raise ValueError("Vectors must be non-empty numeric finite sequences.")
            try:
                numeric_value = float(value)
            except (TypeError, ValueError, OverflowError):
                raise ValueError("Vectors must be non-empty numeric finite sequences.")
            if not math.isfinite(numeric_value):
                raise ValueError("Vectors must be non-empty numeric finite sequences.")


def cosine_similarity(vector1, vector2):
    _validate_vector_pair(vector1, vector2)

    dot_product = sum(val1 * val2 for val1, val2 in zip(vector1, vector2))
    norm_vector1 = math.sqrt(sum(val * val for val in vector1))
    norm_vector2 = math.sqrt(sum(val * val for val in vector2))
    if norm_vector1 == 0 or norm_vector2 == 0:
        raise ValueError("Cosine similarity is undefined for zero vectors.")

    similarity = dot_product / (norm_vector1 * norm_vector2)
    return similarity


def euclidean_distance(vector1, vector2):
    _validate_vector_pair(vector1, vector2)

    squared_diffs = [(val1 - val2) ** 2 for val1,
                     val2 in zip(vector1, vector2)]
    sum_squared_diffs = sum(squared_diffs)
    distance = math.sqrt(sum_squared_diffs)
    return distance


def manhattan_distance(vector1, vector2):
    _validate_vector_pair(vector1, vector2)

    distance = sum(abs(val1 - val2) for val1, val2 in zip(vector1, vector2))
    return distance


def _record_estimated_cost(num_tokens, cost_per_1k_token):
    estimated_cost = num_tokens * (cost_per_1k_token / 1000)
    st_state = st.session_state
    current_cost = float(st_state.get('cost', "$0").replace("$", ""))
    st_state['cost'] = f"${current_cost + estimated_cost:.10f}"


def validate_embedding_response(data):
    if not isinstance(data, list) or not data:
        raise ValueError("Embedding response must contain at least one embedding.")

    expected_dimensions = None
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Embedding response item {index} must be an object.")

        embedding = item.get("embedding")
        if not isinstance(embedding, (list, tuple)) or not embedding:
            raise ValueError(
                f"Embedding response item {index} must include a non-empty embedding."
            )

        for value in embedding:
            if (
                isinstance(value, (bool, complex, np.complexfloating))
                or not isinstance(value, (int, float, np.number))
            ):
                raise ValueError(
                    f"Embedding response item {index} values must be numeric finite numbers."
                )
            try:
                numeric_value = float(value)
            except (TypeError, ValueError, OverflowError):
                raise ValueError(
                    f"Embedding response item {index} values must be numeric finite numbers."
                )
            if not math.isfinite(numeric_value):
                raise ValueError(
                    f"Embedding response item {index} values must be numeric finite numbers."
                )

        dimensions = len(embedding)
        if expected_dimensions is None:
            expected_dimensions = dimensions
        elif dimensions != expected_dimensions:
            raise ValueError("Embedding response items must have the same dimensionality.")


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
        cache_file = get_cache_file(cache_folder, query)

    # Check if the cache file exists
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cached_response = json.load(f)
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise ValueError("Embedding cache must be valid UTF-8 JSON.") from None
        validate_embedding_response(cached_response)
        return cached_response

    res = openai.Embedding.create(
        input=[query],
        engine="text-embedding-ada-002"
    )
    try:
        response_data = res['data']
    except (KeyError, TypeError):
        raise ValueError("Embedding API response must include data.") from None
    validate_embedding_response(response_data)

    # add to total cost
    num_tokens = res['usage']['total_tokens']
    print(f"Number of tokens: {num_tokens}")
    _record_estimated_cost(num_tokens, 0.0004)

    # Create the cache folder if it doesn't exist
    if not os.path.exists(cache_folder):
        os.makedirs(cache_folder)

    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(response_data, f)

    return response_data


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
    validate_query_embedding(embedding, nn_model)
    distances, indices = nn_model.kneighbors([embedding])
    return [metadata[i] for i in indices[0]]


def validate_query_embedding(embedding, nn_model):
    try:
        dimensions = len(embedding)
    except TypeError:
        raise ValueError("Query embedding must be a non-empty numeric sequence.")

    if dimensions == 0:
        raise ValueError("Query embedding must be a non-empty numeric sequence.")

    for value in embedding:
        if (
            isinstance(value, (bool, complex, np.complexfloating))
            or not isinstance(value, (int, float, np.number))
        ):
            raise ValueError("Query embedding values must be numeric finite numbers.")
        try:
            numeric_value = float(value)
        except (TypeError, ValueError, OverflowError):
            raise ValueError("Query embedding values must be numeric finite numbers.")
        if not math.isfinite(numeric_value):
            raise ValueError("Query embedding values must be numeric finite numbers.")

    expected_dimensions = getattr(nn_model, "n_features_in_", None)
    if expected_dimensions is not None and dimensions != expected_dimensions:
        raise ValueError("Query embedding must match the trained model dimensionality.")


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
    _record_estimated_cost(num_tokens, 0.002)

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
    _record_estimated_cost(num_tokens, 0.02)
    # return the response
    return res['choices'][0]
