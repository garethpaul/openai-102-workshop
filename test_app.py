import json
import os
import pickle
import sys
import types

import numpy as np
import pytest


fake_streamlit = types.SimpleNamespace(
    cache_resource=lambda function: function,
    session_state={},
)
sys.modules.setdefault("streamlit", fake_streamlit)

from utils import generate  # noqa: E402


def test_load_embeddings_and_train_model(tmp_path):
    sample_saved_embeddings = [
        (1, np.array([0.1, 0.2]), {"text": "sample text 1"}),
        (2, np.array([0.2, 0.3]), {"text": "sample text 2"}),
    ]
    pickle_path = tmp_path / "embeddings.pkl"
    with pickle_path.open("wb") as file:
        pickle.dump(sample_saved_embeddings, file)

    nn_model, metadata = generate.load_embeddings_and_train_model(pickle_path)

    assert list(metadata) == [
        {"text": "sample text 1"},
        {"text": "sample text 2"},
    ]
    distances, indices = nn_model.kneighbors([[0.1, 0.2]])
    assert distances.shape == (1, 2)
    assert indices.shape == (1, 2)


def test_load_embeddings_and_train_model_rejects_empty_fixtures(tmp_path):
    pickle_path = tmp_path / "embeddings.pkl"
    with pickle_path.open("wb") as file:
        pickle.dump([], file)

    with pytest.raises(ValueError, match="embedding fixture row"):
        generate.load_embeddings_and_train_model(pickle_path)


def test_get_cache_file_does_not_escape_cache_dir(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    cache_file = generate.get_cache_file(str(cache_dir), "../secret")

    assert os.path.commonpath([str(cache_dir), cache_file]) == str(cache_dir)
    assert cache_file.endswith(".json")
    assert ".." not in os.path.basename(cache_file)


def test_get_embeddings_reads_cache_without_api_call(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    cache_file = generate.get_cache_file(str(cache_dir), "sample query")
    cached_payload = [{"embedding": [0.1, 0.2]}]
    with open(cache_file, "w") as file:
        json.dump(cached_payload, file)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("OpenAI API should not be called for cache hits")

    monkeypatch.setattr(generate.openai.Embedding, "create", fail_if_called)

    assert generate.get_embeddings("sample query") == cached_payload


def test_get_top_k_metadata():
    class FakeNearestNeighbors:
        def kneighbors(self, values):
            assert values == [[0.1, 0.2]]
            return np.array([[0.0, 0.1]]), np.array([[0, 1]])

    metadata = [{"text": "sample text 1"}, {"text": "sample text 2"}]

    assert generate.get_top_k_metadata(
        [0.1, 0.2], FakeNearestNeighbors(), metadata
    ) == metadata


def test_create_augmented_query():
    top_k_metadata = [{"text": "sample text 1"}, {"text": "sample text 2"}]

    assert generate.create_augmented_query(
        top_k_metadata, "sample query"
    ) == "sample text 1\n\n---\n\nsample text 2\n\n-----\n\nsample query"


def test_distance_dimension_mismatch():
    with pytest.raises(ValueError):
        generate.euclidean_distance([1.0, 2.0], [1.0])


def test_cosine_similarity_dimension_mismatch():
    with pytest.raises(ValueError):
        generate.cosine_similarity([1.0, 2.0], [1.0])


def test_cosine_similarity_zero_vector():
    with pytest.raises(ValueError):
        generate.cosine_similarity([0.0, 0.0], [1.0, 1.0])


def test_record_estimated_cost_adds_first_and_subsequent_values():
    fake_streamlit.session_state.clear()

    generate._record_estimated_cost(10, 0.001)
    generate._record_estimated_cost(5, 0.001)

    assert fake_streamlit.session_state["cost"] == "$0.0000150000"
