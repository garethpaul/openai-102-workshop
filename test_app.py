import json
import math
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

from components import recommendations  # noqa: E402
from utils import generate, token  # noqa: E402


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


def test_recommend_product_uses_customer_relative_nearest_industry():
    customers = [{"customer_id": 7, "industry": "Telecommunications"}]
    embeddings = {
        "E-commerce": [{"embedding": [1.0, 0.0]}],
        "Telecommunications": [{"embedding": [0.0, 1.0]}],
    }
    products = {
        "E-commerce": ["commerce product"],
        "Telecommunications": ["telecom product"],
    }

    product, scores = recommendations.recommend_product(
        7, customers, embeddings, products
    )

    assert product == "telecom product"
    assert scores["E-commerce"]["Telecommunications"] == 0.0
    assert scores["Telecommunications"]["Telecommunications"] == 1.0


def test_recommend_product_falls_back_to_product_backed_industry():
    with open("customer_profiles.json", encoding="utf-8") as file:
        customers = json.load(file)
    with open("industry_embeddings.json", encoding="utf-8") as file:
        embeddings = json.load(file)
    products = {
        "E-commerce": ["Twilio Engage"],
        "Telecommunications": ["Programmable Messaging"],
    }

    product, scores = recommendations.recommend_product(
        5,
        customers,
        embeddings,
        products,
        choose_product=lambda choices: choices[0],
    )

    assert product == "Programmable Messaging"
    assert scores["Technology"]["Technology"] == pytest.approx(1.0)
    assert (
        scores["Technology"]["Telecommunications"]
        > scores["Technology"]["E-commerce"]
    )


@pytest.mark.parametrize(
    ("customer_id", "customers", "embeddings", "products"),
    [
        (99, [], {"E-commerce": [{"embedding": [1.0, 0.0]}]}, {}),
        (1, [{"customer_id": 1, "industry": "Missing"}], {}, {}),
        (
            1,
            [{"customer_id": 1, "industry": "Healthcare"}],
            {"Healthcare": [{"embedding": [1.0, 0.0]}]},
            {},
        ),
        (
            1,
            [{"customer_id": 1, "industry": "Healthcare"}],
            {"Healthcare": [{"embedding": [1.0, 0.0]}]},
            {"Healthcare": "not a product list"},
        ),
    ],
)
def test_recommend_product_returns_none_for_unavailable_inputs(
    customer_id, customers, embeddings, products
):
    product, scores = recommendations.recommend_product(
        customer_id, customers, embeddings, products
    )

    assert product is None
    assert isinstance(scores, dict)


def test_load_embeddings_and_train_model_rejects_empty_fixtures(tmp_path):
    pickle_path = tmp_path / "embeddings.pkl"
    with pickle_path.open("wb") as file:
        pickle.dump([], file)

    with pytest.raises(ValueError, match="embedding fixture row"):
        generate.load_embeddings_and_train_model(pickle_path)


def test_load_embeddings_and_train_model_rejects_malformed_rows(tmp_path):
    pickle_path = tmp_path / "embeddings.pkl"
    with pickle_path.open("wb") as file:
        pickle.dump([(1, np.array([0.1, 0.2]))], file)

    with pytest.raises(ValueError, match="id, embedding, and metadata"):
        generate.load_embeddings_and_train_model(pickle_path)


def test_load_embeddings_and_train_model_rejects_dimension_mismatch(tmp_path):
    pickle_path = tmp_path / "embeddings.pkl"
    with pickle_path.open("wb") as file:
        pickle.dump([
            (1, np.array([0.1, 0.2]), {"text": "sample text 1"}),
            (2, np.array([0.3]), {"text": "sample text 2"}),
        ], file)

    with pytest.raises(ValueError, match="same dimensionality"):
        generate.load_embeddings_and_train_model(pickle_path)


def test_load_embeddings_and_train_model_rejects_metadata_without_text(tmp_path):
    pickle_path = tmp_path / "embeddings.pkl"
    with pickle_path.open("wb") as file:
        pickle.dump([(1, np.array([0.1, 0.2]), {"title": "missing text"})], file)

    with pytest.raises(ValueError, match="metadata must include text"):
        generate.load_embeddings_and_train_model(pickle_path)


@pytest.mark.parametrize("bad_embedding", [
    np.array([0.1, "bad"], dtype=object),
    np.array([0.1, "0.2"], dtype=object),
    np.array([0.1, np.nan]),
    np.array([0.1, np.inf]),
])
def test_load_embeddings_and_train_model_rejects_non_finite_embedding_values(tmp_path, bad_embedding):
    pickle_path = tmp_path / "embeddings.pkl"
    with pickle_path.open("wb") as file:
        pickle.dump([(1, bad_embedding, {"text": "sample text"})], file)

    with pytest.raises(ValueError, match="numeric finite numbers"):
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


@pytest.mark.parametrize("cached_payload", [
    {},
    [],
    ["not an object"],
    [{}],
    [{"embedding": []}],
    [{"embedding": [True, 0.2]}],
    [{"embedding": ["0.1", 0.2]}],
    [{"embedding": [math.nan, 0.2]}],
    [{"embedding": [math.inf, 0.2]}],
    [{"embedding": [10 ** 400, 0.2]}],
    [{"embedding": [0.1, 0.2]}, {"embedding": [0.3]}],
])
def test_get_embeddings_rejects_invalid_cache_without_api_call(
    tmp_path, monkeypatch, cached_payload
):
    monkeypatch.chdir(tmp_path)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    cache_file = generate.get_cache_file(str(cache_dir), "sample query")
    with open(cache_file, "w", encoding="utf-8") as file:
        json.dump(cached_payload, file)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("Invalid cache data must not trigger an OpenAI API call")

    monkeypatch.setattr(generate.openai.Embedding, "create", fail_if_called)

    with pytest.raises(ValueError, match="Embedding (response|cache)"):
        generate.get_embeddings("sample query")


def test_get_embeddings_rejects_malformed_json_without_api_call(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    cache_file = generate.get_cache_file(str(cache_dir), "sample query")
    with open(cache_file, "w", encoding="utf-8") as file:
        file.write("{not json")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("Malformed cache data must not trigger an OpenAI API call")

    monkeypatch.setattr(generate.openai.Embedding, "create", fail_if_called)

    with pytest.raises(ValueError, match="valid UTF-8 JSON"):
        generate.get_embeddings("sample query")


def test_get_embeddings_rejects_invalid_api_data_before_cache_write(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        generate.openai.Embedding,
        "create",
        lambda **kwargs: {"data": [{"embedding": [math.nan]}]},
    )

    with pytest.raises(ValueError, match="numeric finite numbers"):
        generate.get_embeddings("sample query")

    assert not (tmp_path / "cache").exists()


def test_get_top_k_metadata():
    class FakeNearestNeighbors:
        def kneighbors(self, values):
            assert values == [[0.1, 0.2]]
            return np.array([[0.0, 0.1]]), np.array([[0, 1]])

    metadata = [{"text": "sample text 1"}, {"text": "sample text 2"}]

    assert generate.get_top_k_metadata(
        [0.1, 0.2], FakeNearestNeighbors(), metadata
    ) == metadata


@pytest.mark.parametrize("bad_embedding", [
    None,
    [],
    [True, 0.2],
    ["0.1", 0.2],
    [np.nan, 0.2],
    [np.inf, 0.2],
    [np.complex128(1 + 2j), 0.2],
    [10 ** 400, 0.2],
])
def test_get_top_k_metadata_rejects_invalid_query_embeddings(bad_embedding):
    class FakeNearestNeighbors:
        n_features_in_ = 2

        def kneighbors(self, values):
            raise AssertionError("invalid query embedding reached nearest-neighbor lookup")

    with pytest.raises(ValueError):
        generate.get_top_k_metadata(
            bad_embedding,
            FakeNearestNeighbors(),
            [{"text": "sample text"}],
        )


def test_get_top_k_metadata_rejects_dimension_mismatch():
    class FakeNearestNeighbors:
        n_features_in_ = 3

        def kneighbors(self, values):
            raise AssertionError("dimension mismatch reached nearest-neighbor lookup")

    with pytest.raises(ValueError, match="trained model dimensionality"):
        generate.get_top_k_metadata(
            [0.1, 0.2],
            FakeNearestNeighbors(),
            [{"text": "sample text"}],
        )


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


@pytest.mark.parametrize("distance_function", [
    generate.cosine_similarity,
    generate.euclidean_distance,
    generate.manhattan_distance,
])
@pytest.mark.parametrize("bad_vector", [
    None,
    [],
    [True, 0.2],
    ["0.1", 0.2],
    [np.nan, 0.2],
    [np.inf, 0.2],
    [np.complex128(1 + 2j), 0.2],
    [10 ** 400, 0.2],
])
def test_vector_math_rejects_invalid_values(distance_function, bad_vector):
    with pytest.raises(ValueError, match="non-empty numeric finite sequences"):
        distance_function(bad_vector, [1.0, 2.0])


def test_vector_math_rejects_invalid_values_in_second_vector():
    with pytest.raises(ValueError, match="non-empty numeric finite sequences"):
        generate.euclidean_distance([1.0, 2.0], [1.0, np.nan])


@pytest.mark.parametrize("distance_function, expected", [
    (generate.cosine_similarity, 8 / math.sqrt(65)),
    (generate.euclidean_distance, math.sqrt(2)),
    (generate.manhattan_distance, 2.0),
])
def test_vector_math_accepts_numpy_numeric_scalars(distance_function, expected):
    result = distance_function(
        [np.float32(1.0), np.int64(2)],
        [np.float64(2.0), np.int32(3)],
    )

    assert result == pytest.approx(expected)


def test_record_estimated_cost_adds_first_and_subsequent_values():
    fake_streamlit.session_state.clear()

    generate._record_estimated_cost(10, 0.001)
    generate._record_estimated_cost(5, 0.001)

    assert fake_streamlit.session_state["cost"] == "$0.0000150000"


def test_recursive_text_splitter_preserves_token_overlap():
    text = " ".join(f"word{index}" for index in range(650))

    chunks = token.text_splitter.split_text(text)

    assert [token.tiktoken_len(chunk) for chunk in chunks] == [500, 500, 367]
    assert chunks[0].endswith("word249")
    assert chunks[1].startswith("word240")
    assert chunks[1].endswith("word489")
    assert chunks[2].startswith("word480")
    assert chunks[2].endswith("word649")
