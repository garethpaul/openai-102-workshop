import json

import pytest

from utils.embedding_cache import load_embedding_cache, save_embedding_cache


def test_embedding_cache_missing_file_is_empty(tmp_path):
    assert load_embedding_cache(tmp_path / "missing.json") == {}


def test_embedding_cache_round_trip_uses_json(tmp_path):
    cache_path = tmp_path / "embedding_cache.json"
    cache = {"prompt": "response", "second": "cached value"}

    save_embedding_cache(cache, cache_path)

    assert json.loads(cache_path.read_text(encoding="utf-8")) == cache
    assert load_embedding_cache(cache_path) == cache
    assert not cache_path.with_suffix(".json.tmp").exists()


@pytest.mark.parametrize("contents", ["not json", "[]", '{"prompt": 123}'])
def test_embedding_cache_rejects_malformed_or_invalid_data(tmp_path, contents):
    cache_path = tmp_path / "embedding_cache.json"
    cache_path.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match="embedding cache must"):
        load_embedding_cache(cache_path)


def test_embedding_cache_rejects_invalid_utf8(tmp_path):
    cache_path = tmp_path / "embedding_cache.json"
    cache_path.write_bytes(b"\xff")

    with pytest.raises(ValueError, match="valid UTF-8 JSON"):
        load_embedding_cache(cache_path)
