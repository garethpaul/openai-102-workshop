import json
import errno
import os

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


def test_embedding_cache_rejects_symlink_and_hard_link_reads(tmp_path):
    external_path = tmp_path / "external.json"
    external_path.write_text('{"prompt": "external"}', encoding="utf-8")

    symlink_path = tmp_path / "symlink-cache.json"
    symlink_path.symlink_to(external_path)
    with pytest.raises(ValueError, match="owned regular file"):
        load_embedding_cache(symlink_path)

    hard_link_path = tmp_path / "hard-link-cache.json"
    hard_link_path.hardlink_to(external_path)
    with pytest.raises(ValueError, match="owned regular file"):
        load_embedding_cache(hard_link_path)


def test_embedding_cache_save_rejects_symlinked_temporary_path(tmp_path):
    cache_path = tmp_path / "embedding_cache.json"
    temporary_path = cache_path.with_suffix(".json.tmp")
    external_path = tmp_path / "external.json"
    external_path.write_text("do not overwrite", encoding="utf-8")
    temporary_path.symlink_to(external_path)

    with pytest.raises(ValueError, match="temporary path"):
        save_embedding_cache({"prompt": "response"}, cache_path)

    assert external_path.read_text(encoding="utf-8") == "do not overwrite"


def test_embedding_cache_rejects_symlink_swap_during_open(tmp_path, monkeypatch):
    cache_path = tmp_path / "embedding_cache.json"
    cache_path.write_text('{"prompt": "cached"}', encoding="utf-8")

    def reject_no_follow_open(*args, **kwargs):
        raise OSError(errno.ELOOP, os.strerror(errno.ELOOP), cache_path)

    monkeypatch.setattr(os, "open", reject_no_follow_open)

    with pytest.raises(ValueError, match="owned regular file"):
        load_embedding_cache(cache_path)
