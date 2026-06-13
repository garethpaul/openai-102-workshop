"""Safe local cache helpers for the clustering workshop."""

import json
from pathlib import Path


EMBEDDING_CACHE_FILE = Path("embedding_cache.json")


def _validated_embedding_cache(value):
    if not isinstance(value, dict) or not all(
        isinstance(key, str) and isinstance(item, str)
        for key, item in value.items()
    ):
        raise ValueError("embedding cache must contain string keys and values")
    return value


def load_embedding_cache(path=EMBEDDING_CACHE_FILE):
    path = Path(path)
    try:
        serialized = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except UnicodeDecodeError:
        raise ValueError("embedding cache must be valid UTF-8 JSON") from None
    try:
        return _validated_embedding_cache(json.loads(serialized))
    except json.JSONDecodeError:
        raise ValueError("embedding cache must be valid UTF-8 JSON") from None


def save_embedding_cache(cache, path=EMBEDDING_CACHE_FILE):
    path = Path(path)
    validated_cache = _validated_embedding_cache(cache)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(validated_cache, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    temporary_path.replace(path)
