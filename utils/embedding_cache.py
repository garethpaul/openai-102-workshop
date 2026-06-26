"""Safe local cache helpers for the clustering workshop."""

import errno
import json
import os
import stat
from pathlib import Path


EMBEDDING_CACHE_FILE = Path("embedding_cache.json")


def _validated_embedding_cache(value):
    if not isinstance(value, dict) or not all(
        isinstance(key, str) and isinstance(item, str)
        for key, item in value.items()
    ):
        raise ValueError("embedding cache must contain string keys and values")
    return value


def _open_owned_regular_file(path):
    try:
        path_stat = path.lstat()
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(path_stat.st_mode) or path_stat.st_nlink != 1:
        raise ValueError("embedding cache must be an owned regular file")

    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as error:
        if error.errno in {errno.ELOOP, errno.ENOENT}:
            raise ValueError("embedding cache must be an owned regular file") from None
        raise
    descriptor_stat = os.fstat(descriptor)
    if (
        not stat.S_ISREG(descriptor_stat.st_mode)
        or descriptor_stat.st_nlink != 1
        or (descriptor_stat.st_dev, descriptor_stat.st_ino)
        != (path_stat.st_dev, path_stat.st_ino)
    ):
        os.close(descriptor)
        raise ValueError("embedding cache must be an owned regular file")
    return descriptor


def load_embedding_cache(path=EMBEDDING_CACHE_FILE):
    path = Path(path)
    descriptor = _open_owned_regular_file(path)
    if descriptor is None:
        return {}
    try:
        with os.fdopen(descriptor, "r", encoding="utf-8") as cache_file:
            serialized = cache_file.read()
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
    serialized = json.dumps(validated_cache, ensure_ascii=False, sort_keys=True)
    try:
        descriptor = os.open(
            temporary_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
        )
    except FileExistsError:
        raise ValueError("embedding cache temporary path must not already exist") from None
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as cache_file:
            cache_file.write(serialized)
            cache_file.flush()
            os.fsync(cache_file.fileno())
        temporary_path.replace(path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
