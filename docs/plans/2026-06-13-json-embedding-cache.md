# JSON Embedding Cache

status: completed

## Context

The clustering workshop page loads a writable local `embedding_cache.pkl` with
`pickle.load`. A replaced cache file can execute arbitrary Python code when the
page starts, even though the cache only stores string keys and string values.

## Requirements

- Replace the generated clustering cache with UTF-8 JSON and never unpickle it.
- Accept only a JSON object with string keys and string values; reject malformed
  or structurally invalid cache data with a stable local error.
- Preserve missing-cache behavior and atomic-enough complete-file writes.
- Keep OpenAI calls, prompts, models, credentials, clustering behavior, and
  trusted test embedding fixtures out of scope.
- Add offline tests, static contracts, docs, and completed verification.

## Scope Boundaries

- Do not make paid API calls, migrate the OpenAI SDK, change lesson prompts, or
  reinterpret trusted `test_embeddings.pkl` fixtures.

## Verification

- `PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q test_embedding_cache.py`
  passed all 6 focused tests.
- An isolated Python 3.12 environment created from `requirements-test.txt`
  passed `python -m pip check`, all 61 tests in `test_app.py` and
  `test_embedding_cache.py`, and `make build` with the host `PYTHONPATH`
  removed.
- `make lint`, `make test`, `make build`, and `make check` passed in that
  isolated environment.
- The static contract rejected six hostile mutations covering restored pickle
  loading, bypassed cache helpers, a non-JSON filename, weakened string-value
  validation, non-atomic direct writes, and removed regression coverage.
- `git diff --check`, generated-artifact checks, secret scans, and the exact
  intended-path diff audit passed before commit.
