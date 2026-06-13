# JSON Embedding Cache

status: planned

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

- Run all Make gates, focused tests, Python compilation, hostile mutations,
  diff checks, artifact scans, and secret scans.
