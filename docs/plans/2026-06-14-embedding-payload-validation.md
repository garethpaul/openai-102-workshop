# Embedding Payload Validation

status: completed

## Context

`get_embeddings` trusts writable JSON cache contents and the legacy OpenAI
response shape. Malformed, non-finite, or dimensionally inconsistent vectors
therefore reach downstream retrieval code, while corrupt cache JSON can produce
an implementation-specific decoder traceback.

## Requirements

- Validate cached and API embedding payloads before returning them.
- Require a non-empty list of object entries with non-empty, equally sized,
  real numeric finite vectors.
- Reject malformed cache JSON locally without falling back to a paid API call.
- Reject invalid API data before creating or updating a cache file.
- Preserve the legacy OpenAI SDK, model, cost accounting, and valid cache shape.
- Add no-network regression tests and a mutation-sensitive static contract.

## Scope Boundaries

- Do not migrate the OpenAI SDK, alter prompts or models, refresh generated
  caches, or change Streamlit credential handling.

## Verification

- An isolated Python 3.12 environment installed from `requirements-test.txt`
  passed `python -m pip check` with the host `PYTHONPATH` removed.
- `PYTHONDONTWRITEBYTECODE=1 python -m pytest -q test_app.py -k
  get_embeddings` passed 14 focused tests, including invalid cache and invalid
  API data paths that cannot make a network call or write a cache.
- `make test` passed all 74 no-network tests, and `make build` compiled every
  maintained Python module without retaining bytecode.
- `make check` passed the static contract and complete no-network suite in the
  isolated environment both from the repository and through an absolute
  Makefile path from `/tmp`.
- `pip-audit -r requirements-test.txt` and `pip-audit -r requirements.txt`
  reported no known vulnerabilities.
- Six hostile mutations were rejected for bypassing cached-response
  validation, writing API data before validation, permitting boolean values,
  permitting empty response data, removing invalid-cache coverage, and
  removing invalid-API-write coverage.
- `git diff --check`, generated-artifact inspection, secret-pattern scanning,
  and the exact intended-path diff audit passed before commit.
