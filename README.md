# openai-102-workshop

<!-- README-OVERVIEW-IMAGE -->
![Project overview](docs/readme-overview.svg)

## Overview

`garethpaul/openai-102-workshop` is a Streamlit learning app for OpenAI API
concepts, embeddings, retrieval-style search, recommendations, clustering, and
fine-tuning exercises.

This README is based on the checked-in source, manifests, scripts, and repository metadata on the `main` branch. The project language mix found during review was: no dominant source language detected.

## Repository Contents

- `README.md` - project overview and local usage notes
- `requirements.txt` - Python dependency or packaging metadata
- `cache` - source or example code
- `CHANGES.md` - baseline change log
- `components` - source or example code
- `Dockerfile` - container build instructions
- `Makefile` - local build or utility targets
- `pages` - source or example code
- `Pipfile` - Python dependency or packaging metadata
- `query_cache` - source or example code
- `scripts/check-workshop-baseline.py` - static baseline checks used by `make check`
- `SECURITY.md` - security reporting and disclosure guidance
- `test_app.py` - no-network tests for cache and retrieval helpers
- `url_cache` - source or example code
- `utils` - source or example code
- `docs/plans/2026-06-08-openai-102-workshop-baseline.md` - completed hardening plan

Additional scan context:

- Source directories: cache, components, pages, query_cache, url_cache, utils
- Dependency and build manifests: Dockerfile, Makefile, Pipfile, requirements.txt
- Entry points or build surfaces: Dockerfile, Makefile
- Test-looking files: `test_app.py`

## Getting Started

### Prerequisites

- Git
- Python 3.10 for the workshop runtime, matching the checked-in Pipfile
- Python 3 with `pytest`, `numpy`, and `scikit-learn` for local no-network checks
- An OpenAI API key supplied through local UI input or `OPENAI_API_KEY` when running API lessons

### Setup

```bash
git clone https://github.com/garethpaul/openai-102-workshop.git
cd openai-102-workshop
python -m pip install -r requirements.txt
make lint
make test
make build
make check
```

The setup commands above are derived from repository files. Legacy mobile, Python, or JavaScript samples may require older SDKs or package versions than a modern workstation uses by default.

## Running or Using the Project

- Run `make lint`, `make test`, `make build`, and `make check` before changing
  workshop logic or generated-cache behavior.
- `make test` runs pytest with Python bytecode writes disabled so verification
  does not leave `__pycache__` files behind.
- Run `make run` or `streamlit run 👋_Hello.py` to start the app.
- Enter an OpenAI API key only through the local sidebar or `OPENAI_API_KEY`.
- Treat the checked-in snippets as legacy OpenAI SDK examples pinned to
  `openai<1.0`. Model or SDK migrations should be deliberate compatibility
  updates.
- Retrieval vector math helpers validate dimensionality and zero-vector cosine
  inputs before returning workshop results.
- Shared vector value validation rejects empty, boolean, string, complex,
  non-finite, and overflowing inputs across cosine, Euclidean, and Manhattan
  helpers.
- Small embedding fixtures cap nearest-neighbor lookup to the available row
  count so no-network tests can query them.
- Empty embedding fixtures fail with a clear validation error before model
  training.
- Malformed embedding fixtures fail before model training when rows are missing
  metadata or embedding dimensions do not match.
- Embedding fixture metadata must include non-empty `text` before retrieval
  examples build augmented queries.
- Finite embedding values are required before nearest-neighbor training so
  invalid fixture vectors fail with clear errors.
- Numeric embedding values must be real numeric types, not stringified numbers,
  before model training.
- Query embedding validation rejects empty, boolean, non-numeric, non-finite,
  and dimension-mismatched vectors before nearest-neighbor lookup.

## Testing and Verification

- `make lint`
- `make test`
- `make build`
- `make check`
- `python3 -m pytest -q test_app.py`
- `python3 scripts/check-workshop-baseline.py`
- Pinned hosted Linux validation installs `requirements-test.txt`, runs
  `python -m pip check`, and executes the same no-network `make check` gate on
  Python 3.10.

When the required SDK or runtime is unavailable, use static checks and source review first, then verify on a machine that has the matching platform toolchain.

## Configuration and Secrets

- Detected references to OpenAI. Keep API keys, OAuth credentials, tokens, and account-specific values in local configuration only.
- Generated caches live under `cache/`, `url_cache/`, `query_cache/`, and pickle files such as `embedding_cache.pkl`. Do not add private cache refreshes or customer data.
- `embedding_cache.pkl` is intentionally ignored and should remain untracked;
  regenerate it locally when running the clustering lesson.
- New text embedding cache writes use hashed filenames so user input cannot escape the cache directory.
- Python bytecode generated by local tooling should stay out of the repository
  and out of completed verification workspaces.

## Security and Privacy Notes

- Review changes touching external API calls or credential-adjacent configuration; examples from the scan include Pipfile.
- Review changes touching network requests, sockets, or service endpoints; examples from the scan include Dockerfile, Pipfile.
- Review changes touching file, media, JSON, XML, CSV, OCR, or data parsing; examples from the scan include Dockerfile.
- Review changes touching pickle files carefully; only load trusted workshop fixtures.
- Review vector math helper changes with no-network tests so retrieval lessons
  fail clearly on invalid fixture data.
- Review small embedding fixtures with no-network nearest-neighbor tests before
  refreshing workshop data.
- Empty embedding fixtures should fail closed instead of producing ambiguous
  unpacking errors.
- Malformed embedding fixtures should fail closed before nearest-neighbor
  training.
- Metadata text validation should fail closed before retrieval examples assume
  `metadata["text"]` exists.
- Finite embedding values should be checked before retrieval examples train
  nearest-neighbor models.
- Numeric embedding values should be enforced before retrieval examples accept
  fixture vectors.
- Python bytecode should not remain after local `make test` or `make check`
  runs.

## Maintenance Notes

- See `SECURITY.md` for vulnerability reporting and safe research guidance.
- See `CHANGES.md` and `docs/plans/2026-06-08-openai-102-workshop-baseline.md` for the current verification baseline.
- See `docs/plans/2026-06-09-make-gate-aliases.md` for the local verification
  gate aliases.
- See `docs/plans/2026-06-10-numeric-embedding-values.md` for the embedding
  fixture numeric-type validation contract.
- See `docs/plans/2026-06-12-vector-value-validation.md` for the shared vector
  math input boundary.
- See `docs/plans/2026-06-10-query-embedding-validation.md` for the retrieval
  query vector validation contract.
- See `docs/plans/2026-06-10-hosted-workshop-validation.md` for the hosted
  Linux test dependency and `make check` contract.
- See `VISION.md` for project direction and contribution guardrails.

## Contributing

Keep changes small and tied to the project that is already present in this repository. For code changes, document the toolchain used, avoid committing generated dependency directories or local configuration, and update this README when setup or verification steps change.
