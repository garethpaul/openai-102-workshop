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
- Python 3.10+ for the workshop runtime
- Python 3 with `pytest`, `numpy`, and `scikit-learn` for local no-network checks
- An OpenAI API key supplied through local UI input or `OPENAI_API_KEY` when running API lessons

### Setup

```bash
git clone https://github.com/garethpaul/openai-102-workshop.git
cd openai-102-workshop
python -m pip install -r requirements.txt
make check
```

The setup commands above are derived from repository files. Legacy mobile, Python, or JavaScript samples may require older SDKs or package versions than a modern workstation uses by default.

## Running or Using the Project

- Run `make check` before changing workshop logic or generated-cache behavior.
- Run `make run` or `streamlit run 👋_Hello.py` to start the app.
- Enter an OpenAI API key only through the local sidebar or `OPENAI_API_KEY`.
- Treat the checked-in snippets as legacy OpenAI SDK examples. Model or SDK
  migrations should be deliberate compatibility updates.

## Testing and Verification

- `make check`
- `python3 -m pytest -q test_app.py`
- `python3 scripts/check-workshop-baseline.py`

When the required SDK or runtime is unavailable, use static checks and source review first, then verify on a machine that has the matching platform toolchain.

## Configuration and Secrets

- Detected references to OpenAI. Keep API keys, OAuth credentials, tokens, and account-specific values in local configuration only.
- Generated caches live under `cache/`, `url_cache/`, `query_cache/`, and pickle files such as `embedding_cache.pkl`. Do not add private cache refreshes or customer data.
- `embedding_cache.pkl` is intentionally ignored and should remain untracked;
  regenerate it locally when running the clustering lesson.
- New text embedding cache writes use hashed filenames so user input cannot escape the cache directory.

## Security and Privacy Notes

- Review changes touching external API calls or credential-adjacent configuration; examples from the scan include Pipfile.
- Review changes touching network requests, sockets, or service endpoints; examples from the scan include Dockerfile, Pipfile.
- Review changes touching file, media, JSON, XML, CSV, OCR, or data parsing; examples from the scan include Dockerfile.
- Review changes touching pickle files carefully; only load trusted workshop fixtures.

## Maintenance Notes

- See `SECURITY.md` for vulnerability reporting and safe research guidance.
- See `CHANGES.md` and `docs/plans/2026-06-08-openai-102-workshop-baseline.md` for the current verification baseline.
- See `VISION.md` for project direction and contribution guardrails.

## Contributing

Keep changes small and tied to the project that is already present in this repository. For code changes, document the toolchain used, avoid committing generated dependency directories or local configuration, and update this README when setup or verification steps change.
