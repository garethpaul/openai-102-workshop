# OpenAI 102 Workshop Baseline Plan

status: completed

## Context

`openai-102-workshop` is a Streamlit workshop app with legacy OpenAI SDK
examples, local/generated embedding caches, Docker support, and lesson pages for
embeddings, search, recommendations, clustering, fine-tuning, and LangChain.

## Risks

- The existing test imported a non-existent `Hello` module and did not run on
  this host.
- The sidebar token input defaulted to `sk-` and did not hide entered tokens.
- Text embedding cache filenames were derived from raw user input, allowing
  path-like queries to escape the cache folder.
- The crawler made requests without a timeout or HTTP status validation.
- Docker installed apt packages without cleanup and hard-coded the remote
  embeddings fixture URL.
- Generated caches and legacy OpenAI SDK examples were not called out clearly in
  root-level verification docs.
- `embedding_cache.pkl` was a generated local pickle cache tracked in source
  control.

## Work Completed

- Added `scripts/check-workshop-baseline.py` and wired `make check`.
- Replaced `test_app.py` with no-network pytest coverage for cache paths,
  cached embeddings, nearest-neighbor metadata, augmented queries, and distance
  validation.
- Made sidebar token entry password-only with no `sk-` default and local
  `OPENAI_API_KEY` assignment.
- Added hashed cache filenames for new text embedding writes while still reading
  existing simple cache fixtures.
- Added request timeout/status handling to the crawler.
- Added generated cache ignores, Docker hardening, changelog, README, security,
  vision, and overview updates.
- Removed `embedding_cache.pkl` from source control while keeping it ignored for
  local regeneration.

## Verification

- `make check`
- `python3 -m pytest -q test_app.py`
- `python3 scripts/check-workshop-baseline.py`
- `git diff --check`

The checks do not call OpenAI or install the full workshop dependency set.
