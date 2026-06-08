# Changes

## 2026-06-08

- Added `make check` with static workshop checks and no-network pytest coverage.
- Fixed the broken test import by testing `utils.generate` directly with a
  fake Streamlit module.
- Removed the `sk-` sidebar default and made the OpenAI token input a password
  field that sets `OPENAI_API_KEY` only in the local process.
- Added safe hashed cache filenames for new text embedding cache writes while
  preserving existing simple cache reads.
- Removed generated `embedding_cache.pkl` from source control and ignored
  future local copies.
- Added crawler request timeouts and HTTP status checks.
- Tightened Docker dependency installation and made the embeddings fixture URL
  configurable.
- Documented generated cache handling and legacy OpenAI SDK assumptions.
- Pinned the legacy OpenAI SDK contract to `openai<1.0` in both dependency
  manifests.
