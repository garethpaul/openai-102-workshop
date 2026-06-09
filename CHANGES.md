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
- Added NumPy and pytest compatibility metadata and fixed first-request cost
  accounting.
- Added no-network vector math validation for cosine-similarity dimension and
  zero-vector inputs.
- Capped nearest-neighbor training at the available row count so small
  embedding fixtures can be queried.
- Added no-network validation for empty embedding fixtures before nearest-neighbor
  training.
- Added no-network validation for malformed embedding fixture rows and
  inconsistent embedding dimensions.
- Added no-network metadata text validation for retrieval embedding fixtures.
- Added no-network finite embedding value validation for retrieval fixtures.
