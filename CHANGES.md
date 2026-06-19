# Changes

## 2026-06-17

- Added public-PyPI artifact hashes to both universal Python locks and required
  hash verification for application, test, CI, and container installation.

## 2026-06-16

- Made exact-lock vulnerability audits independent of package builds while
  retaining the separate full application install and compatibility smoke.
- Added safe JSON embedding fixtures across the shared nearest-neighbor loader,
  step-4 demo, container build, and tracked test data, removing hidden remote
  pickle download and executable deserialization paths.
- Corrected the displayed fine-tuning retry example to retry only legacy SDK
  rate limits and re-raise the final attempt instead of masking other failures.
- Added an explicit Starlette resolver floor so fresh lock generation preserves
  the reviewed 1.3.1 application security pin from public PyPI regardless of
  caller index configuration.
- Added a customer-industry recommendation tie break so equal top scores prefer
  the matched customer's own product-backed industry over mapping order.
- Updated both generated Python locks to `aiohttp==3.14.1` after the 3.14.0
  release acquired eight published security advisories.
- Updated the application lock to `starlette==1.3.1` after the 1.2.1 release
  acquired two published request-processing advisories.

## 2026-06-15

- Fixed Recommendations to select the nearest industry from the selected
  customer's embedding instead of the first JSON key, with a safe no-product
  fallback.
- Made Recommendations fall back to the nearest product-backed industry when
  the customer's own industry has no configured catalog entry.
- Filtered malformed and blank catalog members before product recommendation.
- Ignored malformed customer-list members during recommendation lookup.
- Required a nonempty string customer industry before recommendation embedding
  lookup.
- Added recommendation container validation for malformed top-level customer,
  embedding, and product collections.
- Skipped invalid recommendation embedding pairs while preserving valid scores.
- Kept recommendation charts usable when invalid embedding pairs leave sparse
  similarity scores.

## 2026-06-14

- Added strict validation for cached and API embedding payloads so malformed,
  non-finite, or dimensionally inconsistent vectors fail before retrieval or
  cache writes, without turning an invalid cache into a paid API fallback.

## 2026-06-13

- Made build, test, static, runtime, smoke, lock, audit, and run tooling resolve
  from the checkout for absolute Makefile invocations.
- Replaced the writable clustering pickle with a strict UTF-8 JSON embedding
  cache that accepts only string keys and values.
- Added isolated no-network cache tests and static contracts that prevent
  generated clustering data from returning to executable pickle loading.
- Added a dated OpenAI API compatibility inventory and learner-visible warnings
  for preserved SDK 0.28.1, legacy model, embedding, and fine-tuning examples.

## 2026-06-12

- Added one shared vector-pair validation boundary across cosine, Euclidean,
  and Manhattan helpers.
- Added no-network tests rejecting empty, boolean, string, complex, non-finite,
  and overflowing values while preserving Python and NumPy numeric support.
- Replaced the exported workstation dependency list with reviewed Python 3.12
  direct inputs and generated exact application and test locks.
- Removed unused vulnerable PyTorch, Transformers, SentencePiece, virtualenv,
  and python-dotenv entries; both resulting locks audit with zero findings.
- Moved the runtime text splitter to `langchain-text-splitters` and preserved
  exact token chunk and overlap behavior in no-network tests.
- Added lock regeneration, vulnerability audit, direct-import, and real
  headless Streamlit health gates for both push and pull-request validation.
- Upgraded the Pipfile and container baseline to Python 3.12 while preserving
  the legacy OpenAI lesson API at `openai==0.28.1`.

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
- Added `make lint`, `make test`, and a bytecode-free `make build` compile
  gate around the existing no-network pytest and static workshop checks.
- Made `make test` bytecode-free and added a baseline guard against leftover
  Python bytecode.

## 2026-06-10

- Required retrieval fixture embedding values to be real numeric types instead
  of stringified numbers before nearest-neighbor training.
- Added a GitHub Actions workflow that installs the minimal Python 3.10
  no-network test dependencies and runs `make check`.
- Added pinned hosted Linux validation with a minimal Python 3.10 test
  dependency set, `pip check`, and the no-network `make check` gate.
- Added query embedding validation before nearest-neighbor lookup for empty,
  non-numeric, non-finite, boolean, and dimension-mismatched vectors.
