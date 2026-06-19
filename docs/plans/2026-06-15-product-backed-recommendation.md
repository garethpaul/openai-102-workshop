# Product-Backed Recommendation Selection

status: completed

## Summary

Choose the most similar industry that can actually supply a product instead of
returning no recommendation whenever the customer's own industry lacks a
product mapping.

## Problem

The customer-relative helper ranks every embedding and then checks products
only for the single highest-scoring industry. Because an industry's embedding
is maximally similar to itself, customers in the 18 checked-in industries
without product mappings receive no recommendation even though E-commerce and
Telecommunications products are available and have valid similarity scores.

## Requirements

- Rank only industries whose product mapping is a nonempty list.
- Preserve same-industry recommendations when that industry has products.
- Fall back to the most similar product-backed industry otherwise.
- Return no recommendation when no valid product-backed candidate exists.
- Preserve the complete pairwise similarity matrix used by visualizations.
- Add deterministic fixture and mutation coverage without paid API calls.

## Implementation

- Filter the selected customer's similarity scores to nonempty list-valued
  product mappings before choosing the highest score.
- Keep random product choice injectable for deterministic tests.
- Add a checked-in Technology-customer regression alongside existing
  unavailable-input coverage.

## Verification

- Run focused recommendation tests, the full no-network suite, every Make gate,
  and external-directory `make check`.
- Reject mutations that restore rank-before-product selection, accept malformed
  product mappings, remove the fixture regression, documentation, or completed
  plan evidence.
- Audit exact diff, generated artifacts, conflict markers, binary/large files,
  and changed-line credential patterns.

## Risks

- Product mappings remain intentionally sparse workshop data; this change
  chooses only among configured products and does not invent catalog entries.
- No paid OpenAI request or live Streamlit interaction is required.
- The change must remain stacked on PR #14 and must not be merged or closed
  without explicit owner authorization.

## Work Completed

- Filtered customer-relative similarity scores to industries with nonempty,
  list-valued product mappings before ranking.
- Preserved same-industry selection when supported and the complete pairwise
  matrix used by the page visualizations.
- Added a checked-in Technology-customer fallback regression, malformed mapping
  coverage, static contracts, and maintenance guidance.

## Verification Completed

- Six focused recommendation cases and the complete 80-test no-network suite
  passed in isolated Python 3.12 environments from the exact test lock.
- `make lint`, `make test`, `make build`, and `make check` passed, and
  `make check` passed from an external working directory.
- Runtime imports and the headless Streamlit health smoke passed against both
  exact locks.
- Exact production and test lock audits using
  `pip-audit --no-deps --disable-pip` reported no known vulnerabilities. The
  standard production `make audit` path could not build a temporary `pyarrow`
  dependency because a Rust compiler is unavailable on this host; it reported
  no vulnerability finding before that environment failure.
- Six isolated hostile mutations restoring rank-before-product selection,
  accepting malformed mappings, removing the fixture regression or empty
  candidate guard, deleting guidance, or reverting plan status were rejected.
- `git diff --check`, exact-diff, generated-artifact, conflict-marker,
  intended-path, protected lock/build-path, and changed-line credential audits
  passed.
