# Product-Backed Recommendation Selection

status: planned

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
