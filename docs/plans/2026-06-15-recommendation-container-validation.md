# Recommendation Container Validation

Status: planned

## Problem

The recommendation helper now validates customer entries, industry names, and
product names, but malformed top-level containers can still raise raw type
errors. `customer_data=None` is not iterable, while non-mapping embedding or
product payloads do not support the mapping operations used by the helper.

## Scope

- Require customer data to be a list and embedding/product collections to be
  mappings before iteration or lookup.
- Return the existing no-recommendation shape for malformed containers.
- Preserve valid recommendation ranking, similarity scores, product filtering,
  random-choice injection, and all OpenAI SDK examples unchanged.
- Add focused runtime coverage, mutation-sensitive static contracts, and
  synchronized guidance.
- Do not change dependencies, model names, API calls, prompts, fixture data, or
  generated caches.

## Verification

- Run focused recommendation tests and the complete no-network suite in the
  existing isolated Python environment.
- Run all four Make gates and the external-directory absolute-Makefile check.
- Reject isolated mutations for each missing container guard, missing focused
  coverage, missing guidance, and stale plan status.
- Audit the exact diff, generated artifacts, credentials, dependencies, data,
  conflicts, modes, binaries, and intended paths.

## Risks

- The workshop's live paid OpenAI paths remain outside local validation.
- This change does not alter dependency locks or resolve default-branch alerts.
- The change must remain stacked on PR #18; neither pull request may be merged
  or closed without explicit owner authorization.

## Success Criteria

- Malformed customer, embedding, and product containers return `(None, scores)`
  without raising.
- A malformed embeddings container returns an empty score mapping.
- Valid recommendation behavior remains unchanged.
