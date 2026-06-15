# Recommendation Container Validation

Status: completed

## Problem

The recommendation helper now validates customer entries, industry names, and
product names, but malformed top-level containers can still raise raw type
errors. `customer_data=None` is not iterable, while non-mapping embedding or
product payloads do not support the mapping operations used by the helper.

## Scope

- Require customer data to be a non-string iterable and embedding/product
  collections to be mappings before iteration or lookup.
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

## Work Completed

- Validated the embedding mapping before score construction and returned an
  empty score mapping when it is malformed.
- Validated the customer iterable and product mapping before lookup while
  preserving computed similarity scores in the no-recommendation result.
- Added parameterized runtime coverage, mutation-sensitive static contracts,
  and synchronized guidance without OpenAI API or dependency changes.

## Verification Completed

- Twenty-one focused recommendation cases and the complete no-network suite
  passed in the isolated Python 3.12 environment.
- All four Make gates passed from the repository and the canonical check passed
  from an external directory.
- Seven isolated hostile mutations were rejected for missing embedding,
  customer iterable, customer string/mapping, and product mapping guards,
  missing focused coverage or guidance, and stale plan status.
- Exact eight-file diff, generated artifact, credential, dependency, data,
  conflict-marker, binary, mode, whitespace, and intended-path audits passed.
- Live or paid OpenAI behavior was not exercised.
