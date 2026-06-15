# Malformed Customer Entry Guard

status: planned

## Context

`components/recommendations.py` assumes every member of `customer_data` is a
mapping. A malformed member before the requested customer raises an attribute
error and prevents the valid customer from receiving a recommendation.

## Goal

Ignore non-mapping customer entries while preserving the existing first-match
lookup and all recommendation scoring and product-validation behavior.

## Scope

- Guard the customer lookup against non-mapping list members.
- Preserve the first valid entry whose `customer_id` matches the request.
- Return the existing no-recommendation result when no valid match exists.
- Add focused regression coverage, a mutation-sensitive static contract, and
  synchronized project guidance.

## Implementation Units

### U1: Guard customer lookup

Files: `components/recommendations.py`, `test_app.py`

Filter lookup candidates by mapping shape before reading `customer_id`, with
coverage for a malformed entry preceding a valid match and an all-malformed
customer list.

### U2: Lock the boundary into project verification

Files: `scripts/check-workshop-baseline.py`, `README.md`, `SECURITY.md`,
`VISION.md`, `CHANGES.md`, `docs/plans/2026-06-15-malformed-customer-entry-guard.md`

Require the guarded lookup, focused tests, synchronized guidance, and truthful
completed verification evidence without changing dependencies or generated
workshop data.

## Verification Plan

- Run focused recommendation tests and the complete no-network test suite.
- Run all Make gates and the absolute-Makefile check externally.
- Reject isolated mutations of the mapping guard, regression tests, guidance,
  and completed plan evidence.
- Audit the exact diff for generated artifacts, dependency changes,
  credentials, binaries, modes, and unintended paths.

## Scope Boundaries

- Do not change recommendation ranking, product selection, embeddings, OpenAI
  API usage, dependency locks, generated data, or Streamlit presentation.
- Do not normalize or mutate valid customer mappings.
