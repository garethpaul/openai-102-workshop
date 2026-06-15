# Customer Industry Name Validation

Status: planned

## Problem

The recommendation helper now skips customer-list members that are not
mappings, but a mapping can still contain an invalid `industry` value. An
unhashable value such as a list reaches `industry_embeddings.get(industry)` and
raises `TypeError` instead of returning the existing no-recommendation result.

## Scope

- Require a matched customer's industry to be a nonempty string before any
  embedding lookup.
- Treat invalid and whitespace-only industry values as unavailable input.
- Preserve valid customer selection, pairwise similarity scores, product-backed
  ranking, product-name filtering, and random-choice injection.
- Add focused runtime cases, mutation-sensitive static contracts, and
  synchronized guidance.

## Verification

- Run the focused recommendation tests and complete no-network suite in the
  existing isolated Python environment.
- Run all four Make gates and the canonical external-directory check with
  explicit timeouts.
- Reject isolated mutations for missing type/blank validation, missing focused
  cases, missing guidance, and stale plan status.
- Audit the exact diff, generated artifacts, credentials, dependencies, data,
  conflicts, modes, binaries, and intended paths.

## Risks

- The workshop's live paid OpenAI paths remain outside local validation.
- This change does not alter dependency locks or claim to resolve default-branch
  Dependabot alerts.
- The change must remain stacked on PR #17; neither pull request may be merged
  or closed without explicit owner authorization.

## Work Completed

Pending implementation.

## Verification Completed

Pending implementation and verification.
