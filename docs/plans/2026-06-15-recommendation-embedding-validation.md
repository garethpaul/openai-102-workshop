---
title: Recommendation Embedding Validation
type: reliability
status: completed
date: 2026-06-15
execution: code
---

# Recommendation Embedding Validation

## Problem

Recommendation containers can be valid mappings while nested embedding values
remain unusable. Zero vectors, mismatched dimensions, and nonnumeric entries
raise from cosine similarity and abort the page instead of treating the affected
industry pair as unavailable recommendation evidence.

## Approach

- Isolate expected similarity validation failures per industry pair.
- Skip only invalid pairs while preserving valid computed scores.
- Return no product when the selected customer has no trustworthy product-backed
  score.
- Add focused regressions plus mutation-sensitive source, guidance, and
  completed-plan contracts.

## Files

- `components/recommendations.py`
- `test_app.py`
- `scripts/check-workshop-baseline.py`
- `README.md`
- `SECURITY.md`
- `VISION.md`
- `CHANGES.md`
- `docs/plans/2026-06-15-recommendation-embedding-validation.md`

## Verification

- Run focused recommendation tests, the complete offline suite, all Make gates,
  and external-directory verification.
- Reject isolated exception-boundary, invalid-pair, test, guidance, and plan
  mutations.
- Audit the exact diff, generated artifacts, credentials, dependencies,
  conflicts, binaries, large files, modes, and whitespace.

## Risks

- Invalid embedding pairs intentionally contribute no similarity score.
- No OpenAI API call, private customer data, or dependency update will be used.
- Keep this change stacked on PR #19; do not merge or close stacked pull
  requests without explicit authorization.

## Status: Completed

## Work Completed

- Isolate expected cosine-similarity validation failures per industry pair.
- Preserve valid scores while excluding invalid pairs from product selection.
- Add zero-vector, dimension-mismatch, nonnumeric, source, guidance, and plan
  contracts.

## Verification Completed

- Three focused recommendation cases passed, and the complete no-network suite
  passed all 98 tests in the pinned Python 3.12 test environment.
- All four Make gates passed, and `make check` passed from an external directory.
- Six isolated hostile mutations were rejected for uncaught similarity errors,
  narrowed exception handling, weakened product and score assertions, missing
  guidance, and reopened plan status.
- Exact diff, bytecode/cache, credential, dependency, conflict, binary,
  large-file, mode, whitespace, and intended-path audits passed.
- No OpenAI API call, private customer data, or dependency update was used.
