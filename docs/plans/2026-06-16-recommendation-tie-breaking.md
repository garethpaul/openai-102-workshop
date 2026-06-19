---
title: Recommendation Tie Breaking
status: completed
date: 2026-06-16
---

# Recommendation Tie Breaking

## Priority

P1 recommendation correctness. Equal top similarity scores can select a
different industry solely because it appeared earlier in the input mapping,
even when the customer's own industry has a valid product.

## Problem

`max(product_scores, key=product_scores.get)` uses mapping insertion order to
break equal scores. Duplicate or equivalent embeddings can therefore route a
customer away from their own equally ranked industry based on fixture order.

## Approach

- Rank by similarity score first and prefer the customer's own industry when
  scores are equal.
- Keep the existing product-backed fallback when the customer's industry has no
  valid product.
- Add a focused regression with the competing identical embedding inserted
  before the customer's industry.
- Preserve all malformed-container, name, embedding, cache, dependency, API,
  model, prompt, and workflow boundaries.

## Files

- `components/recommendations.py`
- `test_app.py`
- `scripts/check-workshop-baseline.py`
- `README.md`
- `SECURITY.md`
- `VISION.md`
- `CHANGES.md`
- `docs/plans/2026-06-16-recommendation-tie-breaking.md`

## Verification

- Prove equal top scores prefer the customer's own product regardless of input
  insertion order.
- Prove the product-backed fallback still works when the customer's own
  industry has no valid product.
- Run the focused recommendation tests, complete no-network suite, all Make
  gates, external-directory gate, and exact test-lock dependency check.
- Reject isolated score, own-industry preference, regression assertion,
  checker, guidance, changelog, and completed-plan mutations.
- Audit exact diff, generated artifacts, credentials, conflicts, binaries,
  large files, modes, dependency locks, fixtures, and whitespace.

## Scope Boundaries

- Do not call OpenAI APIs, change models/prompts, regenerate embeddings, alter
  dependency locks, or change random selection within the winning industry.
- Do not redesign recommendation scoring or introduce cross-customer state.
- Keep PR #20 and its predecessors open and retain base-first stack ordering.

## Success Criteria

- Equal scores no longer let mapping order displace a valid product from the
  customer's own industry.
- Existing higher-score fallback and invalid-input behavior remain unchanged.

## Work Completed

- Ranked candidate industries by similarity score first and the matched
  customer's own industry second.
- Added a regression where an equally embedded competitor is inserted before
  the customer's own product-backed industry.
- Extended mutation-sensitive static contracts, maintained guidance, and
  changelog evidence without changing APIs, models, prompts, direct dependency
  inputs, or fixtures.

## Verification Completed

- Three focused recommendation cases passed, including equal-score preference
  and the existing product-backed fallback.
- The complete no-network suite passed with 99 tests in a disposable Python
  3.12 environment installed from the exact test lock.
- All four Make gates passed from the repository root and an external directory.
- Eight isolated hostile mutations were rejected across score direction,
  own-industry preference, regression coverage/assertion, README, changelog, and
  completed-plan status.
- The disposable environment passed dependency integrity checks, and the
  separately documented transitive lock remediation restored zero-finding audits.
- Exact diff, generated artifact, credential, conflict marker, binary,
  large-file, mode, dependency-lock, fixture, and whitespace audits passed.
