---
title: Recommendation Tie Breaking
status: planned
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
- Prove a strictly higher product-backed score still wins.
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
