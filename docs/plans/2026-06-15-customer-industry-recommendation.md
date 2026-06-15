# Customer Industry Recommendation

status: completed

## Summary

Make the Recommendations page choose an industry from the selected customer's
embedding similarity instead of always selecting the first industry in the
JSON mapping.

## Problem

`pages/4_🤞_Recommendations.py` loads `customer_embedding` but never uses it.
It builds an industry-to-industry similarity matrix and then assigns
`top_industry` from the first dictionary key, so the recommendation is
independent of the selected customer. If that first industry has no configured
products, `np.random.choice` can also fail on an empty list.

## Requirements

- Compare the selected customer's industry embedding with every available
  industry embedding.
- Select the industry with the highest customer-relative cosine similarity.
- Keep the pairwise similarity matrix used by the existing visualizations.
- Return no recommendation when the customer, source embedding, or selected
  industry's product mapping is unavailable.
- Validate vectors through the maintained finite, non-empty,
  equal-dimensional similarity contract.
- Add deterministic tests that reject first-key selection and empty-product
  crashes without importing Streamlit page side effects.

## Implementation Units

### Pure Recommendation Logic

Files: `components/recommendations.py`, `pages/4_🤞_Recommendations.py`

- Extract customer lookup, customer-relative scoring, pairwise scoring, and
  product selection into import-safe pure helpers.
- Reuse `utils.generate.cosine_similarity` for vector validation and scoring.
- Keep the page responsible for loading checked-in JSON and rendering output.

### Regression Coverage

Files: `test_app.py`, `scripts/check-workshop-baseline.py`

- Prove a customer whose nearest industry is not the first mapping key selects
  the nearest industry's product.
- Prove missing customers, source industries, and product mappings return a
  stable no-recommendation result.
- Add static contracts that reject restoring first-key selection or bypassing
  the maintained similarity helper.

### Maintenance Evidence

Files: `README.md`, `SECURITY.md`, `CHANGES.md`, this plan

- Document customer-relative recommendation selection and controlled fallback
  behavior.
- Record focused, full-suite, mutation, artifact, and hosted verification after
  implementation.

## Verification

- Run focused recommendation tests, the full no-network test suite, every Make
  gate, application smoke checks, production/test dependency audits, and
  external-directory `make check`.
- Reject isolated mutations that restore first-key selection, ignore the
  customer embedding, bypass validated cosine similarity, call random choice
  for empty products, remove regression coverage, or leave plan evidence
  incomplete.
- Audit exact diff, generated artifacts, conflict markers, binary/large files,
  and changed-line credential patterns.

## Risks

- Product mappings remain intentionally sparse workshop data; the safe result
  for an unmapped nearest industry is no recommendation.
- No paid OpenAI request or live Streamlit interaction is required for this
  deterministic checked-in-data fix.
- The change must remain stacked on PR #13 and must not be merged or closed
  without explicit owner authorization.

## Verification Completed

- All four focused recommendation tests passed in the isolated managed Python
  3.12.12 environment.
- The complete no-network test suite, every Make gate, application smoke
  checks, and `make check` from an external working directory passed.
- Production and test dependency audits reported no known vulnerabilities on
  the exact locked dependency graphs.
- Six isolated hostile mutations restoring first-key selection, ignoring the
  customer embedding, bypassing validated cosine similarity, selecting from an
  empty product list, removing regression coverage, or reverting completed
  plan evidence were rejected.
- `git diff --check`, exact-diff, generated-artifact, conflict-marker,
  intended-path, binary, large-file, and changed-line credential audits passed.
- No paid OpenAI request or live Streamlit interaction was used.
