# Small Embedding Fixtures Plan

status: completed

## Context

`load_embeddings_and_train_model` always configured five nearest neighbors.
Small workshop fixtures with fewer than five embedding rows could load and fit,
then fail when queried.

## Objectives

- Cap `n_neighbors` at the number of loaded embedding rows.
- Extend pytest coverage so a two-row fixture can be queried.
- Document the small fixture behavior in the workshop baseline.

## Verification

- `python3 -m pytest -q test_app.py`
- `make check`
- `python3 scripts/check-workshop-baseline.py`
- `git diff --check`
