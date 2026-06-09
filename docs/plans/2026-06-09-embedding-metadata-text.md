# Embedding Metadata Text Validation

status: completed

## Context

`load_embeddings_and_train_model` validates fixture row shape and embedding
dimensions before training nearest-neighbor models. Retrieval helpers later
assume each metadata row includes `text` when building augmented queries, so a
fixture with missing or blank text could still fail later with an unclear key
error.

## Objectives

- Preserve no-network embedding fixture loading.
- Reject fixture rows whose metadata is not a dictionary with non-empty text.
- Add pytest coverage for missing metadata text.
- Extend the static baseline and docs so retrieval fixture shape remains clear.

## Verification

- `make check`
- `python3 -m pytest -q test_app.py`
- `python3 scripts/check-workshop-baseline.py`
- `git diff --check`
