# Malformed Embedding Fixtures Plan

status: completed

## Context

The workshop loads trusted pickle fixtures for no-network nearest-neighbor
examples. Empty fixtures already fail with a clear validation error, but rows
with missing fields or inconsistent embedding dimensions could still reach
tuple unpacking or NumPy stacking errors.

## Objectives

- Validate that each embedding fixture row contains id, embedding, and metadata.
- Validate that each embedding has at least one dimension.
- Validate that all fixture embeddings have the same dimensionality.
- Add no-network tests and static checks for malformed fixture failures.

## Verification

- `make check`
- `python3 -m pytest -q test_app.py`
- `git diff --check`
