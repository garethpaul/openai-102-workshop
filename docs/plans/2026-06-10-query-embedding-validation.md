# Query Embedding Validation

status: completed

## Context

Stored embedding fixtures are validated before model training, but query
embeddings loaded from cache or API responses previously reached
nearest-neighbor lookup without equivalent checks. Malformed vectors should
fail clearly before entering scikit-learn.

## Objectives

- Reject empty or non-sequence query embeddings.
- Reject boolean, non-numeric, NaN, and infinite query values.
- Reject query dimensions that do not match the trained model.
- Prove invalid input never reaches nearest-neighbor lookup.
- Extend active docs and static checks for the query boundary.

## Verification

- `make lint`
- `make test`
- `make build`
- `make check`
- `git diff --check`
