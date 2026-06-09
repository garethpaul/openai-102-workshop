# Finite Embedding Values

status: completed

## Context

Retrieval fixture loading validates row shape, embedding dimensionality, and
metadata text before nearest-neighbor training. Non-numeric, NaN, or infinite
embedding values could still reach model training and fail with less useful
errors.

## Objectives

- Require every embedding value to be numeric.
- Reject NaN and infinite embedding values.
- Keep fixture validation no-network and API-free.
- Extend tests, docs, and the static baseline for finite embedding values.

## Verification

- `make check`
- `python3 -m pytest -q test_app.py`
- `python3 scripts/check-workshop-baseline.py`
- `git diff --check`
