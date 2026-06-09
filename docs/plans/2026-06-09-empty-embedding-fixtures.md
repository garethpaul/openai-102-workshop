# Empty Embedding Fixtures

status: completed

## Context

`load_embeddings_and_train_model` now supports small embedding fixtures, but an
empty fixture still failed later through tuple unpacking. Workshop fixtures
should fail with a clear local validation error before nearest-neighbor model
training starts.

## Objectives

- Preserve no-network fixture loading and nearest-neighbor training behavior.
- Reject empty saved embedding lists with a clear `ValueError` naming the
  missing embedding fixture row.
- Add pytest coverage for the empty fixture path.
- Extend the static baseline and docs so malformed fixture behavior remains
  visible.

## Verification

- `make check`
- `python3 -m pytest -q test_app.py`
- `python3 scripts/check-workshop-baseline.py`
- `git diff --check`
