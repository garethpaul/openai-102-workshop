# Vector Math Validation

status: completed

## Context

The workshop retrieval helpers include simple vector distance and similarity
functions used by embedding lessons. `cosine_similarity` silently truncated
mismatched vector dimensions through `zip()` and could divide by zero when a
fixture vector had zero magnitude.

## Objectives

- Preserve the legacy OpenAI SDK examples and no-network test strategy.
- Raise clear `ValueError` exceptions for invalid `cosine_similarity` inputs.
- Cover dimension mismatch and zero-vector cases in `test_app.py`.
- Extend the static baseline and docs for vector math validation.

## Verification

- `make check`
- `python3 -m pytest -q test_app.py`
- `python3 scripts/check-workshop-baseline.py`
- `git diff --check`

The checks do not call OpenAI or install the full workshop dependency set.
