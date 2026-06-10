# Numeric Embedding Values

status: completed

## Context

Embedding fixture validation already rejects non-finite values before
nearest-neighbor training, but stringified numbers can still pass a `float()`
coercion check while leaving object-typed fixture data in place for downstream
model code.

## Objectives

- Require numeric embedding values to be actual numeric Python or NumPy scalar
  types.
- Reject booleans and stringified numbers before model training.
- Keep fixture validation no-network and API-free.
- Extend tests, docs, and the static baseline for numeric embedding values.

## Verification

- `make test`
- `make check`
- `git diff --check`
