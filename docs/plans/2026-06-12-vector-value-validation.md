# Vector Value Validation

status: completed

## Context

Embedding fixtures and retrieval queries reject nonnumeric and non-finite
values before nearest-neighbor operations. The standalone cosine, Euclidean,
and Manhattan workshop helpers only check dimensions, so empty vectors,
booleans, strings, complex values, `NaN`, infinity, and overflowing integers
can still produce misleading results or leak low-level exceptions.

## Priority

These helpers teach vector math directly. They should fail consistently and
clearly for invalid numeric inputs rather than demonstrating silent `NaN`
propagation or Python type-coercion edge cases.

## Prioritized Engineering Backlog

1. Share one numeric finite vector-pair validator across all distance helpers.
2. Expand retrieval tests if additional vector operations are introduced.
3. Separate educational math helpers into a small dependency-light module if
   the workshop grows beyond the current utility file.

## Requirements

- R1. Cosine, Euclidean, and Manhattan helpers must reject empty vectors.
- R2. All three helpers must reject booleans, strings, complex values, `NaN`,
  infinity, and values that overflow float conversion.
- R3. Dimension mismatch behavior must remain a clear `ValueError`.
- R4. Cosine similarity must continue to reject zero-magnitude vectors.
- R5. Valid Python and NumPy numeric scalars must remain supported.
- R6. Tests must run without API credentials or network calls.
- R7. README, security guidance, vision, changes, and the static baseline must
  document and protect the shared vector-value boundary.

## Scope Boundaries

- Do not alter embedding fixtures, OpenAI SDK examples, or model calls.
- Do not coerce strings or booleans into numeric values.
- Do not add dependencies.

## Verification

- `make lint`
- `make test`
- `make build`
- `make check`
- `PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q test_app.py`
- `git diff --check`

## Work Completed

- Added one numeric finite vector-pair validator shared by cosine, Euclidean,
  and Manhattan helpers.
- Added parameterized no-network tests for empty, boolean, string, complex,
  non-finite, and overflowing values across all three helpers.
- Preserved valid Python and NumPy numeric scalar support, dimension mismatch
  errors, and zero-vector cosine behavior.
- Updated the workshop baseline and maintenance documentation.
