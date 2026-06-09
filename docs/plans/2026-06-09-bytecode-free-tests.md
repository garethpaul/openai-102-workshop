# Bytecode-Free Tests

status: completed

## Context

The workshop already ran static checks and compile checks with
`PYTHONDONTWRITEBYTECODE=1`, but `make test` invoked pytest without that guard.
Normal verification could leave ignored `__pycache__` files behind, which made
completed workspaces noisier than necessary.

## Objectives

- Run `make test` with Python bytecode writes disabled.
- Extend the static baseline so leftover Python bytecode is detected.
- Keep the existing no-network pytest suite and Makefile gate shape.
- Document the bytecode-free verification contract.

## Verification

- `make lint`
- `make test`
- `make build`
- `make check`
- `git diff --check`
