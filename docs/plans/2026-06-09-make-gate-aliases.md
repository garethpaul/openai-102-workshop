# Make Gate Aliases

status: completed

## Context

The repository had working `make test`, `make build`, and `make check` targets,
but the shared maintenance workflow also runs `make lint` before tests and
expects `make check` to delegate through a named verification target. The
aliases should preserve the no-network static and pytest baseline.

## Objectives

- Add `make lint` as the SDK-free static workshop baseline.
- Keep `make test` on the no-network pytest suite.
- Keep `make build` as a bytecode-free Python compile gate for maintained
  modules.
- Route `make check` through `make verify`.
- Extend docs and the static baseline for the standard gate contract.

## Verification

- `make lint`
- `make test`
- `make build`
- `make check`
- `git diff --check`
