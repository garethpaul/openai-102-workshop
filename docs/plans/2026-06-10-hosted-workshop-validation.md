# Hosted Workshop Validation

status: completed

## Context

The workshop has a fast no-network `make check` baseline for retrieval math,
fixture validation, cache path safety, and documentation contracts. It has no
hosted validation, while the application requirements file is intentionally
large and unsuitable for a focused test job.

## Priorities

1. Run the canonical no-network baseline for pushes and pull requests.
2. Pin workflow actions, Python, permissions, runner, timeout, and concurrency.
3. Install only the dependencies imported by the maintained test surface.
4. Run `pip check` before tests to expose incompatible dependency resolution.
5. Keep API credentials and paid OpenAI requests out of hosted validation.

## Implementation Units

Files:

- `.github/workflows/check.yml`
- `requirements-test.txt`
- `scripts/check-workshop-baseline.py`
- `README.md`
- `SECURITY.md`
- `VISION.md`
- `CHANGES.md`

Add a commit-pinned, read-only Python 3.10 workflow on `ubuntu-24.04`. Cache and
install the explicit test dependency set, validate the environment with
`python -m pip check`, and run `make check`. Require that workflow and
dependency contract from the static checker.

## Verification

- `make lint`
- `make test`
- `make build`
- `make check`
- workflow YAML parse
- `git diff --check`
- successful hosted Linux `Check` workflow for the pushed commit

## Boundaries

- Do not make OpenAI API calls or require credentials in tests.
- Do not install the full workshop application dependency set in hosted CI.
- Do not migrate the legacy OpenAI SDK examples in this pass.
