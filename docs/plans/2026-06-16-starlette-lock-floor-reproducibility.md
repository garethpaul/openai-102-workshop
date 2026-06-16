# Starlette Lock-Floor Reproducibility

status: completed

## Context

The generated application lock intentionally contains `starlette==1.3.1` to
avoid advisories in 1.2.1, but `requirements.in` does not constrain Starlette.
A fresh `make lock-check` therefore regenerated the lock at 1.2.1 and failed
the repository's own security contract.

## Requirements

- R1. Encode `starlette==1.3.1` as an explicit application resolver floor.
- R2. Regenerate the application lock and preserve every unrelated pin.
- R3. Make `make lock-check` use public PyPI and reproducibly retain the
  reviewed security version regardless of caller index configuration, and use
  the same source for vulnerability audits.
- R4. Keep the verification lock, OpenAI SDK pin, Streamlit pin, and lesson
  behavior unchanged.
- R5. Add mutation-sensitive input, lock, documentation, and completed-plan
  contracts.
- R6. Run without credentials, paid API calls, or dependency upgrades beyond
  the already reviewed Starlette version.

## Implementation Units

### U1. Resolver input and generated lock

**Files:** `requirements.in`, `requirements.txt`, `Makefile`

Add the exact reviewed Starlette constraint to the application input and
regenerate the universal Python 3.12 lock from the repository-owned public-PyPI
index contract.

### U2. Maintained validation and guidance

**Files:** `scripts/check-workshop-baseline.py`, `AGENTS.md`, `README.md`,
`SECURITY.md`, `VISION.md`, `CHANGES.md`, and this plan.

Require the input floor as well as the generated pin and explain why the
transitive security version is now resolver input.

## Verification

- Run `make lock-check` twice to prove stable regeneration.
- Run `uv pip check`, all Make gates, the external-directory gate, both lock
  audits, runtime imports, and the credential-free Streamlit smoke.
- Reject isolated mutations for the input floor, generated pin, checker,
  guidance, and completed plan.
- Audit the exact diff, generated artifacts, added lines for credentials,
  unrelated lock drift, modes, and whitespace.

## Scope Boundaries

- Do not change `requirements-test.in` or `requirements-test.txt`.
- Do not upgrade Streamlit, OpenAI, models, or any unrelated dependency.
- Do not make live OpenAI requests or require an API key.

## Verification Completed

- make lock-check passed twice from the public-PyPI contract while the caller
  exported a conflicting package index.
- The complete no-network suite passed with 101 tests.
- Both exact locks audited with no known vulnerabilities.
- Application lock compatibility, runtime imports and credential-free Streamlit smoke passed
  in a fresh 108-package Python 3.12 environment.
- Six isolated hostile mutations were rejected for the input floor, generated
  pin, public index contract, checker contract, guidance, and plan status.
- Exact diff, generated-artifact, added-line secret, unrelated lock drift,
  mode, and whitespace audits passed before commit.
