# Universal Lock Audit

status: completed

## Context

`make audit` asks `pip-audit` to resolve and install both universal Python 3.12
locks before auditing them. The application lock currently includes
`pyarrow==24.0.0`; on this supported Linux environment, dependency collection
falls back to a source build and fails because a transitive build dependency
requires Rust. The exact lock is already complete, and the hosted
`application-smoke` job separately installs it and runs `pip check`.

## Requirements

- R1. Audit every exact package version in both generated locks without asking
  pip to build or resolve the dependency graph again.
- R2. Preserve public-PyPI lock generation, the reviewed Starlette and aiohttp
  security floors, and every existing exact dependency pin.
- R3. Preserve the separate full application install, compatibility check,
  runtime import check, and credential-free Streamlit smoke.
- R4. Add mutation-sensitive static and test contracts for the no-resolution
  audit mode.
- R5. Run without credentials, paid API calls, Rust installation, or unrelated
  dependency upgrades.

## Implementation Units

### U1. Exact-lock audit command

**Files:** `Makefile`, `scripts/check-workshop-baseline.py`

Run `pip-audit` with `--no-deps --disable-pip` for both complete, exact locks
and keep the repository-owned public-PyPI index contract.

### U2. Maintained verification and guidance

**Files:** `test_app.py`, `README.md`, `SECURITY.md`, `CHANGES.md`, and this
plan.

Require both flags on both audit commands and document why application
installation remains a separate hosted gate.

## Verification

- Run the focused static and pytest contracts for the audit target.
- Run `make audit`, `make lock-check`, `make check`, runtime imports, and the
  credential-free Streamlit smoke with bounded commands.
- Run the repository baseline from both the repository and an external
  directory.
- Reject isolated mutations removing either audit flag or restoring resolver
  use for either lock.
- Audit the exact diff, generated artifacts, added lines for credentials,
  modes, and whitespace.

## Scope Boundaries

- Do not change either dependency input or generated lock.
- Do not weaken exact-version, security-floor, lock-regeneration, or full
  application-install coverage.
- Do not install Rust or make live OpenAI requests.

## Verification Completed

- `make audit` used `--no-deps --disable-pip` for both exact locks and reported
  no known vulnerabilities without invoking package builds.
- `make lock-check` passed twice against public PyPI with no generated-lock
  drift, and the complete no-network suite passed with 108 tests.
- The existing application environment passed compatibility checks for 108
  packages, every direct runtime import, and the credential-free Streamlit
  health smoke; hosted `application-smoke` retains the same full-lock install.
- Four isolated hostile mutations removing either no-resolution flag from
  either the Makefile command or its static fixture were rejected.
- `make check` passed from both the repository and an external working
  directory before the final diff, artifact, secret, mode, and whitespace
  audits.
