# Hash-Verified Universal Locks

status: in_progress

## Context

The workshop's Python 3.12 application and verification locks are complete,
exactly pinned, reproducible from public PyPI, and audit-clean. They do not yet
record distribution hashes, so an installer can confirm package names and
versions but cannot enforce the exact artifact digests selected from the
package index. The existing `uv` lock workflow supports universal hash
generation without changing dependency inputs or selected versions.

## Prioritized Requirements

- P0. Generate both universal locks with `uv pip compile --generate-hashes`
  while preserving every existing package version and reviewed security floor.
- P0. Require hashes whenever CI or documented verification installs either
  generated lock. Pip automatically enables hash-checking mode when every
  requirement carries hashes; explicit local and container commands should
  retain `--require-hashes` for clarity.
- P1. Keep lock regeneration pinned to the repository-owned public-PyPI source
  and deterministic under hostile caller index environment variables.
- P1. Add mutation-sensitive static contracts for both compile commands, all
  lock-install call sites, and representative hash coverage.
- P2. Document the intentionally larger universal lock files and the boundary
  between artifact integrity, vulnerability auditing, and provenance.

## Implementation Units

### U1. Hash-generating lock workflow

**Files:** `Makefile`, `requirements.txt`, `requirements-test.txt`

Add `--generate-hashes` to normal and upgrade lock generation, regenerate both
Python 3.12 universal locks from unchanged inputs, and verify package/version
pins do not drift.

### U2. Hash-enforced installation

**Files:** `.github/workflows/check.yml`, repository verification helpers, and
maintained setup guidance where exact locks are installed.

Pass `--require-hashes` for application and verification lock installation.
Preserve the separate no-build vulnerability audit and full application smoke.

### U3. Contracts and maintenance evidence

**Files:** `scripts/check-workshop-baseline.py`, focused tests, `README.md`,
`SECURITY.md`, `VISION.md`, `CHANGES.md`, and this plan.

Require hash-generating compile commands, hash-enforced install commands,
complete hash annotations, unchanged dependency selections, and completed
verification evidence.

## Verification

- Prove pre-change locks lack artifact hashes and record the generated-lock
  size increase truthfully.
- Run focused static and test contracts, the complete no-network suite, all
  canonical Make gates, and the absolute Makefile check externally.
- Run `make lock-check` at least twice under hostile caller index variables.
- Install both locks in fresh Python 3.12 environments with
  `--require-hashes`, run `uv pip check`, runtime imports, and the
  credential-free Streamlit smoke.
- Audit both exact locks without builds and reject mutations removing compile
  hashing, install enforcement, representative hashes, guidance, or completed
  plan evidence.
- Audit exact paths, dependency pin drift, generated artifacts, credentials,
  conflict markers, modes, large files, and whitespace before commit.

## Scope Boundaries

- Do not change direct dependency inputs, selected package versions, workshop
  lessons, prompts, model identifiers, fixtures, or live API behavior.
- Do not claim hashes establish publisher identity or external provenance;
  they enforce index artifact integrity for the reviewed lock contents.
- Do not make live OpenAI requests, use credentials, or merge or close any
  stacked pull request.

## Work Completed

- Added hash generation to normal and upgrade compilation for both universal
  Python 3.12 locks without changing any dependency input or selected pin.
- Required hashes for CI application and verification installs through the
  fully hashed lock files, and made enforcement explicit in the container
  build and documented local application setup.
- Added baseline contracts requiring every exact pin to begin a SHA-256 hash
  block and requiring hash enforcement at maintained install call sites.
- Documented the intentionally larger lock files and the distinction between
  artifact integrity and publisher provenance.

## Verification Pending

- A fresh Python 3.12 verification environment installed all 73 test-lock
  packages with `--require-hashes`; the full no-network suite passed all 108
  tests and the static baseline passed.
- Consecutive lock generation preserved every application and verification
  package/version selection; only generated hashes and command provenance were
  added to the locks.
- The local public index available to this host does not expose the checked-in
  future `pyarrow==24.0.0` wheel, so a local application-lock hash install
  cannot reproduce the already hosted application graph. The canonical
  exact-head `application-smoke` job must complete successfully before this
  plan can be marked completed.
- Final hostile mutations, exact diff and artifact audits, commit/push, and
  exact-head hosted evidence remain pending.
