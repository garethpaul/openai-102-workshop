# Hash-Verified Universal Locks

status: planned

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
  generated lock, so the new metadata is enforced rather than decorative.
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
