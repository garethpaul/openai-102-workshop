# Product Recommendation Name Validation

status: completed

## Context

Recommendation selection now requires a nonempty product list, but it does not
validate list members. Catalog values such as `None`, objects, or blank strings
can therefore make an industry appear product-backed and reach the selection
callback as malformed recommendations.

## Goal

Treat an industry as product-backed only when it has at least one nonempty
string product name, and pass only validated names to product selection.

## Scope

- Filter product-list members to nonempty strings without mutating source data.
- Preserve customer-relative industry scoring and nearest valid-industry
  selection.
- Return no product when every catalog entry is malformed or blank.
- Add focused tests, static contracts, synchronized guidance, and completed
  verification evidence.

## Verification Plan

- Run focused recommendation cases, the no-network suite, all Make gates, the
  external-directory gate, runtime imports, Streamlit health smoke, and audit.
- Reject isolated mutations of type validation, blank rejection, filtered-list
  handoff, tests, guidance, and plan evidence.
- Run `git diff --check` and audit generated artifacts, dependency files,
  credentials, binaries, modes, and intended paths.

## Work Completed

- Filtered each product mapping to trimmed, nonempty string names without
  mutating the source catalog.
- Scored industries as before while passing only validated names to product
  selection.
- Added focused malformed-member and all-invalid-list coverage plus
  mutation-sensitive static and documentation contracts.

## Verification Completed

- Eight focused recommendation cases and the 82-test no-network suite passed in
  an isolated Python 3.12 environment installed from both checked-in locks.
- `make lint`, `make test`, `make build`, and `make check` passed from the
  repository, and `make check` passed from an external working directory.
- Runtime imports and the Streamlit health smoke passed against the full locked
  application graph.
- `pip-audit --no-deps --disable-pip` reported no known vulnerabilities for
  both exact lock files; plain `make audit` could not collect the old source
  build graph because the host lacks a Rust compiler, so no successful plain
  audit is claimed.
- Six isolated hostile mutations covering type validation, blank rejection,
  filtered-list handoff, focused tests, guidance, and plan evidence were
  rejected.
- `git diff --check` plus generated-artifact, dependency-file, credential,
  binary, mode, and intended-path audits passed.
