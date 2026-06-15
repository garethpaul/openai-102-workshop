# Product Recommendation Name Validation

status: in_progress

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

Pending implementation.

## Verification Completed

Pending implementation and validation.
