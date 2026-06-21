# Safe Make Root

## Problem

Whitespace-splitting Make functions and caller-controlled `MAKEFILE_LIST`
values could redirect workshop verification outside the checkout.

## Change

- Resolve the raw Makefile path with POSIX-compatible system tooling.
- Reject non-file origins for GNU Make's automatic `MAKEFILE_LIST` value.
- Add SDK-free regressions for every public target, spaces, a literal
  apostrophe, command-line and environment `ROOT`, and command-line and
  environment `MAKEFILE_LIST` injection.

## Validation

- Run the static workshop baseline and root-policy tests without API access.
- Run the complete locked test gate when the reviewed dependency graph is
  available.
- Confirm pinned Ubuntu CI and CodeQL pass at the exact pull-request head.
