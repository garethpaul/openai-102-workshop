# CI Baseline

status: completed

## Context

The repository had a local no-network `make check` baseline for workshop helper
tests, cache guardrails, and static documentation checks, but no hosted
workflow ran it for pushes and pull requests.

## Changes

- Added a GitHub Actions workflow that installs Python 3.10 and the minimal
  no-network test dependencies before running `make check`.
- Extended the static baseline and docs so the hosted CI path stays visible.

## Verification

- `make check`
