# Safe Make Root

status: completed

## Problem

GNU Make list functions split absolute Makefile paths on whitespace. A caller
could also replace `MAKEFILE_LIST`, redirecting workshop verification and lock
operations to another tree.

## Change

- Resolve the raw Makefile path with POSIX-compatible system tooling.
- Reject non-file origins for GNU Make's automatic `MAKEFILE_LIST` value.
- Keep the checkout-derived root under command-line and environment `ROOT`.
- Cover all fifteen public Make targets and paths with spaces or an apostrophe.
- Cover command-line and environment `ROOT`.
- Cover command-line and environment `MAKEFILE_LIST` override attempts.

## Validation

- Run the maintained no-network verification gate from the repository and
  through a hostile absolute Makefile path from an unrelated directory.
- Confirm pinned Ubuntu CI and CodeQL pass at the exact pull-request head.

## Boundaries

- Do not change lesson code, dependency inputs or locks, workflows, fixtures,
  Docker or Pipenv files, model identifiers, prompts, or paid OpenAI paths.
