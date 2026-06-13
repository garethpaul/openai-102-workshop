# Location-Independent Workshop Tooling

status: in progress

## Context

Make recipes resolve lesson modules, tests, checkers, dependency inputs, and
the Streamlit entry point from the caller's working directory. Absolute
Makefile invocation from elsewhere can therefore fail, inspect the wrong tree,
or generate lock files outside the checkout.

## Objectives

- Resolve build, test, static, runtime, smoke, lock, audit, and run commands
  from the checkout containing the loaded Makefile.
- Preserve the existing target graph plus `PYTHON` and `UV` overrides.
- Enforce the exact rooted recipes, operator guidance, completed status, and
  verification evidence in the active baseline checker.
- Prove root and external-directory verification behavior with
  mutation-sensitive checks.

## Implementation Units

### Make Contract

Files: `Makefile` and `scripts/check-workshop-baseline.py`.

Derive one absolute checkout root and execute every path-sensitive command from
it. Require the complete Makefile so targets, overrides, and path resolution
cannot drift independently.

### Documentation And Evidence

Files: `README.md`, `CHANGES.md`, and this plan.

Document absolute Makefile invocation and record bounded local, external, lock
containment, and hostile-mutation verification after it completes.

## Boundaries

- Do not change lesson pages, Python application behavior, OpenAI calls,
  prompts, model identifiers, dependency inputs or locks, Docker/Pipenv files,
  fixtures, tests, or workflows.
- Do not use credentials, make live or paid OpenAI calls, or run interactive
  Streamlit in validation.
- Preserve the existing stacked PR chain and exact-head evidence.
