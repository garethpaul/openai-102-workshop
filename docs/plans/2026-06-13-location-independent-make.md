# Location-Independent Workshop Tooling

status: completed

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

## Work Completed

- Rooted build, test, static, runtime, smoke, lock, audit, and run commands at
  the checkout containing the loaded Makefile while preserving the target graph
  and `PYTHON` and `UV` overrides.
- Added an exact Makefile contract plus README and completed-plan evidence to
  `scripts/check-workshop-baseline.py`.
- Documented absolute Makefile invocation without changing lesson or dependency
  behavior.

## Verification Completed

- Root and external-directory `lint`, `test`, `build`, `verify`, and `check`
  gates passed with 61 no-network tests per test invocation in the existing
  isolated Python 3.12 environment.
- Root and external-directory `runtime-check` and `smoke` passed without OpenAI
  credentials or paid API calls.
- Ten isolated hostile mutations covering root derivation, build, test, static,
  lock, lock-check, runtime, smoke, completed plan evidence, and README guidance
  were rejected by the intended contracts.
- Python compilation, `git diff --check`, intended-path, secret-pattern,
  generated-artifact, lesson, API, model, dependency-input, lock, Docker,
  Pipenv, fixture, test, and workflow audits passed.
