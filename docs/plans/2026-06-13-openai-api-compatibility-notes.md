# OpenAI API Compatibility Notes

status: completed

## Context

The workshop intentionally pins `openai==0.28.1`, but several learner-facing
examples present module-level Completion, ChatCompletion, Embedding, and legacy
fine-tuning calls without a consistent compatibility warning. Historical model
names can therefore be mistaken for current OpenAI API guidance.

## Priorities

1. Inventory the preserved legacy SDK calls and model identifiers by lesson.
2. Add visible compatibility warnings where learners encounter those examples.
3. Link current official OpenAI model and API guidance without selecting a new
   model or changing lesson behavior.
4. Preserve the pinned dependency graph and no-network validation boundary.

## Implementation Units

### Compatibility Inventory

File: `docs/openai-api-compatibility.md`

Record the legacy SDK surface, affected files, historical model identifiers,
current official documentation links, and migration boundaries as of
2026-06-13.

### Learner Warnings

Files:

- `pages/1_🧐_Getting_Started.py`
- `pages/2_⚡️_API.py`
- `pages/2_📝_Embeddings.py`
- `pages/3_🔍_Text_Search.py`
- `pages/8_🦾_FineTuning.py`
- `README.md`
- `SECURITY.md`
- `VISION.md`
- `CHANGES.md`

Mark the preserved examples as historical and direct learners to the inventory
before using them for a new integration.

### Static Contract

Files:

- `scripts/check-workshop-baseline.py`
- `docs/plans/2026-06-13-openai-api-compatibility-notes.md`

Require the inventory, official links, visible warnings, unchanged legacy SDK
pin, completed status, and verification evidence.

## Verification Plan

- `python3 -m py_compile scripts/check-workshop-baseline.py` and changed pages
- `make lint`
- `make test`
- `make build`
- `make check`
- run the checker outside the repository working directory
- parse workflow YAML and dependency manifests
- run focused hostile mutations against compatibility-note contracts
- verify OpenAI calls, prompts, model strings, dependency inputs, and lock files
  have no behavioral diff
- `git diff --check`
- scan the intended diff for secrets and generated artifacts

## Boundaries

- Do not migrate SDK calls, model identifiers, prompts, fine-tuning data, or
  response parsing in this unit.
- Do not claim a historical example remains callable without credentialed live
  API verification.
- Do not choose a replacement model before a dedicated compatibility migration.
- Do not make validation perform OpenAI API calls.

## Work Completed

Added a dated compatibility inventory, learner-visible warnings, current
official documentation links, and a static contract while preserving the
historical SDK, model identifiers, prompts, and dependency graph.

## Verification Completed

- `python3 -m py_compile scripts/check-workshop-baseline.py` and the changed
  learner pages passed.
- `make lint`, `make test`, `make build`, and `make check` passed using the
  locked test graph in an isolated `uv` environment where dependencies were
  required.
- The checker passed from an external working directory.
- Ten focused `hostile mutations rejected` altered warnings, inventory facts,
  preserved dependency/API surfaces, and incomplete plan evidence.
- `legacy API behavior paths had no diff`, and each changed learner page
  differed only by its compatibility warning.
- `git diff --check`, workflow parsing, dependency-manifest checks, and the
  intended-diff secret/artifact scan passed.
