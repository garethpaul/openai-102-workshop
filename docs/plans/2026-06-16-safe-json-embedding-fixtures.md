# Safe JSON Embedding Fixtures

status: in_progress

## Context

The shared nearest-neighbor fixture loader executes unrestricted pickle data,
and `embeddings_demo_step4.py` downloads a remote pickle before loading it.
Pickle deserialization can execute code before the existing row validation
runs, so a replaced local fixture or compromised download crosses the workshop
security boundary before shape, metadata, or numeric checks apply.

## Priority

Remove executable deserialization and the hidden remote download before adding
more retrieval examples. This closes a direct code-execution path in runnable
lesson code while retaining the workshop's no-network fixture workflow.

## Requirements

- R1. Replace unrestricted pickle fixture loading with UTF-8 JSON parsing in
  the shared nearest-neighbor loader.
- R2. Preserve existing row-shape, nonempty, dimensionality, metadata-text,
  numeric-type, and finite-value validation before model training.
- R3. Make the step-4 demo use an explicit local JSON fixture and fail clearly
  when it is absent; do not download or deserialize a remote pickle.
- R4. Keep the loaded embedding structure compatible with NumPy and
  `NearestNeighbors`, including small fixtures and metadata ordering.
- R5. Replace the tracked test pickle with a reviewable JSON fixture and update
  fixture-generation guidance without retaining pickle imports or calls in
  application/demo source.
- R6. Add runtime and static contracts for valid JSON, malformed JSON,
  wrong top-level types, missing files, and forbidden pickle/network fallback.
- R7. Update security, contributor, workshop, changelog, and completed-plan
  guidance without making live OpenAI calls or external downloads.

## Implementation Units

### U1. Safe shared fixture loader

**Files:** `utils/generate.py`, `test_app.py`, `test_embeddings.json`

Parse explicit JSON fixtures, retain all semantic validation, and prove valid
and malformed inputs fail at the intended boundary without executable
deserialization.

### U2. Safe lesson path

**Files:** `embeddings_demo_step4.py`, `create.py`

Use a local JSON fixture in the step-4 example, remove the hidden download and
pickle path, and generate the small sample fixture as JSON.

### U3. Maintained contracts and guidance

**Files:** `scripts/check-workshop-baseline.py`, `README.md`, `SECURITY.md`,
`AGENTS.md`, `VISION.md`, `CHANGES.md`, and this plan.

Add mutation-sensitive source, runtime, documentation, and completed-plan
contracts while keeping the existing complete repository gate authoritative.

## Verification

- Run focused JSON fixture loader tests, then the complete no-network suite.
- Run repository and external-directory `make check` with the pinned locks.
- Reject isolated mutations that restore pickle loading, remote downloading,
  missing-file ambiguity, or bypass malformed/top-level fixture checks.
- Audit the exact diff, generated artifacts, credential patterns, dependency
  and workflow drift, file modes, and whitespace before commit.

## Scope Boundaries

- Do not make live OpenAI, Twilio, storage, or other external requests.
- Do not change embedding models, recommendation ranking, or API compatibility.
- Do not regenerate dependency locks or broaden this change into all workshop
  datasets; only executable embedding fixture paths are in scope.
