# Fine-Tuning Rate-Limit Retry Example

status: completed

## Context

The displayed fine-tuning data-generation example catches every exception and
labels it as a rate limit. Its attached `else: raise` cannot execute for a
failed request because the `except` always continues, so authentication,
validation, and programming errors are silently retried and misreported.

## Priority

Correct the learner-facing failure boundary before broader modernization. The
workshop intentionally preserves `openai==0.28.1`, and that pinned SDK exposes
`openai.error.RateLimitError` for this historical module-level example.

## Requirements

- R1. Retry only `openai.error.RateLimitError` in the displayed example.
- R2. Re-raise the final rate-limit failure instead of falling through with
  unassigned response variables.
- R3. Let authentication, validation, transport, and programming errors
  propagate immediately without being mislabeled.
- R4. Preserve exponential backoff with jitter for retryable attempts.
- R5. Add mutation-sensitive source, checker, documentation, and completed-plan
  contracts.
- R6. The retry correction must not call OpenAI, add credentials, change
  dependencies, or migrate the historical `openai==0.28.1` API surface.

## Implementation Units

### U1. Displayed retry example

**File:** `pages/8_🦾_FineTuning.py`

Replace the bare exception handler with the pinned SDK's rate-limit exception,
raise on the final attempt, and remove the unreachable `try`/`else` branch.

### U2. Maintained validation

**Files:** `test_app.py`, `scripts/check-workshop-baseline.py`

Verify the displayed code keeps the narrow exception, final-attempt raise,
backoff, and no bare handler or unreachable exception branch.

### U3. Maintained guidance

**Files:** `AGENTS.md`, `README.md`, `SECURITY.md`, `VISION.md`, `CHANGES.md`,
and this plan.

Record that sample retry guidance must distinguish rate limits from other API
failures while retaining the historical compatibility warning.

## Verification

- Run focused static tests and the complete no-network suite.
- Run lint, test, build, check, lock verification, and external-directory check
  under explicit timeouts.
- Reject isolated mutations for the exception type, final-attempt raise,
  backoff, test contract, guidance, and plan status.
- Audit the exact diff, generated artifacts, added lines for credentials,
  dependency/lock drift, and whitespace before commit.

## Scope Boundaries

- Do not make live OpenAI requests or require an API key.
- The retry correction must not update models, CLI commands, SDK versions, or
  lock files; the separately planned Starlette floor closes a validation
  blocker without changing this sample's dependency use.
- Do not redesign the lesson or execute the displayed generation loop.

## Verification Completed

- The focused fine-tuning retry regression passed.
- The complete no-network suite passed with 101 tests.
- All four Make gates passed, including the external directory check.
- Exact test-lock installation and `uv pip check` passed under isolated Python
  3.12 validation.
- Six isolated hostile mutations were rejected for the exception type,
  final-attempt raise, backoff, test registration, maintained guidance, and
  plan status.
- make lock-check passed twice after the Starlette resolver floor fix.
- Exact diff, generated-artifact, added-line secret, dependency/lock drift,
  mode, and whitespace audits passed.
