---
title: Transitive Dependency Security Update
status: completed
date: 2026-06-16
---

# Transitive Dependency Security Update

## Context

The exact application and verification locks resolved `aiohttp==3.14.0`
through the preserved `openai==0.28.1` workshop dependency. A fresh audit
reported eight request-processing advisories fixed by aiohttp 3.14.1. The same
audit found two request-processing advisories in application-only
`starlette==1.2.1`, fixed by 1.3.1.

## Requirements

- Keep the direct dependency inputs and legacy OpenAI lesson API unchanged.
- Upgrade only the affected transitive pins in their generated locks.
- Preserve reproducible Python 3.12 universal lock regeneration.
- Reject either rollback in the dependency-free baseline checker.
- Re-run compatibility, audit, runtime, smoke, test, and external-directory
  gates before shipping.

## Work Completed

- Used the existing generated locks as resolver preferences and confirmed that
  targeted upgrades change only the affected pins.
- Updated both locks from aiohttp 3.14.0 to 3.14.1 and the application lock
  from Starlette 1.2.1 to 1.3.1.
- Added static lock, plan, security guidance, and changelog contracts for both
  reviewed security floors.

## Verification Completed

- `requirements.txt` and `requirements-test.txt` both retain
  `aiohttp==3.14.1`, while the application lock retains `starlette==1.3.1` and
  every other generated pin remains unchanged.
- `make lock-check` regenerated both Python 3.12 universal locks without drift.
- `pip-audit` checked both exact locks and reported no known vulnerabilities.
- Clean lock compatibility checks, runtime imports, the bounded Streamlit
  smoke, the complete no-network suite, all Make gates, and external-directory
  execution passed.
- Isolated lock, guidance, and completed-plan mutations were rejected.
