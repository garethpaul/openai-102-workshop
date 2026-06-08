---
title: Urgent Pinned Embeddings Pickle
type: fix
status: active
date: 2026-06-08
origin: public repository security audit
execution: code
---

# Urgent Pinned Embeddings Pickle

## Summary

Pin and verify the remote `embeddings.pkl` artifact before the workshop demo deserializes it with pickle.

## Problem Frame

`embeddings_demo_step4.py` downloaded `embeddings.pkl` from Google Storage when the file was missing and immediately loaded it with `pickle.load`. The Docker build also fetched the same pickle without an integrity check. Pickle deserialization can execute code, so a changed remote artifact would become code execution for anyone running the workshop script or image build.

## Requirements

- R1. The demo script must verify the SHA-256 of `embeddings.pkl` before deserialization.
- R2. The demo script must fail closed on failed downloads and checksum mismatches.
- R3. The Docker build must verify the downloaded artifact before continuing.
- R4. A guard script must detect missing checksum verification.
- R5. The GitHub issue and PR must be marked `URGENT`.

## Implementation Unit

### U1. Pinned Pickle Artifact

- **Goal:** Add an expected SHA-256, verify downloaded/local pickle bytes, and keep the Docker artifact fetch pinned to the same hash.
- **Files:** `embeddings_demo_step4.py`, `Dockerfile`, `scripts/check-pinned-embeddings-pickle.sh`
- **Verification:** `scripts/check-pinned-embeddings-pickle.sh`, `python3 -m py_compile embeddings_demo_step4.py`, `git diff --check`.

## Risks

- If the remote pickle is intentionally regenerated, the checksum must be updated in the same change. That review friction is intentional for a code-executing format.
