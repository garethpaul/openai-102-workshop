# Verified Embeddings Artifact

status: completed

## Context

The standalone embeddings demo and Docker build fetch a 625,199,795-byte
`embeddings.pkl` fixture from Google Cloud Storage. Both paths currently trust
the mutable URL, and the demo deserializes the file with `pickle.load`.
Deserializing a changed pickle can execute arbitrary code. The existing urgent
PR pins the artifact in two consumers, but it predates current `main`, buffers
the full response in memory, and does not provide reusable no-network tests.

The reviewed artifact identity is:

- SHA-256: `0331e16d863953ab90d26fa3a2a16fe963990553216fd465d5a0d08f4e002c58`
- Size: `625199795` bytes

## Objectives

- Stream remote artifact downloads to a temporary file with bounded request
  timeouts and HTTP status validation.
- Verify the exact byte count and SHA-256 before atomically replacing the local
  fixture.
- Verify an existing fixture before the demo calls `pickle.load`.
- Make the Docker build enforce the same reviewed checksum.
- Exclude local pickle, cache, secret, VCS, and Python-generated files from the
  Docker context so `COPY . .` cannot overwrite the verified artifact.
- Add no-network tests using small fake response chunks for success, checksum
  mismatch, size mismatch, cleanup, and existing-file verification.
- Keep local generated pickle fixtures ignored and avoid downloading or
  deserializing the remote artifact in CI.

## Scope Boundaries

- Do not refresh or commit `embeddings.pkl`.
- Do not change the legacy OpenAI SDK examples or model behavior.
- Do not close or modify existing pull requests.
- Do not add a runtime dependency for integrity verification.

## Verification

- `PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q test_app.py` passed with 59
  tests on 2026-06-12.
- `make lint` passed on 2026-06-12.
- `make test` passed with 59 tests on 2026-06-12.
- `make build` passed on 2026-06-12.
- `make check` passed on 2026-06-12.
- The focused tests rejected mutations bypassing checksum verification and
  accepting a truncated response on 2026-06-12.
- `make lint` rejected removal of pre-deserialization verification from the
  demo on 2026-06-12.
- `make lint` passed with a Docker context guard that excludes local
  `embeddings.pkl` after the verified download step on 2026-06-12.
- `make lint` rejected a mutation that restored local `embeddings.pkl` to the
  Docker build context on 2026-06-12.
- `git diff --check` passed on 2026-06-12.
