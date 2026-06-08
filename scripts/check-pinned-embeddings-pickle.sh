#!/usr/bin/env sh
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
EXPECTED_SHA="0331e16d863953ab90d26fa3a2a16fe963990553216fd465d5a0d08f4e002c58"

if ! grep -Fq "EMBEDDINGS_SHA256 = '$EXPECTED_SHA'" "$ROOT_DIR/embeddings_demo_step4.py"; then
  printf '%s\n' "embeddings_demo_step4.py must pin the embeddings pickle checksum." >&2
  exit 1
fi

if ! grep -Fq "hashlib.sha256" "$ROOT_DIR/embeddings_demo_step4.py"; then
  printf '%s\n' "embeddings_demo_step4.py must verify downloaded pickle bytes before loading." >&2
  exit 1
fi

if ! grep -Fq "response.raise_for_status()" "$ROOT_DIR/embeddings_demo_step4.py"; then
  printf '%s\n' "embeddings_demo_step4.py must fail closed on failed downloads." >&2
  exit 1
fi

if ! grep -Fq "$EXPECTED_SHA  embeddings.pkl" "$ROOT_DIR/Dockerfile"; then
  printf '%s\n' "Dockerfile must verify embeddings.pkl with the pinned checksum." >&2
  exit 1
fi

printf '%s\n' "Pinned embeddings pickle checks passed."
