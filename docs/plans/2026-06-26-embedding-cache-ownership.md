# Embedding Cache Ownership Boundary

status: completed

## Problem

The clustering cache validated UTF-8 JSON structure but read through filesystem
aliases. A symbolic link or hard link could bind `embedding_cache.json` to an
external inode, and a pre-created `embedding_cache.json.tmp` symlink could make
the save path overwrite an external file before replacement.

## Requirements

1. Reject symbolic-link and multiply linked cache files before parsing JSON.
2. Open cache reads with no-follow semantics and compare path and descriptor
   identity before trusting bytes.
3. Convert symlink-swap races into the same stable local ownership error.
4. Create temporary writes exclusively with no-follow semantics and durable
   flushing before atomic replacement.
5. Preserve missing-cache behavior, JSON validation, and offline lesson flow.
6. Add focused regressions, static contracts, hostile mutations, and complete
   repository verification without OpenAI credentials or paid API calls.

## Work Completed

- Added descriptor-bound regular-file and single-link validation for reads.
- Added exclusive no-follow temporary creation, file flushing, and cleanup for
  writes.
- Added focused tests for symlink and hard-link reads, a symlinked temporary
  path, and a symlink swap during descriptor open.
- Updated maintainer, security, product-direction, and changelog guidance.

## Verification Completed

- All nine focused tests in `test_embedding_cache.py` passed after the three new
  regressions first failed against prior behavior.
- The static baseline rejected six isolated hostile mutations covering path
  following, read no-follow removal, single-link removal, weakened temporary
  creation, removed regression coverage, and incomplete plan status.
- `make check` passed with the full no-network Python suite and Make authority
  cases from the repository root and an external working directory.
- Python syntax compilation, `git diff --check`, strict Git object validation,
  generated-artifact checks, and secret/conflict scans passed before review.
