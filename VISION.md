## OpenAI 102 Workshop Vision

This document explains the current state and direction of the project.
Project overview and developer docs: [`README.md`](README.md)

OpenAI 102 Workshop is a Streamlit learning app for API concepts, embeddings,
search, recommendations, clustering, fine-tuning exercises, and retrieval-style
question answering.

The repository is useful as a hands-on workshop: learners can run the app,
inspect small scripts, and compare generic model responses with responses
grounded in prepared embeddings and metadata.

The goal is to keep the workshop runnable, teachable, and honest about API,
model, and dependency assumptions.

The current focus is:

Priority:

- Preserve the Streamlit lesson flow and local Docker path
- Keep generated caches and prepared data clearly separated from source logic
- Keep local generated cache files such as `embedding_cache.json` out of source
  control and use strict non-executable formats for writable caches
- Make API-token handling explicit and local to the learner
- Document model, SDK, and Pinecone assumptions when examples depend on them
- Keep `make lint`, `make test`, `make build`, and `make check` available for
  local verification
- Keep `make check` fast and free of OpenAI API calls
- Keep retrieval vector math validation explicit and covered by tests
- Keep vector value validation consistent across cosine, Euclidean, and
  Manhattan helpers
- Keep small embedding fixtures queryable without private generated caches
- Keep empty embedding fixtures rejected with clear no-network tests
- Keep malformed embedding fixtures rejected before nearest-neighbor training
- Keep metadata text validation in place for retrieval fixtures
- Keep finite embedding values validated before nearest-neighbor training
- Keep numeric embedding values validated as numeric types before training
- Keep query embedding validation ahead of nearest-neighbor lookup
- Keep text-search crawling pinned to globally routable HTTP(S) addresses and
  revalidate every bounded redirect before connecting
- Validate generated per-query embedding cache and API payloads before they
  reach retrieval code or are persisted
- Keep verification targets from leaving Python bytecode behind
- Keep the no-network baseline running in pinned, read-only hosted Linux CI
- Keep Python 3.12 direct dependency inputs and generated exact locks small,
  synchronized, and audit-clean
- Keep unused exported-environment packages out of the application graph
- Keep hosted test dependencies minimal and explicit in
  `requirements-test.in`
- Keep a separate hosted full-lock import and headless Streamlit health smoke
- Keep GitHub Actions aligned with the canonical `make check` baseline
- Keep historical OpenAI SDK and model examples visibly marked until migrated
- Keep the fine-tuning retry example narrow so only rate limits are retried
- Keep the Starlette resolver floor explicit and reproducible from public PyPI
- Keep recommendations constrained to validated, product-backed industries
- Keep each product-backed recommendation limited to a nonempty string product name
- Keep malformed customer-list members from blocking later valid records
- Keep each matched record limited to a nonempty string customer industry
- Keep recommendation container validation ahead of iteration and mapping lookup
- Keep invalid recommendation embedding pairs out of similarity scoring
- Keep the customer-industry recommendation tie break independent of mapping order

Next priorities:

- Add lightweight tests for nearest-neighbor lookup changes
- Add more fixture tests for malformed retrieval data
- Document which files are workshop fixtures versus generated output
- Add compatibility notes before migrating legacy OpenAI SDK examples

Contribution rules:

- One PR = one focused lesson, dependency, data, or documentation change.
- Do not commit real API keys, customer data, or private workshop material.
- Keep code examples small enough for learners to trace.
- Explain model or API migrations in the README.
- Run `make lint`, `make test`, `make build`, and `make check` before pushing
  changes.
- Run `make lock-check` and `make audit` for dependency changes, then exercise
  `make runtime-check` and `make smoke` in the application environment.
- Keep universal lock artifact hashes generated from public PyPI and require
  them for application and verification lock installation.
- Preserve metadata text validation when changing retrieval fixture loading.
- Preserve finite embedding value validation when changing fixture loading.
- Preserve numeric embedding values validation when changing fixture loading.
- Preserve query embedding validation when changing retrieval lookup.
- Preserve the crawler's globally routable address checks, connection pinning,
  proxy isolation, and redirect revalidation when changing URL retrieval.
- Preserve bytecode-free test execution when changing Makefile gates.

## Security And Responsible Use

Canonical security policy and reporting:

- [`SECURITY.md`](SECURITY.md)

Workshop users provide their own API credentials. The app should not persist,
print, or transmit those credentials except to the APIs that the user
explicitly enables while running the lesson.
Generated caches and safe JSON embedding fixtures should remain reproducible
workshop data, not private learner output. Active lessons must reject malformed
JSON fixture data and never download or deserialize generated data with
`pickle`.
Retrieval fixtures should reject stringified numeric embedding values before
nearest-neighbor training.

## What We Will Not Merge (For Now)

- Checked-in secrets or private data
- Hidden network calls outside the lesson being demonstrated
- Model upgrades without compatibility notes
- Large cache refreshes that are not reproducible

This list is a roadmap guardrail, not a permanent rule.
Strong user demand and strong technical rationale can change it.
