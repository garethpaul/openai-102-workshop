# Security Policy

## Supported Versions

The supported security scope for `openai-102-workshop` is the current default branch, `main`. Older commits, tags, branches, forks, demos, and generated artifacts are not actively supported unless the repository explicitly marks them as maintained.

Project summary: OpenAI 102 Workshop

## Reporting a Vulnerability

Please report suspected vulnerabilities through GitHub's private vulnerability reporting or by opening a draft GitHub Security Advisory for `garethpaul/openai-102-workshop` when that option is available. If GitHub does not show a private reporting option for this repository, contact the repository owner through GitHub and avoid posting exploit details publicly until the issue can be assessed.

Do not open a public issue that includes exploit code, secrets, personal data, or detailed reproduction steps for an unpatched vulnerability.

## What to Include

Helpful reports include:

- the affected file, endpoint, permission, dependency, or workflow
- a concise impact statement explaining what an attacker could do
- reproduction steps using test data and accounts you control
- the branch, commit SHA, platform version, device, runtime, or dependency versions used
- logs, screenshots, or proof-of-concept snippets that demonstrate impact without exposing private data

## Project Security Posture

- This repository appears to be a Python web API or service project. The active security scope is the code and documentation on the default branch.
- Review found authentication, token, or session-related code paths; changes in those areas should receive security-focused review before merge.
- Review found external API integrations or credential-adjacent configuration; changes in those areas should receive security-focused review before merge.
- Review found network clients, sockets, web APIs, or service endpoints; changes in those areas should receive security-focused review before merge.
- Review found file, document, data, or media parsing flows; changes in those areas should receive security-focused review before merge.
- Review found database, model, query, or persistence-related code; changes in those areas should receive security-focused review before merge.
- Review found infrastructure, deployment, proxy, or cloud configuration; changes in those areas should receive security-focused review before merge.
- Review found secret-like configuration names that require careful review before use; changes in those areas should receive security-focused review before merge.
- Dependency manifests detected: `requirements.in`, `requirements.txt`,
  `requirements-test.in`, `requirements-test.txt`, and `Pipfile`. Direct inputs
  and generated locks must stay synchronized, exactly pinned, and limited to
  packages with a demonstrated runtime or verification purpose.
- Workshop users should provide OpenAI credentials through local UI input or `OPENAI_API_KEY`; credentials must not be committed, printed, or placed in generated caches.
- Generated caches under `cache/`, `url_cache/`, and `query_cache/`, plus
  pickle fixtures, may contain prompts, crawled text, or embeddings. Treat
  cache refreshes as reviewable data changes.
- The writable clustering JSON embedding cache accepts only a UTF-8 object with
  string keys and values. `embedding_cache.json` and its temporary file remain
  ignored; generated cache data must never be loaded with `pickle`.
- Python bytecode should not remain after local verification; rerun the gates
  with bytecode writes disabled before committing.
- Hosted Linux validation uses Python 3.12 and separate exact test and
  application locks. It runs `pip check`, audits both locks, regenerates them,
  executes `make check`, imports every direct runtime package, and launches a
  bounded localhost-only Streamlit health smoke without API credentials.
- Keep that hosted path free of private generated caches and customer data.
- Historical OpenAI API examples are inventoried in
  `docs/openai-api-compatibility.md`; do not remove their learner warnings or
  present them as current integration guidance before a credentialed migration.

## Service and API Notes

For web services, APIs, sockets, or scraping workflows, prioritize reports involving authentication bypass, authorization errors, injection, server-side request forgery, unsafe deserialization, credential leakage, data exposure, or denial-of-service conditions. Use test accounts and minimal proof-of-concept traffic only.

For this workshop, also prioritize reports involving API-token persistence,
unsafe generated cache filenames, untrusted pickle loading, hidden network calls,
or lesson code that sends data to APIs outside the visible exercise.
Retrieval vector math should fail closed on malformed fixture vectors instead of
silently truncating dimensions or dividing by zero.
Small embedding fixtures should cap nearest-neighbor lookup to the available
row count so local tests do not require large private caches.
Empty embedding fixtures should fail with a clear validation error before model
training.
Malformed embedding fixtures should fail before nearest-neighbor training when
rows are missing metadata or embeddings have inconsistent dimensions.
Metadata text validation should reject retrieval fixtures that cannot provide
the `text` field used to build augmented queries.
Finite embedding values should be required before nearest-neighbor training so
NaN, infinite, or non-numeric fixture data fails closed.
Numeric embedding values should be real numeric types rather than stringified
numbers before retrieval fixtures train nearest-neighbor models.
Query embedding validation should reject malformed cache or API vectors before
they reach nearest-neighbor lookup.
Embedding API and per-query JSON cache payloads must contain non-empty,
equally sized real numeric finite vectors. Invalid cache data must fail locally
without triggering a replacement API request, and invalid API data must be
rejected before it is written to disk.
Vector value validation should reject empty, boolean, string, complex,
non-finite, and overflowing values before workshop math helpers calculate a
distance or similarity.
Customer recommendations should use validated customer-relative cosine
similarity, ignore malformed or empty product mappings, and select only from a
product-backed industry with at least one nonempty string product name.

## Dependency and Supply Chain Security

Dependency updates should come from trusted package managers. Regenerate both
locks with `make lock-check`, audit them with `make audit`, and do not restore
unused machine-environment packages. Do not commit credentials, private keys,
tokens, generated secrets, or machine-local configuration. If a vulnerability
depends on a compromised package, typosquatting risk, insecure transitive
dependency, or unsafe build step, include the package name, affected version,
and the path through which it is used.

The app contains legacy OpenAI SDK examples pinned to `openai==0.28.1`. Model,
endpoint, or SDK migrations
should be reviewed as compatibility work and verified with `make lint`,
`make test`, `make build`, and `make check`.

## Safe Research Guidelines

Good-faith research is welcome when it stays within these boundaries:

- use only accounts, devices, data, and infrastructure that you own or have explicit permission to test
- avoid destructive actions, persistence, spam, phishing, social engineering, or denial-of-service testing
- minimize access to personal data and stop testing immediately if private data is exposed
- do not exfiltrate secrets or third-party data; report the minimum evidence needed to verify impact
- keep vulnerability details confidential until the maintainer has assessed the report

## Maintainer Response

The maintainer will review complete reports as availability allows, prioritize issues by exploitability and impact, and coordinate a fix or mitigation when the affected code is still maintained. For sample, archived, or educational repositories, the likely remediation may be documentation, dependency updates, or clearly marking unsupported code rather than a production-style patch release.
