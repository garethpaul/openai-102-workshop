# Supported Python Dependency Graph

status: completed

## Context

`requirements.txt` is an exported workstation environment rather than a
reviewed application manifest. GitHub reports 70 open alerts against it,
including critical advisories in PyArrow, PyTorch, and Transformers. PyTorch,
Transformers, SentencePiece, and most of the listed packages are not imported
by the workshop. Hosted validation intentionally installs only
`requirements-test.txt`, so neither the full application graph nor a real
Streamlit startup is currently proven.

The maintained source directly needs Streamlit, the legacy OpenAI SDK lesson
surface, NumPy/Pandas/scikit-learn plotting helpers, requests and
BeautifulSoup, spaCy, tiktoken, and one recursive text splitter. The LangChain
page itself renders examples as code strings; only `utils/token.py` imports a
LangChain splitter at runtime.

## Priority

Remove the critical and high-risk dependency exposure by making the installed
graph small, explicit, reproducible, and executable before attempting a
separate OpenAI lesson/API migration.

## Requirements

- R1. Replace the exported environment with a reviewed direct-dependency input
  and a generated, fully pinned application lock for Python 3.12.
- R2. Remove packages that are not imported by the application, including
  PyTorch, Transformers, SentencePiece, and the full LangChain distribution.
- R3. Replace the legacy `langchain.text_splitter` runtime import with the
  supported `langchain-text-splitters` package while preserving chunking
  behavior and no-network tests.
- R4. Upgrade every retained package beyond its recorded patched advisory
  floor and make `pip-audit` report zero known vulnerabilities for both the
  application lock and focused test requirements.
- R5. Keep legacy OpenAI SDK calls on `openai==0.28.1` in this pass; model and
  client API migration is a separate compatibility project.
- R6. Upgrade the supported local, hosted, Pipfile, and container runtime
  contract from Python 3.10 to Python 3.12.
- R7. Add a dependency-free manifest contract that rejects removed packages,
  floating requirements, lock drift, unsupported Python versions, and missing
  audit/runtime-smoke gates.
- R8. Add a no-credential runtime import check for every direct dependency and
  a bounded headless Streamlit health smoke that does not call OpenAI.
- R9. Hosted push and pull-request validation must run the existing no-network
  test baseline and a separate full-lock install, `pip check`, `pip-audit`,
  runtime import, and Streamlit smoke job with pinned actions and read-only
  permissions.
- R10. Update README, security, vision, changes, contributor guidance, and this
  plan with the supported graph, exact verification, and any platform limits.

## Implementation Units

1. Add a minimal direct dependency input and compile the universal application
   lock with `uv pip compile`; modernize `requirements-test.txt`, Pipfile, and
   Python runtime declarations without adding generated environment state.
2. Move the recursive splitter import to `langchain_text_splitters` and add a
   deterministic chunk-boundary regression test.
3. Add runtime-import, dependency-policy, audit, and bounded Streamlit health
   checks; run the full graph in a clean Python 3.12 environment.
4. Expand GitHub Actions with separate focused-test and application-smoke jobs
   for both canonical events, then update maintenance documentation and plan
   evidence.

## Scope Boundaries

- Do not migrate OpenAI SDK calls, lesson models, prompts, or API response
  parsing in this pass.
- Do not make paid API calls, require credentials, download spaCy models, or
  refresh generated caches during tests.
- Do not restore unused machine-environment packages merely to match the old
  exported manifest.
- Do not alter or merge the repository's other open pull requests.

## Verification

- Clean Python 3.12 installs of the focused test requirements and application
  lock
- `python -m pip check`
- `pip-audit` against both requirement sets with zero findings
- `make lint`, `make test`, `make build`, and `make check`
- runtime direct-import check
- bounded headless Streamlit health smoke without credentials
- lock regeneration produces no diff
- hostile manifest, runtime, workflow, and plan mutations are rejected
- `git diff --check`
- successful exact-head push, pull-request, and CodeQL runs

## Work Completed

- Replaced the exported workstation manifest with 12 reviewed direct
  application pins and generated exact Python 3.12 application and test locks.
- Removed unused PyTorch, Transformers, SentencePiece, virtualenv,
  python-dotenv, and the full LangChain package from the installed graph.
- Moved recursive chunking to `langchain-text-splitters` and added exact
  token-limit and overlap regression coverage.
- Added stable lock regeneration plus an explicit upgrade command, zero-finding
  audits for both locks, direct runtime import verification, and a bounded
  localhost-only Streamlit health smoke.
- Upgraded the Pipfile, Docker image, contributor guidance, and both canonical
  hosted event paths to Python 3.12 while retaining `openai==0.28.1` for the
  legacy lesson API boundary.

## Verification Results

- Clean Python 3.12.8 environments installed 73 focused verification packages
  and 108 application packages; both compatibility checks passed.
- `pip-audit -r requirements-test.txt` and `pip-audit -r requirements.txt`
  each reported no known vulnerabilities.
- `make lint`, `make test`, `make build`, and `make check` passed with 55
  no-network tests; clean-head `make lock-check` regenerated both locks without
  a diff.
- `make runtime-check` imported all 12 reviewed direct packages at their exact
  versions, and `make smoke` received a healthy response from a real headless
  Streamlit process without an API credential.
- Canonical push run `27430175764` and pull-request run `27430176870` passed
  both the focused baseline and full application smoke at implementation head
  `9459cbe007b2fe9bac9a6dd95e10745a46497d98`.
- CodeQL run `27430174890` passed Actions and Python analysis at the same head.
