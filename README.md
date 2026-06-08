# openai-102-workshop

<!-- README-OVERVIEW-IMAGE -->
![Project overview](docs/readme-overview.svg)

## Overview

`garethpaul/openai-102-workshop` is a Python web API or service project. OpenAI 102 Workshop

This README is based on the checked-in source, manifests, scripts, and repository metadata on the `main` branch. The project language mix found during review was: no dominant source language detected.

## Repository Contents

- `README.md` - project overview and local usage notes
- `requirements.txt` - Python dependency or packaging metadata
- `cache` - source or example code
- `components` - source or example code
- `Dockerfile` - container build instructions
- `Makefile` - local build or utility targets
- `pages` - source or example code
- `Pipfile` - Python dependency or packaging metadata
- `query_cache` - source or example code
- `SECURITY.md` - security reporting and disclosure guidance
- `url_cache` - source or example code
- `utils` - source or example code

Additional scan context:

- Source directories: cache, components, pages, query_cache, url_cache, utils
- Dependency and build manifests: Dockerfile, Makefile, Pipfile, requirements.txt
- Entry points or build surfaces: Dockerfile, Makefile
- Test-looking files: no obvious test files detected

## Getting Started

### Prerequisites

- Git
- Python matching the era of the project

### Setup

```bash
git clone https://github.com/garethpaul/openai-102-workshop.git
cd openai-102-workshop
python -m pip install -r requirements.txt
```

The setup commands above are derived from repository files. Legacy mobile, Python, or JavaScript samples may require older SDKs or package versions than a modern workstation uses by default.

## Running or Using the Project

- Run `make` or inspect `Makefile` for available targets.

## Testing and Verification

- `make test` if the Makefile defines that target

When the required SDK or runtime is unavailable, use static checks and source review first, then verify on a machine that has the matching platform toolchain.

## Configuration and Secrets

- Detected references to OpenAI. Keep API keys, OAuth credentials, tokens, and account-specific values in local configuration only.

## Security and Privacy Notes

- Review changes touching external API calls or credential-adjacent configuration; examples from the scan include Pipfile.
- Review changes touching network requests, sockets, or service endpoints; examples from the scan include Dockerfile, Pipfile.
- Review changes touching file, media, JSON, XML, CSV, OCR, or data parsing; examples from the scan include Dockerfile.

## Maintenance Notes

- See `SECURITY.md` for vulnerability reporting and safe research guidance.
- See `VISION.md` for project direction and contribution guardrails.

## Contributing

Keep changes small and tied to the project that is already present in this repository. For code changes, document the toolchain used, avoid committing generated dependency directories or local configuration, and update this README when setup or verification steps change.

## Existing Project Notes

Prior README summary:

> Level 2 - API 101? What’s an embedding? How do you use APIs? Code Sample <https://github.com/garethpaul/gpt-docs-api> Local Install To get started you can run the following commands:
