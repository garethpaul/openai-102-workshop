# Crawler SSRF Boundary

Status: completed

## Problem

The text-search workshop accepts crawler URLs directly from a Streamlit text
area. The shared crawler passed that value to `requests`, and the rendered
tutorial repeated the same unguarded pattern. A user-controlled URL or redirect
could therefore target loopback, link-local metadata, or another private
network service. CodeQL reported this path as `py/full-ssrf` alert 1.

## Requirements

- Accept only HTTP and HTTPS URLs without embedded credentials.
- Resolve the destination before connecting and reject the entire answer set
  when any address is not globally routable.
- Pin each request to one of the validated addresses while preserving the
  original hostname for the HTTP Host header and HTTPS certificate checks.
- Disable environment-derived proxies and automatic redirect following.
- Bound redirects and apply the complete URL and address policy to every hop.
- Keep the rendered tutorial on the same guarded implementation.
- Add no-network regressions for invalid schemes, credentials, private DNS
  answers, connection pinning, private redirect pivots, and redirect limits.
- Preserve the existing timeout and HTTP status handling.

## Implementation

- `utils/crawler.py` owns URL parsing, DNS policy, address pinning, redirect
  handling, and HTML-to-text conversion.
- `pages/3_🔍_Text_Search.py` delegates its rendered example to the shared
  crawler instead of teaching a second request implementation.
- `test_app.py` exercises the boundary with deterministic DNS and HTTP fakes.
- `scripts/check-workshop-baseline.py` keeps the controls and regression names
  in the repository's static baseline contract.

## Verification Completed

- 12 focused crawler cases passed without network access.
- The complete no-network suite passed with 122 tests.
- A certificate-verified request to `https://example.com/` passed through the
  pinned HTTPS transport and returned the expected public page text.
- `make build`, `make verify`, and `make lock-check` passed.
- Both exact-lock `make audit` checks reported no known vulnerabilities.
- `make runtime-check` passed for all reviewed direct dependencies.
- The credential-free `make smoke` Streamlit health check passed.
- `git diff --check` passed and generated locks remained byte-for-byte clean.
