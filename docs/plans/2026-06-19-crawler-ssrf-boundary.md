# Crawler SSRF And Resource Boundary

Status: completed

## Problem

The text-search workshop accepts crawler URLs from a Streamlit text area. The
previous implementation passed a reconstructed URL to `requests`, depended on
the running Python patch release's `ipaddress.is_global` behavior, and eagerly
buffered decoded response bodies without a total wall-clock deadline. Python
3.12.0 therefore accepted documented IANA non-global ranges, and hostile slow,
chunked, or compressed responses could exhaust workers or memory.

CodeQL reported the original path as `py/full-ssrf`. Alert #8 on PR #26 was
manually dismissed as a false positive on 2026-06-19T06:22:12Z against exact
SHA `3ade17a6557db61e21a4716a46beae604443613a`. The dismissal claimed every hop
was globally routable and safely pinned, but the supported Python 3.12.0
classification contradicted that premise. The dismissal is historical state,
not verification evidence for this replacement.

## Requirements

- Accept only canonical HTTP and HTTPS URLs without credentials, fragments,
  invalid ports, scoped addresses, controls, or ambiguous authorities.
- Classify every DNS answer and redirect hop with an explicit IANA-derived
  policy that is independent of `ipaddress.is_global` runtime changes.
- Reject NAT64 well-known-prefix addresses when their embedded IPv4 address is
  not globally reachable under the same policy.
- Connect a numeric socket only to the validated address while retaining the
  original hostname solely for the HTTP `Host` header, TLS SNI, and certificate
  verification.
- Avoid proxy, `.netrc`, and automatic redirect behavior structurally.
- Bound DNS, connect, read, parsing, and total wall-clock time; URL and redirect
  count; wire bytes; decoded bytes; decompression; and aggregate crawl work.
- Inspect redirect headers before reading a redirect body and close every
  response and connection on success or failure.
- Keep the rendered tutorial on the bounded multi-URL implementation.

## Implementation

- `utils/crawler.py` owns explicit IPv4/IPv6 policy tables, bounded DNS and HTML
  parsing workers, direct numeric HTTP(S) connections, hostname-preserving TLS,
  redirect revalidation, incremental gzip/deflate decoding, and hard budgets.
- `pages/3_🔍_Text_Search.py` uses `crawler.get_texts()` for the live page and
  rendered tutorial so URL count and aggregate work share one deadline.
- `test_crawler.py` isolates the crawler suite from scientific application
  dependencies, allowing exact Python 3.10, 3.12.0, and 3.14 verification.
- `scripts/check-workshop-baseline.py` rejects restoration of the old
  `requests.Session`/`is_global` sink and authenticates the new controls and
  regression names.

## Verification Completed

- 92 focused crawler cases passed on Python 3.10.16, exact Python 3.12.0, and
  Python 3.14.5.
- The exact Python 3.12.0 probe reproduced the stdlib mismatch: it marked
  `192.0.0.8` and `64:ff9b:1::1` global and `2001:1::1` private, while the
  explicit crawler policy returned the IANA registry outcomes.
- A real local certificate-verified pinned HTTPS request passed with direct-IP
  connection, original `Host`, original SNI, certificate hostname validation,
  and hostile proxy/`.netrc` environment variables present.
- The complete no-network suite passed with 202 tests.
- `make build`, `make verify`, and `make lock-check` passed.
- Both exact-lock `make audit` checks reported no known vulnerabilities.
- `make runtime-check` passed for all reviewed direct dependencies.
- The credential-free `make smoke` Streamlit health check passed.
- `git diff --check` passed and generated locks remained byte-for-byte clean.
- Historical CodeQL Alert #8 dismissal state was documented and was not used
  as evidence for the replacement transport.
- Fresh exact-head hosted baseline, application smoke, and CodeQL results are
  recorded in the external publication report after the guarded branch update.
