import http.client
import ipaddress
import itertools
import queue
import socket
import ssl
import threading
import time
import zlib
from dataclasses import dataclass
from urllib.parse import urljoin, urlsplit

import requests


MAX_REDIRECTS = 5
MAX_URLS = 8
MAX_URL_LENGTH = 2048
MAX_WIRE_BYTES = 4 * 1024 * 1024
MAX_DECODED_BYTES = 2 * 1024 * 1024
MAX_AGGREGATE_DECODED_BYTES = 16 * 1024 * 1024
WIRE_CHUNK_BYTES = 16 * 1024
DNS_TIMEOUT = 3.0
CONNECT_TIMEOUT = 5.0
READ_TIMEOUT = 5.0
TOTAL_TIMEOUT = 20.0
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 "
    "Safari/537.36"
)

_IPV4_GLOBAL_EXCEPTIONS = tuple(
    ipaddress.ip_network(network)
    for network in (
        "192.0.0.9/32",
        "192.0.0.10/32",
        "192.31.196.0/24",
        "192.52.193.0/24",
        "192.175.48.0/24",
    )
)
_IPV4_NON_GLOBAL_NETWORKS = tuple(
    ipaddress.ip_network(network)
    for network in (
        "0.0.0.0/8",
        "10.0.0.0/8",
        "100.64.0.0/10",
        "127.0.0.0/8",
        "169.254.0.0/16",
        "172.16.0.0/12",
        "192.0.0.0/24",
        "192.0.2.0/24",
        "192.88.99.0/24",
        "192.168.0.0/16",
        "198.18.0.0/15",
        "198.51.100.0/24",
        "203.0.113.0/24",
        "224.0.0.0/4",
        "240.0.0.0/4",
    )
)
_IPV6_GLOBAL_EXCEPTIONS = tuple(
    ipaddress.ip_network(network)
    for network in (
        "64:ff9b::/96",
        "2001:1::1/128",
        "2001:1::2/128",
        "2001:1::3/128",
        "2001:3::/32",
        "2001:4:112::/48",
        "2001:20::/28",
        "2001:30::/28",
    )
)
_IPV6_NON_GLOBAL_NETWORKS = tuple(
    ipaddress.ip_network(network)
    for network in (
        "::/128",
        "::1/128",
        "::ffff:0:0/96",
        "64:ff9b:1::/48",
        "100::/64",
        "100:0:0:1::/64",
        "2001::/23",
        "2001::/32",
        "2001:2::/48",
        "2001:10::/28",
        "2001:db8::/32",
        "2002::/16",
        "3fff::/20",
        "5f00::/16",
        "fc00::/7",
        "fe80::/10",
        "ff00::/8",
    )
)
_IPV6_GLOBAL_UNICAST = ipaddress.ip_network("2000::/3")
_NAT64_WELL_KNOWN_PREFIX = ipaddress.ip_network("64:ff9b::/96")
_DNS_SLOTS = threading.BoundedSemaphore(4)
_PARSER_SLOTS = threading.BoundedSemaphore(2)


class UnsafeUrlError(ValueError):
    """Raised when a crawler URL can reach a non-public network target."""


class CrawlerLimitError(RuntimeError):
    """Raised when a crawler response exceeds a hard resource limit."""


class CrawlerDeadlineExceeded(TimeoutError):
    """Raised when crawler work exceeds its wall-clock budget."""


class _Deadline:
    def __init__(self, seconds=TOTAL_TIMEOUT):
        self._expires_at = time.monotonic() + seconds

    def remaining(self, operation="crawler total deadline"):
        remaining = self._expires_at - time.monotonic()
        if remaining <= 0:
            raise CrawlerDeadlineExceeded(f"{operation} exceeded")
        return remaining

    def timeout(self, maximum, operation="crawler total deadline"):
        return min(maximum, self.remaining(operation))

    def check(self, operation="crawler total deadline"):
        self.remaining(operation)


@dataclass(frozen=True)
class _ValidatedTarget:
    scheme: str
    hostname: str
    port: int
    address: object
    host_header: str
    request_target: str

    @property
    def pinned_url(self):
        address = f"[{self.address}]" if self.address.version == 6 else str(self.address)
        return f"{self.scheme}://{address}:{self.port}{self.request_target}"


class _PinnedHTTPConnection(http.client.HTTPConnection):
    def __init__(self, target, timeout):
        super().__init__(str(target.address), target.port, timeout=timeout)
        self._target = target

    def connect(self):
        family = socket.AF_INET6 if self._target.address.version == 6 else socket.AF_INET
        sock = socket.socket(family, socket.SOCK_STREAM)
        try:
            sock.settimeout(self.timeout)
            if family == socket.AF_INET6:
                destination = (str(self._target.address), self._target.port, 0, 0)
            else:
                destination = (str(self._target.address), self._target.port)
            sock.connect(destination)
        except BaseException:
            sock.close()
            raise
        self.sock = sock


class _PinnedHTTPSConnection(_PinnedHTTPConnection):
    def __init__(self, target, timeout, context=None):
        super().__init__(target, timeout)
        self._context = context or _create_ssl_context()

    def connect(self):
        super().connect()
        try:
            self.sock = self._context.wrap_socket(
                self.sock,
                server_hostname=self._target.hostname,
            )
        except BaseException:
            self.sock.close()
            self.sock = None
            raise


def _create_ssl_context():
    return ssl.create_default_context()


def _is_globally_reachable(address):
    if address.version == 4:
        if any(address in network for network in _IPV4_GLOBAL_EXCEPTIONS):
            return True
        return not any(address in network for network in _IPV4_NON_GLOBAL_NETWORKS)

    if address in _NAT64_WELL_KNOWN_PREFIX:
        embedded = ipaddress.ip_address(int(address) & 0xFFFFFFFF)
        return _is_globally_reachable(embedded)
    if any(address in network for network in _IPV6_GLOBAL_EXCEPTIONS):
        return True
    if any(address in network for network in _IPV6_NON_GLOBAL_NETWORKS):
        return False
    return address in _IPV6_GLOBAL_UNICAST


def _resolve_with_deadline(hostname, port, deadline):
    timeout = deadline.timeout(DNS_TIMEOUT, "DNS resolution deadline")
    if not _DNS_SLOTS.acquire(timeout=timeout):
        raise CrawlerDeadlineExceeded("DNS resolution deadline exceeded")

    result_queue = queue.Queue(maxsize=1)

    def resolve():
        try:
            result_queue.put(
                (True, socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM))
            )
        except BaseException as error:
            result_queue.put((False, error))
        finally:
            _DNS_SLOTS.release()

    threading.Thread(target=resolve, daemon=True).start()
    try:
        succeeded, result = result_queue.get(timeout=timeout)
    except queue.Empty as error:
        raise CrawlerDeadlineExceeded("DNS resolution deadline exceeded") from error
    deadline.check("DNS resolution deadline")
    if not succeeded:
        raise result
    return result


def _public_addresses(hostname, port, deadline=None):
    deadline = deadline or _Deadline()
    try:
        answers = _resolve_with_deadline(hostname, port, deadline)
    except socket.gaierror as error:
        raise UnsafeUrlError("URL hostname could not be resolved") from error

    raw_addresses = {answer[4][0] for answer in answers}
    if not raw_addresses:
        raise UnsafeUrlError("URL hostname did not resolve to an address")
    if any("%" in address for address in raw_addresses):
        raise UnsafeUrlError("URL hostname resolved to a scoped address")

    try:
        addresses = [ipaddress.ip_address(address) for address in raw_addresses]
    except ValueError as error:
        raise UnsafeUrlError("URL hostname resolved to an invalid address") from error
    if any(not _is_globally_reachable(address) for address in addresses):
        raise UnsafeUrlError("URL hostname resolves to a non-public address")

    return sorted(addresses, key=lambda address: (address.version, int(address)))


def _canonical_hostname(hostname):
    if "%" in hostname:
        raise UnsafeUrlError("URL scoped addresses are not allowed")
    try:
        literal = ipaddress.ip_address(hostname)
    except ValueError:
        try:
            hostname = hostname.rstrip(".").encode("idna").decode("ascii").lower()
        except UnicodeError as error:
            raise UnsafeUrlError("URL hostname is invalid") from error
        if not hostname or len(hostname) > 253:
            raise UnsafeUrlError("URL hostname is invalid")
        return hostname
    return str(literal)


def _validated_target(url, deadline=None):
    deadline = deadline or _Deadline()
    if not isinstance(url, str) or not url or len(url) > MAX_URL_LENGTH:
        raise UnsafeUrlError("URL is invalid")
    if url != url.strip() or any(ord(character) < 32 for character in url):
        raise UnsafeUrlError("URL contains forbidden whitespace or controls")

    try:
        parsed = urlsplit(url)
        port = parsed.port
    except (TypeError, ValueError) as error:
        raise UnsafeUrlError("URL is invalid") from error

    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"}:
        raise UnsafeUrlError("URL scheme must be http or https")
    if not parsed.hostname:
        raise UnsafeUrlError("URL must include a hostname")
    if parsed.username is not None or parsed.password is not None:
        raise UnsafeUrlError("URL credentials are not allowed")
    if parsed.fragment:
        raise UnsafeUrlError("URL fragments are not allowed")
    if port is not None and not 1 <= port <= 65535:
        raise UnsafeUrlError("URL port is invalid")

    hostname = _canonical_hostname(parsed.hostname)
    effective_port = port or (443 if scheme == "https" else 80)
    address = _public_addresses(hostname, effective_port, deadline)[0]
    default_port = 443 if scheme == "https" else 80
    if ":" in hostname:
        host_name = f"[{hostname}]"
    else:
        host_name = hostname
    host_header = host_name if effective_port == default_port else f"{host_name}:{effective_port}"
    path = parsed.path or "/"
    request_target = path if not parsed.query else f"{path}?{parsed.query}"
    return _ValidatedTarget(
        scheme=scheme,
        hostname=hostname,
        port=effective_port,
        address=address,
        host_header=host_header,
        request_target=request_target,
    )


def _open_connection(target, timeout):
    if target.scheme == "https":
        return _PinnedHTTPSConnection(target, timeout)
    return _PinnedHTTPConnection(target, timeout)


def _response_status(response):
    return getattr(response, "status", getattr(response, "status_code", 0))


def _response_header(response, name, default=None):
    if hasattr(response, "getheader"):
        return response.getheader(name, default)
    return response.headers.get(name, default)


def _bounded_append(chunks, chunk, decoded_bytes, maximum):
    decoded_bytes += len(chunk)
    if decoded_bytes > maximum:
        raise CrawlerLimitError("crawler exceeded decoded byte limit")
    chunks.append(chunk)
    return decoded_bytes


def _read_response_body(response, deadline, connection=None, maximum=MAX_DECODED_BYTES):
    content_length = _response_header(response, "Content-Length")
    if content_length is not None:
        try:
            declared_length = int(content_length)
        except ValueError as error:
            raise CrawlerLimitError("crawler received invalid wire byte length") from error
        if declared_length < 0 or declared_length > MAX_WIRE_BYTES:
            raise CrawlerLimitError("crawler exceeded wire byte limit")

    encoding = (_response_header(response, "Content-Encoding", "") or "").strip().lower()
    if encoding not in {"", "identity", "gzip", "deflate"}:
        raise CrawlerLimitError("crawler received unsupported content encoding")
    decompressor = None
    if encoding == "gzip":
        decompressor = zlib.decompressobj(16 + zlib.MAX_WBITS)
    elif encoding == "deflate":
        decompressor = zlib.decompressobj(zlib.MAX_WBITS)

    chunks = []
    wire_bytes = 0
    decoded_bytes = 0
    while True:
        timeout = deadline.timeout(READ_TIMEOUT)
        if connection is not None and getattr(connection, "sock", None) is not None:
            connection.sock.settimeout(timeout)
        chunk = response.read(min(WIRE_CHUNK_BYTES, MAX_WIRE_BYTES - wire_bytes + 1))
        deadline.check()
        if not chunk:
            break
        wire_bytes += len(chunk)
        if wire_bytes > MAX_WIRE_BYTES:
            raise CrawlerLimitError("crawler exceeded wire byte limit")

        if decompressor is None:
            decoded_bytes = _bounded_append(chunks, chunk, decoded_bytes, maximum)
            continue

        remaining = maximum - decoded_bytes
        try:
            decoded = decompressor.decompress(chunk, remaining + 1)
        except zlib.error as error:
            raise CrawlerLimitError("crawler received invalid compressed content") from error
        decoded_bytes = _bounded_append(chunks, decoded, decoded_bytes, maximum)
        if decompressor.unconsumed_tail:
            raise CrawlerLimitError("crawler exceeded decoded byte limit")
        if decompressor.unused_data:
            raise CrawlerLimitError("crawler received trailing compressed content")

    if decompressor is not None:
        remaining = maximum - decoded_bytes
        try:
            decoded = decompressor.flush(remaining + 1)
        except zlib.error as error:
            raise CrawlerLimitError("crawler received invalid compressed content") from error
        _bounded_append(chunks, decoded, decoded_bytes, maximum)
        if not decompressor.eof:
            raise CrawlerLimitError("crawler received truncated compressed content")

    return b"".join(chunks)


def _fetch_html(url, deadline=None, maximum=MAX_DECODED_BYTES):
    deadline = deadline or _Deadline()
    current_url = url

    for redirect_count in range(MAX_REDIRECTS + 1):
        target = _validated_target(current_url, deadline)
        connection = _open_connection(
            target,
            deadline.timeout(CONNECT_TIMEOUT, "connect deadline"),
        )
        response = None
        try:
            connection.request(
                "GET",
                target.request_target,
                headers={
                    "Accept-Encoding": "gzip, deflate",
                    "Host": target.host_header,
                    "User-Agent": USER_AGENT,
                },
            )
            response = connection.getresponse()
            deadline.check()
            status = _response_status(response)

            if status in {301, 302, 303, 307, 308}:
                location = _response_header(response, "Location")
                if not location:
                    raise requests.HTTPError("crawler redirect omitted Location")
                if redirect_count == MAX_REDIRECTS:
                    raise requests.TooManyRedirects(
                        f"crawler exceeded {MAX_REDIRECTS} redirects"
                    )
                current_url = urljoin(current_url, location)
                continue

            if status < 200 or status >= 300:
                raise requests.HTTPError(f"crawler HTTP status {status}")
            body = _read_response_body(response, deadline, connection, maximum)
            encoding = getattr(response, "encoding", None) or "utf-8"
            return body.decode(encoding, errors="replace")
        finally:
            if response is not None:
                response.close()
            connection.close()

    raise requests.TooManyRedirects(f"crawler exceeded {MAX_REDIRECTS} redirects")


def _parse_html(html_content):
    from bs4 import BeautifulSoup

    return BeautifulSoup(html_content, "html.parser").get_text()


def _html_to_text(html_content, deadline=None):
    deadline = deadline or _Deadline()
    timeout = deadline.remaining("HTML parsing deadline")
    if not _PARSER_SLOTS.acquire(timeout=timeout):
        raise CrawlerDeadlineExceeded("HTML parsing deadline exceeded")

    result_queue = queue.Queue(maxsize=1)

    def parse():
        try:
            result_queue.put((True, _parse_html(html_content)))
        except BaseException as error:
            result_queue.put((False, error))
        finally:
            _PARSER_SLOTS.release()

    threading.Thread(target=parse, daemon=True).start()
    try:
        succeeded, result = result_queue.get(timeout=timeout)
    except queue.Empty as error:
        raise CrawlerDeadlineExceeded("HTML parsing deadline exceeded") from error
    deadline.check("HTML parsing deadline")
    if not succeeded:
        raise result
    return result


def get_text(url):
    deadline = _Deadline()
    return _html_to_text(_fetch_html(url, deadline=deadline), deadline)


def get_texts(urls):
    urls = list(itertools.islice(iter(urls), MAX_URLS + 1))
    if not urls or len(urls) > MAX_URLS:
        raise CrawlerLimitError(f"crawler exceeded URL count limit of {MAX_URLS}")

    deadline = _Deadline()
    results = []
    aggregate_bytes = 0
    for url in urls:
        remaining = MAX_AGGREGATE_DECODED_BYTES - aggregate_bytes
        if remaining <= 0:
            raise CrawlerLimitError("crawler exceeded aggregate decoded byte limit")
        maximum = min(MAX_DECODED_BYTES, remaining)
        html_content = _fetch_html(url, deadline=deadline, maximum=maximum)
        aggregate_bytes += len(html_content.encode("utf-8"))
        if aggregate_bytes > MAX_AGGREGATE_DECODED_BYTES:
            raise CrawlerLimitError("crawler exceeded aggregate decoded byte limit")
        results.append(_html_to_text(html_content, deadline))
    return results
