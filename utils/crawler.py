import ipaddress
import socket
from urllib.parse import urljoin, urlsplit, urlunsplit

import requests
from requests.adapters import HTTPAdapter


MAX_REDIRECTS = 5
REQUEST_TIMEOUT = 15
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 "
    "Safari/537.36"
)


class UnsafeUrlError(ValueError):
    """Raised when a crawler URL can reach a non-public network target."""


class _PinnedAddressAdapter(HTTPAdapter):
    def __init__(self, hostname, scheme):
        self._hostname = hostname
        self._scheme = scheme
        super().__init__()

    def init_poolmanager(self, connections, maxsize, block=False, **pool_kwargs):
        if self._scheme == "https":
            pool_kwargs["assert_hostname"] = self._hostname
            pool_kwargs["server_hostname"] = self._hostname
        super().init_poolmanager(connections, maxsize, block, **pool_kwargs)


def _public_addresses(hostname, port):
    try:
        answers = socket.getaddrinfo(
            hostname,
            port,
            type=socket.SOCK_STREAM,
        )
    except socket.gaierror as error:
        raise UnsafeUrlError("URL hostname could not be resolved") from error

    addresses = {
        answer[4][0].split("%", 1)[0]
        for answer in answers
    }
    if not addresses:
        raise UnsafeUrlError("URL hostname did not resolve to an address")

    parsed_addresses = [ipaddress.ip_address(address) for address in addresses]
    if any(not address.is_global for address in parsed_addresses):
        raise UnsafeUrlError("URL hostname resolves to a non-public address")

    return sorted(parsed_addresses, key=lambda address: (address.version, int(address)))


def _validated_target(url):
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except (TypeError, ValueError) as error:
        raise UnsafeUrlError("URL is invalid") from error

    if parsed.scheme not in {"http", "https"}:
        raise UnsafeUrlError("URL scheme must be http or https")
    if not parsed.hostname:
        raise UnsafeUrlError("URL must include a hostname")
    if parsed.username is not None or parsed.password is not None:
        raise UnsafeUrlError("URL credentials are not allowed")

    hostname = parsed.hostname.encode("idna").decode("ascii")
    effective_port = port or (443 if parsed.scheme == "https" else 80)
    address = _public_addresses(hostname, effective_port)[0]
    address_text = f"[{address}]" if address.version == 6 else str(address)
    netloc = address_text if port is None else f"{address_text}:{port}"
    pinned_url = urlunsplit(
        (parsed.scheme, netloc, parsed.path or "/", parsed.query, "")
    )

    default_port = 443 if parsed.scheme == "https" else 80
    host_header = hostname if port in {None, default_port} else f"{hostname}:{port}"
    return parsed, pinned_url, host_header, hostname


def _fetch_html(url):
    current_url = url

    for redirect_count in range(MAX_REDIRECTS + 1):
        parsed, pinned_url, host_header, hostname = _validated_target(current_url)
        session = requests.Session()
        session.trust_env = False
        session.mount(
            f"{parsed.scheme}://",
            _PinnedAddressAdapter(hostname, parsed.scheme),
        )

        try:
            # lgtm[py/full-ssrf] The request is pinned to a validated public IP.
            response = session.get(
                pinned_url,
                headers={"User-Agent": USER_AGENT, "Host": host_header},
                timeout=REQUEST_TIMEOUT,
                allow_redirects=False,
            )

            if response.is_redirect:
                location = response.headers["Location"]
                response.close()
                if redirect_count == MAX_REDIRECTS:
                    raise requests.TooManyRedirects(
                        f"crawler exceeded {MAX_REDIRECTS} redirects"
                    )
                current_url = urljoin(current_url, location)
                continue

            response.raise_for_status()
            return response.text
        finally:
            session.close()

    raise requests.TooManyRedirects(f"crawler exceeded {MAX_REDIRECTS} redirects")


def get_text(url):
    from bs4 import BeautifulSoup

    html_content = _fetch_html(url)
    soup = BeautifulSoup(html_content, "html.parser")
    return soup.get_text()
