import gzip
import itertools
from pathlib import Path
import socket
import ssl
import subprocess
import threading
import time
import zlib

import pytest

from utils import crawler

def public_dns_answer(address="93.184.216.34"):
    return [(2, 1, 6, "", (address, 443))]


class FakeCrawlerResponse:
    def __init__(
        self,
        body=b"<p>public page</p>",
        status_code=200,
        location=None,
        headers=None,
        raw=None,
    ):
        self.status_code = status_code
        self.headers = {"Content-Type": "text/html; charset=utf-8"}
        self.headers.update(headers or {})
        if location is not None:
            self.headers["Location"] = location
        self.raw = raw or FakeCrawlerRaw(body)
        self.closed = False

    @property
    def is_redirect(self):
        return self.status_code in {301, 302, 303, 307, 308} and bool(
            self.headers.get("Location")
        )

    def close(self):
        self.closed = True

    def getheader(self, name, default=None):
        for key, value in self.headers.items():
            if key.lower() == name.lower():
                return value
        return default

    def read(self, amount):
        return self.raw.read(amount)

    def raise_for_status(self):
        if self.status_code >= 400:
            raise crawler.requests.HTTPError(f"HTTP {self.status_code}")


class FakeCrawlerRaw:
    def __init__(self, body, *, clock=None, seconds_per_read=0):
        self._body = body
        self._offset = 0
        self._clock = clock
        self._seconds_per_read = seconds_per_read
        self.read_calls = 0

    def read(self, amount, decode_content=False):
        assert decode_content is False
        self.read_calls += 1
        if self._clock is not None:
            self._clock.advance(self._seconds_per_read)
        chunk = self._body[self._offset:self._offset + amount]
        self._offset += len(chunk)
        return chunk


class FakeCrawlerConnection:
    def __init__(self, response):
        self.response = response
        self.requests = []
        self.closed = False
        self.sock = None

    def request(self, method, target, *, headers):
        self.requests.append((method, target, headers))

    def getresponse(self):
        return self.response

    def close(self):
        self.closed = True


class DelayedCrawlerConnection(FakeCrawlerConnection):
    def __init__(self, response, release):
        super().__init__(response)
        self.release = release

    def getresponse(self):
        self.release.wait(timeout=1)
        return self.response


class FakeCrawlerSocket:
    def __init__(self):
        self.timeouts = []

    def settimeout(self, timeout):
        self.timeouts.append(timeout)


class FakeConnectedSocket:
    def __init__(self, peer):
        self.peer = peer
        self.closed = False
        self.destination = None
        self.timeout = None

    def settimeout(self, timeout):
        self.timeout = timeout

    def connect(self, destination):
        self.destination = destination

    def getpeername(self):
        return self.peer

    def close(self):
        self.closed = True


class FakeClock:
    def __init__(self):
        self.now = 100.0

    def monotonic(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def start_trickling_http_server(chunks, delay):
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]
    finished = threading.Event()

    def serve():
        try:
            client, _ = listener.accept()
            with client:
                client.recv(8192)
                for chunk in chunks:
                    try:
                        client.sendall(chunk)
                    except (BrokenPipeError, ConnectionResetError):
                        break
                    time.sleep(delay)
        finally:
            listener.close()
            finished.set()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    return port, thread, finished


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "ftp://example.com/resource",
        "https://user:password@example.com/",
        "https:///missing-host",
    ],
)
def test_crawler_rejects_non_web_or_credentialed_urls(url):
    with pytest.raises(crawler.UnsafeUrlError):
        crawler._validated_target(url)


@pytest.mark.parametrize(
    "address",
    ["127.0.0.1", "10.0.0.1", "169.254.169.254", "::1", "fc00::1"],
)
def test_crawler_rejects_non_public_dns_answers(monkeypatch, address):
    monkeypatch.setattr(
        crawler.socket,
        "getaddrinfo",
        lambda *args, **kwargs: public_dns_answer(address),
    )

    with pytest.raises(crawler.UnsafeUrlError, match="non-public"):
        crawler._validated_target("https://example.com/private")


@pytest.mark.parametrize(
    "network",
    [
        "0.0.0.0/8",
        "10.0.0.0/8",
        "100.64.0.0/10",
        "127.0.0.0/8",
        "169.254.0.0/16",
        "172.16.0.0/12",
        "192.0.0.0/29",
        "192.0.0.0/24",
        "192.0.0.8/32",
        "192.0.0.170/31",
        "192.0.2.0/24",
        "192.88.99.0/24",
        "192.168.0.0/16",
        "198.18.0.0/15",
        "198.51.100.0/24",
        "203.0.113.0/24",
        "224.0.0.0/4",
        "240.0.0.0/4",
        "::/128",
        "::1/128",
        "::ffff:0:0/96",
        "64:ff9b:1::/48",
        "100::/64",
        "100:0:0:1::/64",
        "2001::/32",
        "2001::/23",
        "2001:2::/48",
        "2001:10::/28",
        "2001:db8::/32",
        "2002::/16",
        "3fff::/20",
        "5f00::/16",
        "fc00::/7",
        "fe80::/10",
        "ff00::/8",
    ],
)
def test_crawler_rejects_every_iana_non_global_special_range(network):
    parsed = crawler.ipaddress.ip_network(network)
    samples = {parsed.network_address, parsed.broadcast_address}
    if parsed.num_addresses > 2:
        samples.add(parsed.network_address + (parsed.num_addresses // 2))

    assert all(not crawler._is_globally_reachable(address) for address in samples)


@pytest.mark.parametrize(
    "address",
    [
        "8.8.8.8",
        "192.0.0.9",
        "192.0.0.10",
        "192.31.196.1",
        "192.52.193.1",
        "192.175.48.1",
        "64:ff9b::808:808",
        "2001:1::1",
        "2001:1::2",
        "2001:1::3",
        "2001:3::1",
        "2001:4:112::1",
        "2001:20::1",
        "2001:30::1",
        "2620:4f:8000::1",
        "2606:4700:4700::1111",
    ],
)
def test_crawler_retains_iana_globally_reachable_exceptions(address):
    assert crawler._is_globally_reachable(crawler.ipaddress.ip_address(address))


@pytest.mark.parametrize(
    "address",
    [
        "64:ff9b::7f00:1",
        "64:ff9b::a00:1",
        "64:ff9b::a9fe:a9fe",
        "64:ff9b::c000:208",
    ],
)
def test_crawler_rejects_nat64_addresses_embedding_non_global_ipv4(address):
    assert not crawler._is_globally_reachable(crawler.ipaddress.ip_address(address))


def test_crawler_address_policy_does_not_use_runtime_is_global():
    source = Path(crawler.__file__).read_text(encoding="utf-8")
    assert ".is_global" not in source


def test_crawler_rejects_mixed_dns_answer_sets(monkeypatch):
    monkeypatch.setattr(
        crawler.socket,
        "getaddrinfo",
        lambda *args, **kwargs: public_dns_answer() + public_dns_answer("10.0.0.1"),
    )

    with pytest.raises(crawler.UnsafeUrlError, match="non-public"):
        crawler._validated_target("https://example.com/")


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com:0/",
        "https://example.com:65536/",
        "https://example.com/#fragment",
        "https://example.com./path\nignored",
        "https://[fe80::1%25eth0]/",
    ],
)
def test_crawler_rejects_ambiguous_authorities_and_urls(url):
    with pytest.raises(crawler.UnsafeUrlError):
        crawler._validated_target(url)


def test_crawler_canonicalizes_idna_host_and_default_port(monkeypatch):
    calls = []
    monkeypatch.setattr(
        crawler.socket,
        "getaddrinfo",
        lambda hostname, port, **kwargs: calls.append((hostname, port))
        or public_dns_answer(),
    )

    target = crawler._validated_target("HTTPS://BÜCHER.Example:443/a%20b?q=1")

    assert target.pinned_url == "https://93.184.216.34:443/a%20b?q=1"
    assert target.host_header == "xn--bcher-kva.example"
    assert target.hostname == "xn--bcher-kva.example"
    assert calls == [("xn--bcher-kva.example", 443)]


@pytest.mark.parametrize(
    "url",
    [
        "https://bad_host.example/",
        "https://-leading.example/",
        "https://trailing-.example/",
        "https://double..example/",
    ],
)
def test_crawler_rejects_non_dns_host_labels(monkeypatch, url):
    monkeypatch.setattr(
        crawler.socket,
        "getaddrinfo",
        lambda *args, **kwargs: public_dns_answer(),
    )

    with pytest.raises(crawler.UnsafeUrlError, match="hostname"):
        crawler._validated_target(url)


def test_crawler_rejects_connected_peer_that_differs_from_validated_address(
    monkeypatch,
):
    fake_socket = FakeConnectedSocket(("127.0.0.1", 443))
    monkeypatch.setattr(crawler.socket, "socket", lambda *args: fake_socket)
    target = crawler._ValidatedTarget(
        scheme="http",
        hostname="example.com",
        port=443,
        address=crawler.ipaddress.ip_address("93.184.216.34"),
        host_header="example.com:443",
        request_target="/",
    )

    connection = crawler._PinnedHTTPConnection(target, crawler.CONNECT_TIMEOUT)

    with pytest.raises(crawler.UnsafeUrlError, match="peer"):
        connection.connect()
    assert fake_socket.closed


def test_crawler_dns_resolution_obeys_total_deadline(monkeypatch):
    blocker = socket.socketpair()

    def blocked_resolution(*args, **kwargs):
        blocker[0].recv(1)

    monkeypatch.setattr(crawler.socket, "getaddrinfo", blocked_resolution)
    deadline = crawler._Deadline(0.01)
    try:
        with pytest.raises(crawler.CrawlerDeadlineExceeded, match="DNS"):
            crawler._public_addresses("example.com", 443, deadline)
    finally:
        blocker[1].send(b"x")
        blocker[0].close()
        blocker[1].close()


def test_crawler_pins_request_to_validated_public_address(monkeypatch):
    responses = [FakeCrawlerResponse()]
    connections = []
    monkeypatch.setattr(
        crawler.socket,
        "getaddrinfo",
        lambda *args, **kwargs: public_dns_answer(),
    )

    def fake_open(target, timeout):
        connection = FakeCrawlerConnection(responses[0])
        connections.append((target, timeout, connection))
        return connection

    monkeypatch.setattr(crawler, "_open_connection", fake_open)

    assert crawler._fetch_html("https://example.com/lesson?q=1") == (
        "<p>public page</p>"
    )
    target, timeout, connection = connections[0]
    assert target.address == crawler.ipaddress.ip_address("93.184.216.34")
    assert target.hostname == "example.com"
    assert timeout == crawler.CONNECT_TIMEOUT
    assert connection.requests == [
        (
            "GET",
            "/lesson?q=1",
            {
                "Accept": "text/html, application/xhtml+xml",
                "Accept-Encoding": "gzip, deflate",
                "Host": "example.com",
                "User-Agent": crawler.USER_AGENT,
            },
        )
    ]
    assert responses[0].closed
    assert connection.closed


def test_crawler_connection_constructor_receives_only_validated_address(monkeypatch):
    monkeypatch.setattr(
        crawler.socket,
        "getaddrinfo",
        lambda *args, **kwargs: public_dns_answer(),
    )
    target = crawler._validated_target("https://example.com/path")

    connection = crawler._open_connection(target, crawler.CONNECT_TIMEOUT)

    assert connection.host == "93.184.216.34"
    assert connection._target.hostname == "example.com"


def test_crawler_disables_proxy_and_netrc_environment(monkeypatch):
    connections = []
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:8080")
    monkeypatch.setenv("NETRC", "/tmp/hostile-netrc")
    monkeypatch.setattr(
        crawler.socket,
        "getaddrinfo",
        lambda *args, **kwargs: public_dns_answer(),
    )

    def fake_open(target, timeout):
        connection = FakeCrawlerConnection(FakeCrawlerResponse())
        connections.append(connection)
        return connection

    monkeypatch.setattr(crawler, "_open_connection", fake_open)
    crawler._fetch_html("https://example.com/")

    assert len(connections) == 1
    source = Path(crawler.__file__).read_text(encoding="utf-8")
    assert "requests.Session" not in source
    assert "trust_env" not in source


def test_crawler_revalidates_redirect_destination(monkeypatch):
    responses = [
        FakeCrawlerResponse(
            status_code=302,
            location="http://169.254.169.254/latest/meta-data/",
        )
    ]
    monkeypatch.setattr(
        crawler.socket,
        "getaddrinfo",
        lambda hostname, *args, **kwargs: public_dns_answer()
        if hostname == "example.com"
        else public_dns_answer("169.254.169.254"),
    )
    monkeypatch.setattr(
        crawler,
        "_open_connection",
        lambda *args: FakeCrawlerConnection(responses[0]),
    )

    with pytest.raises(crawler.UnsafeUrlError, match="non-public"):
        crawler._fetch_html("https://example.com/redirect")
    assert responses[0].closed
    assert responses[0].raw.read_calls == 0


def test_crawler_cross_origin_redirect_forwards_no_credentials(monkeypatch):
    responses = [
        FakeCrawlerResponse(status_code=302, location="https://other.example/final"),
        FakeCrawlerResponse(),
    ]
    connections = []
    monkeypatch.setattr(
        crawler.socket,
        "getaddrinfo",
        lambda *args, **kwargs: public_dns_answer(),
    )

    def fake_open(target, timeout):
        connection = FakeCrawlerConnection(responses[len(connections)])
        connections.append(connection)
        return connection

    monkeypatch.setattr(crawler, "_open_connection", fake_open)

    assert crawler._fetch_html("https://user.example/start") == "<p>public page</p>"
    assert len(connections) == 2
    for connection in connections:
        headers = connection.requests[0][2]
        assert set(headers) == {"Accept", "Accept-Encoding", "Host", "User-Agent"}
        assert "Authorization" not in headers
        assert "Cookie" not in headers
        assert "Proxy-Authorization" not in headers
    assert connections[0].requests[0][2]["Host"] == "user.example"
    assert connections[1].requests[0][2]["Host"] == "other.example"


@pytest.mark.parametrize(
    ("content_type", "body", "expected"),
    [
        ("text/html; charset=iso-8859-1", b"<p>caf\xe9</p>", "<p>caf\xe9</p>"),
        ("application/xhtml+xml; charset=utf-8", b"<p>safe</p>", "<p>safe</p>"),
    ],
)
def test_crawler_accepts_only_html_media_types_and_declared_charset(
    monkeypatch,
    content_type,
    body,
    expected,
):
    response = FakeCrawlerResponse(body=body, headers={"Content-Type": content_type})
    monkeypatch.setattr(
        crawler.socket,
        "getaddrinfo",
        lambda *args, **kwargs: public_dns_answer(),
    )
    monkeypatch.setattr(
        crawler,
        "_open_connection",
        lambda *args: FakeCrawlerConnection(response),
    )

    assert crawler._fetch_html("https://example.com/") == expected


@pytest.mark.parametrize(
    "content_type",
    [
        None,
        "application/json",
        "image/svg+xml",
        "text/plain",
        "text/html, application/json",
    ],
)
def test_crawler_rejects_non_html_success_responses_before_reading(
    monkeypatch,
    content_type,
):
    headers = {} if content_type is None else {"Content-Type": content_type}
    response = FakeCrawlerResponse(headers=headers)
    if content_type is None:
        response.headers.pop("Content-Type")
    connection = FakeCrawlerConnection(response)
    monkeypatch.setattr(
        crawler.socket,
        "getaddrinfo",
        lambda *args, **kwargs: public_dns_answer(),
    )
    monkeypatch.setattr(crawler, "_open_connection", lambda *args: connection)

    with pytest.raises(crawler.CrawlerLimitError, match="content type"):
        crawler._fetch_html("https://example.com/")
    assert response.raw.read_calls == 0
    assert response.closed
    assert connection.closed


def test_crawler_limits_redirect_chain(monkeypatch):
    monkeypatch.setattr(
        crawler.socket,
        "getaddrinfo",
        lambda *args, **kwargs: public_dns_answer(),
    )
    monkeypatch.setattr(
        crawler,
        "_open_connection",
        lambda *args: FakeCrawlerConnection(
            FakeCrawlerResponse(status_code=302, location="/again")
        ),
    )

    with pytest.raises(crawler.requests.TooManyRedirects):
        crawler._fetch_html("https://example.com/start")


def test_crawler_rejects_declared_oversized_body_before_reading():
    response = FakeCrawlerResponse(
        headers={"Content-Length": str(crawler.MAX_WIRE_BYTES + 1)},
    )

    with pytest.raises(crawler.CrawlerLimitError, match="wire byte"):
        crawler._read_response_body(response, crawler._Deadline(1))
    assert response.raw.read_calls == 0


def test_crawler_bounds_chunked_identity_body():
    response = FakeCrawlerResponse(body=b"x" * (crawler.MAX_DECODED_BYTES + 1))

    with pytest.raises(crawler.CrawlerLimitError, match="decoded byte"):
        crawler._read_response_body(response, crawler._Deadline(1))


def test_crawler_streams_gzip_without_expanding_zip_bomb():
    compressed = gzip.compress(b"z" * (crawler.MAX_DECODED_BYTES + 1))
    response = FakeCrawlerResponse(
        body=compressed,
        headers={"Content-Encoding": "gzip"},
    )

    with pytest.raises(crawler.CrawlerLimitError, match="decoded byte"):
        crawler._read_response_body(response, crawler._Deadline(1))
    assert response.raw.read_calls > 0


def test_crawler_streams_deflate_body():
    body = b"<p>deflated</p>"
    response = FakeCrawlerResponse(
        body=zlib.compress(body),
        headers={"Content-Encoding": "deflate"},
    )

    assert crawler._read_response_body(response, crawler._Deadline(1)) == body


def test_crawler_rejects_unsupported_or_chained_content_encoding():
    for encoding in ("br", "gzip, br"):
        response = FakeCrawlerResponse(headers={"Content-Encoding": encoding})
        with pytest.raises(crawler.CrawlerLimitError, match="content encoding"):
            crawler._read_response_body(response, crawler._Deadline(1))


def test_crawler_total_deadline_stops_slow_chunked_response(monkeypatch):
    clock = FakeClock()
    raw = FakeCrawlerRaw(
        b"small response",
        clock=clock,
        seconds_per_read=crawler.TOTAL_TIMEOUT,
    )
    response = FakeCrawlerResponse(raw=raw)
    monkeypatch.setattr(crawler.time, "monotonic", clock.monotonic)

    with pytest.raises(crawler.CrawlerDeadlineExceeded, match="total deadline"):
        crawler._read_response_body(
            response,
            crawler._Deadline(crawler.TOTAL_TIMEOUT),
        )


@pytest.mark.parametrize(
    "chunks",
    [
        [b"H", b"T", b"T", b"P", b"/1.1 200 OK\r\n", b"Content-Length: 0\r\n\r\n"],
        [b"HTTP/1.1 200 OK\r\n", b"X-Slow: ", b"a", b"b", b"c", b"\r\n", b"Content-Length: 0\r\n\r\n"],
    ],
    ids=["partial-status-line", "partial-headers"],
)
def test_crawler_total_deadline_bounds_status_and_header_parsing(
    monkeypatch,
    chunks,
):
    port, server, finished = start_trickling_http_server(chunks, delay=0.04)
    monkeypatch.setattr(
        crawler,
        "_public_addresses",
        lambda hostname, requested_port, deadline: [
            crawler.ipaddress.ip_address("127.0.0.1")
        ],
    )
    started = time.monotonic()

    with pytest.raises(crawler.CrawlerDeadlineExceeded):
        crawler._fetch_html(
            f"http://example.test:{port}/",
            deadline=crawler._Deadline(0.09),
        )

    elapsed = time.monotonic() - started
    assert elapsed < 0.16
    assert finished.wait(timeout=0.5)
    server.join(timeout=0.1)
    assert not server.is_alive()


def test_crawler_closes_header_response_that_arrives_after_deadline():
    response = FakeCrawlerResponse()
    release = threading.Event()
    connection = DelayedCrawlerConnection(response, release)

    with pytest.raises(crawler.CrawlerDeadlineExceeded, match="header"):
        crawler._get_response_with_deadline(connection, crawler._Deadline(0.01))

    release.set()
    for _ in range(100):
        if response.closed:
            break
        time.sleep(0.005)
    assert response.closed
    assert connection.closed


def test_crawler_sets_each_read_timeout_from_remaining_deadline():
    response = FakeCrawlerResponse(body=b"ok")
    connection = FakeCrawlerConnection(response)
    connection.sock = FakeCrawlerSocket()

    assert crawler._read_response_body(
        response,
        crawler._Deadline(1),
        connection,
    ) == b"ok"
    assert connection.sock.timeouts
    assert all(0 < timeout <= 1 for timeout in connection.sock.timeouts)


def test_crawler_closes_response_and_session_after_body_error(monkeypatch):
    response = FakeCrawlerResponse(body=b"x" * (crawler.MAX_DECODED_BYTES + 1))
    connection = FakeCrawlerConnection(response)
    monkeypatch.setattr(
        crawler.socket,
        "getaddrinfo",
        lambda *args, **kwargs: public_dns_answer(),
    )

    monkeypatch.setattr(crawler, "_open_connection", lambda *args: connection)

    with pytest.raises(crawler.CrawlerLimitError):
        crawler._fetch_html("https://example.com/")
    assert response.closed
    assert connection.closed


def test_crawler_closes_response_and_connection_after_http_error(monkeypatch):
    response = FakeCrawlerResponse(status_code=500)
    connection = FakeCrawlerConnection(response)
    monkeypatch.setattr(
        crawler.socket,
        "getaddrinfo",
        lambda *args, **kwargs: public_dns_answer(),
    )
    monkeypatch.setattr(crawler, "_open_connection", lambda *args: connection)

    with pytest.raises(crawler.requests.HTTPError, match="500"):
        crawler._fetch_html("https://example.com/")
    assert response.closed
    assert response.raw.read_calls == 0
    assert connection.closed


def test_crawler_limits_url_count_and_aggregate_decoded_bytes(monkeypatch):
    monkeypatch.setattr(crawler, "_fetch_html", lambda url, **kwargs: "x" * 10)
    monkeypatch.setattr(crawler, "_html_to_text", lambda content, deadline: content)

    with pytest.raises(crawler.CrawlerLimitError, match="URL count"):
        crawler.get_texts(["https://example.com"] * (crawler.MAX_URLS + 1))

    monkeypatch.setattr(crawler, "MAX_AGGREGATE_DECODED_BYTES", 15)
    with pytest.raises(crawler.CrawlerLimitError, match="aggregate"):
        crawler.get_texts(["https://one.example", "https://two.example"])


def test_crawler_caps_url_iterables_before_materializing_them():
    urls = (f"https://example.com/{index}" for index in itertools.count())

    with pytest.raises(crawler.CrawlerLimitError, match="URL count"):
        crawler.get_texts(urls)


def test_crawler_total_deadline_includes_html_parsing(monkeypatch):
    blocker = socket.socketpair()

    def blocked_parse(html_content):
        blocker[0].recv(1)

    monkeypatch.setattr(crawler, "_parse_html", blocked_parse)
    try:
        with pytest.raises(crawler.CrawlerDeadlineExceeded, match="HTML parsing"):
            crawler._html_to_text("<p>blocked</p>", crawler._Deadline(0.01))
    finally:
        blocker[1].send(b"x")
        blocker[0].close()
        blocker[1].close()


def test_text_search_page_uses_bounded_batch_crawler():
    source = Path("pages/3_🔍_Text_Search.py").read_text(encoding="utf-8")
    assert "crawler.get_texts(url_list)" in source


def test_crawler_pinned_https_preserves_sni_and_certificate_hostname(
    monkeypatch,
    tmp_path,
):
    certificate = tmp_path / "certificate.pem"
    private_key = tmp_path / "private-key.pem"
    subprocess.run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "ec",
            "-pkeyopt",
            "ec_paramgen_curve:P-256",
            "-nodes",
            "-days",
            "1",
            "-subj",
            "/CN=example.test",
            "-addext",
            "subjectAltName=DNS:example.test",
            "-keyout",
            str(private_key),
            "-out",
            str(certificate),
        ],
        check=True,
        capture_output=True,
    )

    server_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    server_context.load_cert_chain(certificate, private_key)
    observed = {"sni": [], "request": b""}
    server_context.set_servername_callback(
        lambda ssl_socket, server_name, context: observed["sni"].append(server_name)
    )
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]

    def serve():
        client, _ = listener.accept()
        with server_context.wrap_socket(client, server_side=True) as tls_client:
            observed["request"] = tls_client.recv(8192)
            tls_client.sendall(
                b"HTTP/1.1 200 OK\r\n"
                b"Content-Type: text/html; charset=utf-8\r\n"
                b"Content-Length: 13\r\n\r\n<p>pinned</p>"
            )
        listener.close()

    server = threading.Thread(target=serve, daemon=True)
    server.start()
    client_context = ssl.create_default_context(cafile=str(certificate))
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:9")
    monkeypatch.setenv("NETRC", str(tmp_path / "hostile-netrc"))
    monkeypatch.setattr(
        crawler,
        "_public_addresses",
        lambda hostname, requested_port, deadline: [
            crawler.ipaddress.ip_address("127.0.0.1")
        ],
    )
    monkeypatch.setattr(crawler, "_create_ssl_context", lambda: client_context)

    assert crawler._fetch_html(f"https://example.test:{port}/secure?q=1") == (
        "<p>pinned</p>"
    )
    server.join(timeout=2)
    assert not server.is_alive()
    assert observed["sni"] == ["example.test"]
    assert b"GET /secure?q=1 HTTP/1.1" in observed["request"]
    assert f"Host: example.test:{port}".encode() in observed["request"]
