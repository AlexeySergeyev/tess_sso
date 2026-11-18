"""Timeouts and retries for network calls: a stalled or dropped connection is retried, not waited on."""

from __future__ import annotations

import socket
import time
import warnings
from contextlib import contextmanager
from typing import Callable, Iterator, Sequence, TypeVar

T = TypeVar("T")

REQUEST_TIMEOUT_S = 60.0  # a connection that sends nothing for this long is dropped
ATTEMPTS = 4  # tries of one step (one block of rows, opening a cube)
RETRY_WAITS_S = (10.0, 30.0, 90.0)  # pause before the 2nd, 3rd, 4th try
_sleep = time.sleep  # looked up at each retry, so tests can skip the pauses


class NetworkRetryWarning(UserWarning):
    """A network step failed and is tried again."""


def is_transient(exc: BaseException) -> bool:
    """A failure worth another try: a timeout, a reset or refused connection, a failed DNS lookup,
    a truncated response, a 5xx status.

    Errors the server meant (a missing object, a 4xx status) are not. Wrapped
    errors (pyvo's, astroquery's) are judged by the network error behind them.
    """
    if _certificate_error(exc):
        return False
    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        verdict = _transient(exc)
        if verdict is not None:
            return verdict
        exc = exc.__cause__ or exc.__context__
    return False


def _certificate_error(exc: BaseException) -> bool:
    """Whether a failed certificate verification is behind ``exc``: another try fails the same way.

    Looks through causes, contexts, ``reason`` (urllib, urllib3) and arguments
    (requests' SSLError, aiohttp's ClientConnectorCertificateError).
    """
    import ssl

    stack, seen = [exc], set()
    while stack:
        current = stack.pop()
        if not isinstance(current, BaseException) or id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, ssl.SSLCertVerificationError):
            return True
        stack += [current.__cause__, current.__context__, getattr(current, "reason", None), *current.args]
    return False


def _transient(exc: BaseException) -> bool | None:
    """True or False when ``exc`` itself says; None to look at what caused it."""
    from urllib.error import HTTPError, URLError

    if isinstance(exc, HTTPError):
        return exc.code >= 500
    if isinstance(exc, (TimeoutError, ConnectionError, EOFError, URLError, socket.gaierror)):
        return True
    for cls in type(exc).__mro__:
        package, name = cls.__module__.split(".")[0], cls.__name__
        if package == "requests" and name == "HTTPError":
            response = getattr(exc, "response", None)
            return response is not None and response.status_code >= 500
        if package == "aiohttp" and name == "ClientResponseError":
            return getattr(exc, "status", 500) >= 500
        if package == "requests" and name in ("ConnectionError", "Timeout", "ChunkedEncodingError"):
            return True
        if package in ("aiohttp", "urllib3", "http"):
            return True
        if package == "botocore" and name in ("HTTPClientError", "ConnectionError"):
            return True
    return None


def with_retries(
    call: Callable[[], T],
    what: str,
    attempts: int = ATTEMPTS,
    waits: Sequence[float] = RETRY_WAITS_S,
    sleep: Callable[[float], None] | None = None,
) -> T:
    """``call()``, tried up to ``attempts`` times while it fails transiently; each retry warns."""
    if attempts < 1:
        raise ValueError(f"attempts must be at least 1, got {attempts}")
    attempt = 1
    while True:
        try:
            return call()
        except Exception as exc:
            if attempt == attempts or not is_transient(exc):
                raise
            wait = waits[min(attempt - 1, len(waits) - 1)] if waits else 0.0
            warnings.warn(
                f"{what}: {type(exc).__name__}: {exc}; try {attempt + 1} of {attempts} in {wait:.0f} s",
                NetworkRetryWarning,
                stacklevel=2,
            )
            (sleep or _sleep)(wait)
            attempt += 1


def use_system_certificates() -> bool:
    """Verify HTTPS against the operating system's trust store, for the whole process; idempotent.

    Python built against OpenSSL (uv's, conda's) reads ``/etc/ssl/cert.pem`` on
    macOS, which lacks roots the keychain has: MAST's ``archive.stsci.edu``
    (emSign Root CA - G1) and NAIF (Sectigo R46) then fail with
    ``CERTIFICATE_VERIFY_FAILED``, on importing tesscube as well. Done when
    tess_sso is imported unless ``TESS_SSO_SYSTEM_CERTS=0``. False when
    truststore is not installed, and nothing changes.
    """
    try:
        import truststore
    except ImportError:
        return False
    import ssl

    if ssl.SSLContext is not truststore.SSLContext:
        truststore.inject_into_ssl()
    return True


def prefer_ipv4() -> None:
    """Try IPv4 addresses before IPv6 ones, for the whole process; idempotent.

    For networks whose IPv6 (e.g. NAT64) resolves but does not connect: every
    request otherwise waits for its IPv6 attempt to time out before using IPv4.
    Also set by ``TESS_SSO_PREFER_IPV4=1`` when tess_sso is imported.
    """
    if getattr(socket.getaddrinfo, "_tess_sso_ipv4_first", False):
        return
    getaddrinfo = socket.getaddrinfo

    def ipv4_first(*args, **kwargs):
        return sorted(getaddrinfo(*args, **kwargs), key=lambda info: info[0] != socket.AF_INET)

    ipv4_first._tess_sso_ipv4_first = True
    socket.getaddrinfo = ipv4_first


@contextmanager
def socket_timeout(seconds: float = REQUEST_TIMEOUT_S) -> Iterator[None]:
    """Give sockets opened without a timeout this one, e.g. urllib or pandas reading a URL.

    Without it a connection that is open but silent (seen with IPv6 to STScI)
    blocks for ever. Process-wide while it lasts; asyncio sockets are unaffected.
    """
    previous = socket.getdefaulttimeout()
    socket.setdefaulttimeout(seconds)
    try:
        yield
    finally:
        socket.setdefaulttimeout(previous)
