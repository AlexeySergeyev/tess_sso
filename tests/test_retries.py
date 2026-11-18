import socket
import ssl
import warnings
from urllib.error import HTTPError, URLError

import pytest

from tess_sso.network import (
    NetworkRetryWarning,
    is_transient,
    prefer_ipv4,
    socket_timeout,
    use_system_certificates,
    with_retries,
)


@pytest.mark.parametrize("exc, transient", [
    (TimeoutError("timed out"), True),
    (ConnectionResetError(54, "reset"), True),
    (URLError("no route"), True),
    (HTTPError("u", 503, "unavailable", None, None), True),
    (HTTPError("u", 404, "not found", None, None), False),
    (ValueError("bad"), False),
    (KeyError("x"), False),
])
def test_is_transient(exc, transient):
    assert is_transient(exc) is transient


def test_retries_until_it_works_with_the_given_pauses():
    calls, pauses = {"n": 0}, []

    def twice_down():
        calls["n"] += 1
        if calls["n"] <= 2:
            raise TimeoutError("silent connection")
        return "data"

    with pytest.warns(NetworkRetryWarning) as record:
        assert with_retries(twice_down, "cube", attempts=4, waits=[10, 30, 90], sleep=pauses.append) == "data"
    assert pauses == [10, 30] and len(record) == 2
    assert "cube: TimeoutError: silent connection; try 2 of 4 in 10 s" in str(record[0].message)


def test_gives_up_after_the_last_attempt():
    calls = {"n": 0}

    def always_down():
        calls["n"] += 1
        raise ConnectionError("down")

    with pytest.warns(NetworkRetryWarning), pytest.raises(ConnectionError):
        with_retries(always_down, "cube", attempts=3, waits=[0], sleep=lambda s: None)
    assert calls["n"] == 3


def test_other_errors_are_raised_at_once():
    calls = {"n": 0}

    def broken():
        calls["n"] += 1
        raise ValueError("bad")

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError):
            with_retries(broken, "cube", sleep=lambda s: None)
    assert calls["n"] == 1


def test_socket_timeout_is_restored():
    before = socket.getdefaulttimeout()
    with socket_timeout(5):
        assert socket.getdefaulttimeout() == 5
    assert socket.getdefaulttimeout() == before
    with pytest.raises(RuntimeError), socket_timeout(5):
        raise RuntimeError
    assert socket.getdefaulttimeout() == before


def test_prefer_ipv4_puts_ipv4_first_once(monkeypatch):
    v6 = (socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("64:ff9b::1", 443, 0, 0))
    v4 = (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("1.2.3.4", 443))
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: [v6, v4])
    prefer_ipv4()
    patched = socket.getaddrinfo
    prefer_ipv4()
    assert socket.getaddrinfo is patched  # idempotent: no second wrapper
    assert socket.getaddrinfo("example.org", 443) == [v4, v6]


@pytest.mark.parametrize("value, expected", [("1", "True"), ("", "False")])
def test_prefer_ipv4_from_the_environment(value, expected):
    import os
    import subprocess
    import sys

    code = "import socket, tess_sso; print(getattr(socket.getaddrinfo, '_tess_sso_ipv4_first', False))"
    env = dict(os.environ, TESS_SSO_PREFER_IPV4=value)
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
    assert out.stdout.strip() == expected


def _wrapped(cause: BaseException) -> Exception:
    """An error raised while handling ``cause``, as pyvo and astroquery do."""
    try:
        raise cause
    except BaseException:
        try:
            raise RuntimeError("DALFormatError: query failed")
        except RuntimeError as exc:
            return exc


def test_failed_dns_and_wrapped_network_errors_are_transient():
    import requests

    assert is_transient(socket.gaierror(8, "nodename nor servname provided"))
    assert is_transient(requests.ConnectionError("Failed to resolve 'gea.esac.esa.int'"))
    assert is_transient(_wrapped(socket.gaierror(8, "nodename nor servname provided")))
    assert not is_transient(_wrapped(ValueError("bad ADQL")))


def test_requests_http_errors_by_status():
    import requests

    def http_error(status):
        response = requests.Response()
        response.status_code = status
        return requests.HTTPError(f"HTTP {status}", response=response)

    assert is_transient(http_error(503))
    assert not is_transient(http_error(404))


def _certificate_failure() -> ssl.SSLCertVerificationError:
    exc = ssl.SSLCertVerificationError(1, "certificate verify failed: unable to get local issuer certificate")
    exc.verify_code, exc.verify_message = 20, "unable to get local issuer certificate"
    return exc


def test_failed_certificate_checks_are_not_retried():
    import requests

    assert not is_transient(_certificate_failure())
    assert not is_transient(URLError(_certificate_failure()))  # urllib, as lkspacecraft opens MAST
    assert not is_transient(requests.exceptions.SSLError(_certificate_failure()))
    assert not is_transient(_wrapped(URLError(_certificate_failure())))
    calls = {"n": 0}

    def untrusted():
        calls["n"] += 1
        raise URLError(_certificate_failure())

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(URLError):
            with_retries(untrusted, "cube", sleep=lambda s: None)
    assert calls["n"] == 1


def test_aiohttp_certificate_errors_are_not_retried():
    aiohttp = pytest.importorskip("aiohttp")
    from aiohttp.client_reqrep import ConnectionKey

    key = ConnectionKey("stpubdata.s3.amazonaws.com", 443, True, True, None, None, None)
    assert not is_transient(aiohttp.ClientConnectorCertificateError(key, _certificate_failure()))


def test_use_system_certificates_is_idempotent():
    truststore = pytest.importorskip("truststore")
    assert use_system_certificates()
    assert use_system_certificates()
    assert ssl.SSLContext is truststore.SSLContext


@pytest.mark.parametrize("value, expected", [("", "truststore"), ("0", "ssl")])
def test_system_certificates_from_the_environment(value, expected):
    import os
    import subprocess
    import sys

    pytest.importorskip("truststore")
    code = "import ssl, tess_sso; print(ssl.SSLContext.__module__.split('.')[0])"
    env = dict(os.environ, TESS_SSO_SYSTEM_CERTS=value)
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
    assert out.stdout.strip() == expected
