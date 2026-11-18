import pytest
import requests

from tess_sso.designation import Target, designations_path, load_target, normalize_name, resolve_designation
from tess_sso.errors import AmbiguousDesignationError, ServiceError, UnknownDesignationError
from tess_sso.network import NetworkRetryWarning


class FakeResponse:
    def __init__(self, payload, status=200):
        self.payload, self.status = payload, status

    def raise_for_status(self):
        if self.status >= 400:
            raise requests.HTTPError(f"HTTP {self.status}")

    def json(self):
        return self.payload


def fake_get(payload, status=200, calls=None):
    def get(url, params, timeout):
        if calls is not None:
            calls.append(params["sstr"])
        return FakeResponse(payload, status)

    return get


def test_normalize_name():
    assert normalize_name("2000_PH5") == "2000 PH5"
    assert normalize_name("  2000   PH5 ") == "2000 PH5"
    assert normalize_name(95) == "95"


def test_resolves_to_primary_designation():
    calls = []
    payload = {"object": {"des": "722079", "fullname": "722079 (2005 EY17 = 2010 PH5)"}}
    target = resolve_designation("2010_PH5", get=fake_get(payload, calls=calls))
    assert target == Target("722079", "722079 (2005 EY17 = 2010 PH5)")
    assert calls == ["2010 PH5"]


def test_dir_name():
    assert Target("2000 PH5", "x").dir_name == "2000_PH5"


def test_unknown():
    payload = {"message": "specified object was not found", "code": "200"}
    with pytest.raises(UnknownDesignationError, match="not found"):
        resolve_designation("zzzzqq", get=fake_get(payload))


def test_ambiguous_lists_candidates():
    payload = {"code": "300", "list": [{"pdes": "1", "name": "1 Ceres"}, {"pdes": "2", "name": "2 Pallas"}]}
    with pytest.raises(AmbiguousDesignationError) as excinfo:
        resolve_designation("x", get=fake_get(payload))
    assert excinfo.value.candidates == ["1 Ceres", "2 Pallas"]


def test_service_failure():
    with pytest.raises(ServiceError):
        resolve_designation("95", get=fake_get({}, status=503))

    calls = []

    def broken(url, params, timeout):
        calls.append(url)
        raise requests.ConnectionError("no network")

    with pytest.warns(NetworkRetryWarning), pytest.raises(ServiceError):
        resolve_designation("95", get=broken)
    assert len(calls) == 4  # tried again before giving up


def test_empty_name():
    with pytest.raises(UnknownDesignationError):
        resolve_designation("  ", get=fake_get({}))


def test_a_name_is_looked_up_once_then_read_from_disk(tmp_path):
    calls = []

    def resolve(name):
        calls.append(name)
        return Target("722079", "722079 (2010 PH5)")

    assert load_target("2010 PH5", tmp_path, resolve) == Target("722079", "722079 (2010 PH5)")
    assert load_target(" 2010_PH5 ", tmp_path, resolve).designation == "722079"  # same normalized name

    def offline(name):
        raise AssertionError("asked JPL although the name is known")

    assert load_target("2010 PH5", tmp_path, offline).fullname == "722079 (2010 PH5)"
    assert calls == ["2010 PH5"]
    assert designations_path(tmp_path).exists() and not designations_path(tmp_path).with_suffix(".json.part").exists()


def test_unknown_names_are_not_kept(tmp_path):
    def unknown(name):
        raise UnknownDesignationError("nope")

    with pytest.raises(UnknownDesignationError):
        load_target("zzzz", tmp_path, unknown)
    assert not designations_path(tmp_path).exists()
