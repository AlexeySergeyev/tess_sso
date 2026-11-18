from unittest.mock import Mock, call

import numpy as np
import pytest
from astropy.table import MaskedColumn, Table

from pyvo.dal.exceptions import DALFormatError, DALQueryError, DALServiceError

from tess_sso import gaia
from tess_sso.errors import ServiceError
from tess_sso.network import NetworkRetryWarning
from tess_sso.gaia import (
    angular_separation,
    build_adql,
    cones_along_track,
    gaia_path,
    load_stars,
    propagate,
    tess_flux,
    tess_magnitude,
)


def test_tess_magnitude():
    # G = 10, BP-RP = 1: 10 - 0.00522555 + 0.0891337 - 0.633923 + 0.0324473
    assert tess_magnitude(10.0, 1.0) == pytest.approx(9.48243, abs=1e-5)
    assert tess_magnitude(10.0, np.nan) == pytest.approx(9.570)
    assert tess_magnitude(10.0, 5.0) == pytest.approx(tess_magnitude(10.0, 3.0))  # clipped


def test_tess_flux():
    assert tess_flux(20.44) == pytest.approx(1.0)
    assert tess_flux(15.44) == pytest.approx(100.0)


def test_propagate():
    ra, dec = propagate([10.0], [60.0], [1000.0], [2000.0], 2026.0)  # 10 years
    assert dec[0] == pytest.approx(60.0 + 20.0 / 3600)
    assert ra[0] == pytest.approx(10.0 + 10.0 / 3600 / np.cos(np.radians(60.0)))
    ra, dec = propagate([10.0], [60.0], [np.nan], [np.nan], 2026.0)
    assert (ra[0], dec[0]) == (10.0, 60.0)


def test_cones_cover_the_band():
    ra = np.linspace(72.0, 75.0, 500)
    dec = np.linspace(24.0, 25.0, 500)
    cones = cones_along_track(ra, dec, half_width_px=20)
    half_width = 20 * 21 / 3600
    for r, d in zip(ra, dec):
        assert min(angular_separation(cr, cd, r, d) for cr, cd, _ in cones) <= half_width
    assert all(rad == pytest.approx(half_width * np.sqrt(2)) for _, _, rad in cones)
    assert len(cones) < 40


def test_adql():
    adql = build_adql([(72.1, 24.2, 0.1), (72.3, 24.3, 0.1)], 17.25)
    assert adql.count("CIRCLE(") == 2 and "phot_g_mean_mag <= 17.25" in adql
    assert "gaiadr3.gaia_source_lite" in adql and "pmra" in adql


def fake_gaia(adql):
    fake_gaia.calls += 1
    return Table({
        "source_id": [1, 2, 2],
        "ra": [72.60, 72.70, 72.70],
        "dec": [24.40, 24.41, 24.41],
        "pmra": MaskedColumn([100.0, 0.0, 0.0], mask=[False, True, True]),
        "pmdec": MaskedColumn([0.0, 0.0, 0.0], mask=[False, True, True]),
        "phot_g_mean_mag": [12.0, 15.0, 15.0],
        "bp_rp": MaskedColumn([1.0, 0.0, 0.0], mask=[False, True, True]),
    })


def test_load_stars_caches_dedups_and_propagates(tmp_path):
    fake_gaia.calls = 0
    cones = [(72.65, 24.40, 0.2)]
    stars = load_stars(tmp_path, cones, 17.0, 2026.0, fake_gaia)
    again = load_stars(tmp_path, cones, 17.0, 2026.0, fake_gaia)
    assert fake_gaia.calls == 1 and len(stars) == len(again) == 2
    assert stars.has_pm.tolist() == [True, False]
    assert stars.ra[0] == pytest.approx(72.60 + 1.0 / 3600 / np.cos(np.radians(24.40)))
    assert stars.tmag[1] == pytest.approx(15.0 - 0.430)
    assert stars.flux[0] == pytest.approx(tess_flux(tess_magnitude(12.0, 1.0)))
    path = gaia_path(tmp_path, build_adql(cones, 17.0))
    assert path.parent == tmp_path and path.name.startswith("gaia_dr3_") and path.exists()


def test_gaia_uses_esa_first(monkeypatch):
    query = Mock(return_value=Table())
    monkeypatch.setattr(gaia, "_query_gaia", query)
    assert gaia.gaia_tap_fetch("SELECT 1") is query.return_value
    query.assert_called_once_with(gaia.GAIA_TAP_URL, "SELECT 1")


@pytest.mark.parametrize("failure", [DALFormatError(reason="E19"), DALServiceError(code=503),
                                     DALServiceError(cause=TimeoutError()), ConnectionError()])
def test_gaia_mirror_for_outage(monkeypatch, failure):
    result = Table()
    query = Mock(side_effect=[failure, result])
    monkeypatch.setattr(gaia, "_query_gaia", query)
    adql = build_adql([(72.1, 24.2, 0.1)], 17.25)
    mirror = adql.replace("POINT(ra, dec), CIRCLE(", "POINT('ICRS', ra, dec), CIRCLE('ICRS', ")
    with pytest.warns(NetworkRetryWarning, match="heidelberg"):
        assert gaia.gaia_tap_fetch(adql) is result
    assert query.call_args_list == [call(gaia.GAIA_TAP_URL, adql), call(gaia.GAIA_TAP_MIRROR_URL, mirror)]


def test_invalid_gaia_query_raises_without_mirror(monkeypatch):
    query = Mock(side_effect=DALQueryError("invalid column"))
    monkeypatch.setattr(gaia, "_query_gaia", query)
    with pytest.raises(ServiceError, match="invalid column"):
        gaia.gaia_tap_fetch("SELECT bad_column")
    assert query.call_count == 1


def test_both_gaia_services_fail(monkeypatch):
    query = Mock(side_effect=[DALFormatError(reason="maintenance page"), DALServiceError("mirror offline")])
    monkeypatch.setattr(gaia, "_query_gaia", query)
    with pytest.warns(NetworkRetryWarning), pytest.raises(ServiceError, match="maintenance page.*mirror offline"):
        gaia.gaia_tap_fetch("SELECT 1")
