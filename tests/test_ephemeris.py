import numpy as np
import pytest
from astropy.table import MaskedColumn, Table

from tess_sso.designation import Target
from tess_sso.ephemeris import classify_horizons_error, ephemeris_path, load_ephemeris
from tess_sso.errors import ServiceError, UnknownDesignationError

TARGET = Target("95", "95 Arethusa (A867 WA)")


def fake_table(n=4):
    jd = 2460233.5 + np.arange(n) / 24
    return Table(
        {
            "datetime_jd": jd,
            "RA": np.linspace(72.6, 72.7, n),
            "DEC": np.full(n, 24.41),
            "V": MaskedColumn([12.3] * (n - 1) + [0.0], mask=[False] * (n - 1) + [True]),
            "r": np.full(n, 2.8),
            "delta": np.full(n, 1.877),
            "lighttime": np.full(n, 15.6),
            "alpha": np.full(n, 17.3),
            "targetname": ["95 Arethusa"] * n,
        }
    )


class CountingFetch:
    def __init__(self):
        self.calls = 0

    def __call__(self, designation, start, stop, step):
        self.calls += 1
        return fake_table()


def test_load_converts_units_and_masks(tmp_path):
    eph = load_ephemeris(TARGET, 2460233.5, 2460259.5, root=tmp_path, fetch=CountingFetch())
    assert eph.ra.shape == (4,)
    assert np.isnan(eph.v[-1]) and eph.v[0] == pytest.approx(12.3)
    assert eph.lighttime[0] == pytest.approx(15.6 / 1440)
    # TDB - UTC in 2023 is 69.184 s plus a periodic term of about a millisecond
    assert (eph.time_tdb[0] - eph.time_utc[0]) * 86400 == pytest.approx(69.184, abs=0.01)


def test_cached_after_first_query(tmp_path):
    fetch = CountingFetch()
    load_ephemeris(TARGET, 2460233.5, 2460259.5, root=tmp_path, fetch=fetch)
    eph = load_ephemeris(TARGET, 2460233.5, 2460259.5, root=tmp_path, fetch=fetch)
    assert fetch.calls == 1
    assert np.isnan(eph.v[-1])
    assert ephemeris_path(TARGET, 2460233.5, 2460259.5, "1h", tmp_path).exists()


def test_refresh_queries_again(tmp_path):
    fetch = CountingFetch()
    load_ephemeris(TARGET, 2460233.5, 2460259.5, root=tmp_path, fetch=fetch)
    load_ephemeris(TARGET, 2460233.5, 2460259.5, root=tmp_path, refresh=True, fetch=fetch)
    assert fetch.calls == 2


def test_error_classification():
    assert isinstance(classify_horizons_error(ValueError("No matches found.")), UnknownDesignationError)
    assert isinstance(classify_horizons_error(ValueError("Unknown target (zz)")), UnknownDesignationError)
    assert isinstance(classify_horizons_error(ConnectionError("timeout")), ServiceError)


def test_position_uncertainty_columns(tmp_path):
    def fetch(designation, start, stop, step):
        table = fake_table()
        table["RA_3sigma"] = [0.011] * 4
        table["DEC_3sigma"] = [0.002] * 4
        return table

    eph = load_ephemeris(TARGET, 2460233.5, 2460259.5, root=tmp_path, fetch=fetch)
    assert eph.ra_3sigma[0] == pytest.approx(0.011) and eph.dec_3sigma[0] == pytest.approx(0.002)
    older = load_ephemeris(TARGET, 2460233.5, 2460259.5, step="2h", root=tmp_path, fetch=CountingFetch())
    assert np.isnan(older.ra_3sigma).all()  # tables without the columns give NaN
