import numpy as np
import pytest
from astropy.table import Table

from tess_sso.storage.store import PixelStore
from tess_sso.designation import Target
from tess_sso.errors import UnknownDesignationError
from tess_sso.search import (
    cadence_seconds,
    filter_by_brightness,
    find_observations,
    format_observations,
    sector_jd_range,
    trajectory_on_ccd,
)

TARGET = Target("95", "95 Arethusa (A867 WA)")
# Asteroid 95 on 2023-10-17, seen from TESS: on sector 71, camera 2, CCD 2.
RA, DEC = 72.67498, 24.41023


def test_cadence_by_sector():
    assert [cadence_seconds(s) for s in (1, 26, 27, 55, 56, 71)] == [1800, 1800, 600, 600, 200, 200]


def test_sector_range_from_tesswcs():
    assert sector_jd_range(71) == (2460233.5, 2460259.5)
    with pytest.raises(ValueError):
        sector_jd_range(99999)


def test_trajectory_on_ccd_uses_wcs():
    on = trajectory_on_ccd(71, 2, 2, np.array([RA, RA + 40]), np.array([DEC, DEC]))
    assert on.tolist() == [True, False]
    assert not trajectory_on_ccd(71, 2, 1, np.array([RA]), np.array([DEC])).any()


def horizons_track(designation, start, stop, step):
    n = 24
    return Table(
        {
            "datetime_jd": start + np.arange(n) / 24,
            "RA": np.linspace(RA, RA + 0.05, n),
            "DEC": np.full(n, DEC),
            "V": np.linspace(12.3, 12.5, n),
            "r": np.full(n, 2.8),
            "delta": np.full(n, 1.877),
            "lighttime": np.full(n, 15.6),
            "alpha": np.full(n, 17.3),
        }
    )


class Sectors:
    def __init__(self, visits):
        self.visits, self.calls = visits, 0

    def __call__(self, designation):
        self.calls += 1
        return self.visits


def run(tmp_path, visits, **kwargs):
    return find_observations(
        "95", root=tmp_path, resolve=lambda name: TARGET, sectors=Sectors(visits) if isinstance(visits, list) else visits,
        horizons=horizons_track, **kwargs,
    )


def test_table_rows_and_wcs_check(tmp_path):
    table = run(tmp_path, [(71, 2, 2), (71, 2, 1)])
    assert table.meta["FULLNAME"] == "95 Arethusa (A867 WA)"
    assert [(r["sector"], r["camera"], r["ccd"]) for r in table] == [(71, 2, 2), (71, 2, 1)]
    good, missed = table[0], table[1]
    assert good["on_ccd"] and good["hours_on_ccd"] == pytest.approx(24.0)
    assert good["cadence_s"] == 200 and good["v_band"] == "V"
    assert good["v_min"] == pytest.approx(12.3) and good["v_max"] == pytest.approx(12.5)
    assert good["start_btjd"] == pytest.approx(3233.5, abs=0.01)
    assert not missed["on_ccd"] and np.isnan(missed["start_btjd"]) and not missed["v_known"]


def test_no_coverage_gives_empty_table(tmp_path):
    table = run(tmp_path, [])
    assert len(table) == 0 and "v_min" in table.colnames


def test_unknown_designation_propagates(tmp_path):
    def resolve(name):
        raise UnknownDesignationError("nope")

    with pytest.raises(UnknownDesignationError):
        find_observations("zz", root=tmp_path, resolve=resolve, sectors=Sectors([]), horizons=horizons_track)


def test_visits_cached_and_refreshable(tmp_path):
    sectors = Sectors([(71, 2, 2)])
    run(tmp_path, sectors)
    run(tmp_path, sectors)
    assert sectors.calls == 1
    run(tmp_path, sectors, refresh=True)
    assert sectors.calls == 2


def test_a_fits_visit_cache_is_kept_as_ecsv(tmp_path):
    from astropy.io import fits

    from tess_sso.storage.ecsv_io import read_table
    from tess_sso.storage.fits_io import write_atomic

    primary = fits.PrimaryHDU()
    primary.header["QUERYDAT"] = "2026-09-24T21:20:00+00:00"
    visits = fits.table_to_hdu(Table({"sector": [71], "camera": [2], "ccd": [2]}, dtype=[np.int32] * 3))
    write_atomic(tmp_path / "95" / "observations.fits", fits.HDUList([primary, visits]))
    sectors = Sectors([(99, 1, 1)])
    assert list(run(tmp_path, sectors)["sector"]) == [71] and sectors.calls == 0
    assert read_table(tmp_path / "95" / "observations.ecsv").meta["QUERYDAT"] == "2026-09-24T21:20:00+00:00"


def test_stored_pixels_counted(tmp_path):
    store = PixelStore(tmp_path / "95" / "s0071-2-2", 71, 2, 2)
    flux = np.zeros((3, 2), np.float32)
    store.write_block([(1000, 1000), (1000, 1001)], flux, flux)
    assert run(tmp_path, [(71, 2, 2)])["local_pixels"][0] == 2


def test_brightness_filter_keeps_unknown_by_default(tmp_path):
    table = run(tmp_path, [(71, 2, 2), (71, 2, 1)])
    assert len(filter_by_brightness(table, 12.0)) == 1  # only the unknown-V row
    assert len(filter_by_brightness(table, 12.0, keep_unknown=False)) == 0
    assert len(filter_by_brightness(table, 13.0)) == 2


def test_format_starts_with_resolved_name(tmp_path):
    text = format_observations(run(tmp_path, [(71, 2, 2), (71, 2, 1)]))
    lines = text.splitlines()
    assert lines[0] == "95 Arethusa (A867 WA): 2 observation(s)"
    assert "s0071-2-2" in lines[1] and "V 12.3-12.5" in lines[1]
    assert "trajectory misses the CCD" in lines[2]
