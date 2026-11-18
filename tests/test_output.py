import os
import sys
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table

from conftest import GaussianPrf
from test_asteroid import ONE_ORBIT, make
from tess_sso.output import TESSMAG_ZERO_POINT, tess_magnitude, to_lightkurve, weighted_ra
from tess_sso.settings import Settings
from tess_sso.timing import LIGHT_TIME_AU_D, BarycentricFallbackWarning

SETTINGS = Settings(cutout_size=5, strip_width=21, group_count=2)
TIMECORR = 0.004  # the stub correction toward the asteroid; the fake cube's BARYCORR is 0.003
BINARIES_SRC = Path(os.environ.get("BINARIES_SRC", Path(__file__).resolve().parents[2] / "binaries" / "src"))


def pipeline(tmp_path, barycentric=lambda t, ra, dec: np.full(len(t), TIMECORR)):
    """Fake cube (6 frames, one star), groups of two exposures."""
    ast, _ = make(tmp_path, SETTINGS)
    track = ast.track(71, 2, 2)
    star_ra, star_dec = track.ra[0], track.dec[0] + 5 * 21 / 3600

    def gaia(adql):
        return Table({"source_id": [7], "ra": [star_ra], "dec": [star_dec], "pmra": [0.0], "pmdec": [0.0],
                      "phot_g_mean_mag": [11.0], "bp_rp": [0.8]})

    ast._gaia, ast._prf, ast._orbit_times = gaia, (lambda s, c, d: GaussianPrf()), (lambda: ONE_ORBIT)
    ast._barycentric = barycentric
    ast.download(71, 2, 2)
    return ast


@pytest.fixture(scope="module")
def written(tmp_path_factory):
    ast = pipeline(tmp_path_factory.mktemp("lc"))
    ast.lightcurve(71, 2, 2)
    return ast, ast.lightcurve_path(71, 2, 2)


def test_file_name_and_hdus(written):
    ast, path = written
    assert path.name == "tess-95-s0071-2-2-lc.fits" and path.parent == ast.results_dir(71, 2, 2)
    with fits.open(path) as hdul:
        assert [h.name for h in hdul] == ["PRIMARY", "LIGHTCURVE_PRF", "LIGHTCURVE_AP", "EXTRAS", "GROUPS", "SEGMENTS"]
        # one row per group in every group table, the same CADENCENO and GROUP_ID in each
        prf, ap, extras = (hdul[n].data for n in ("LIGHTCURVE_PRF", "LIGHTCURVE_AP", "EXTRAS"))
        assert len(prf) == len(ap) == len(extras) == 3 and len(hdul["GROUPS"].data) == 6
        for table in (ap, extras):
            np.testing.assert_array_equal(table["CADENCENO"], prf["CADENCENO"])
            np.testing.assert_array_equal(table["GROUP_ID"], prf["GROUP_ID"])


def test_group_rows_are_weighted_means_of_the_members(written):
    _, path = written
    with fits.open(path) as hdul:
        extras, members = hdul["EXTRAS"].data, hdul["GROUPS"].data
    for row in extras:
        m = members[members["GROUP_ID"] == row["GROUP_ID"]]
        w = m["WEIGHT"]
        assert w.sum() == pytest.approx(1.0)
        for name in ("TIME", "TIMECORR", "ORIGINAL_TIME", "OBS_DISTANCE", "TIME_EMIT"):  # float64
            assert row[name] == pytest.approx(np.sum(w * m[name]), rel=0, abs=1e-12), name
        for name in ("EPHEM1", "EPHEM2"):  # float32
            assert row[name] == pytest.approx(np.sum(w * m[name]), rel=1e-6), name
        # CADENCENO: the member closest to the group time
        closest = m["CADENCENO"][np.argmin(np.abs(m["ORIGINAL_TIME"] - row["ORIGINAL_TIME"]))]
        assert row["CADENCENO"] == closest


def test_time_columns(written):
    _, path = written
    with fits.open(path) as hdul:
        assert hdul[0].header["TIMECORR"] == "lkspacecraft"
        members, extras, prf = hdul["GROUPS"].data, hdul["EXTRAS"].data, hdul["LIGHTCURVE_PRF"].data
        np.testing.assert_allclose(members["TIMECORR"], TIMECORR)
        np.testing.assert_allclose(members["ORIGINAL_TIMECORR"], 0.003)
        np.testing.assert_allclose(members["TIME"], members["ORIGINAL_TIME"] - 0.003 + TIMECORR, rtol=0, atol=1e-9)
        emit = extras["TIME"] - extras["TIMECORR"] - extras["OBS_DISTANCE"] * LIGHT_TIME_AU_D
        np.testing.assert_allclose(extras["TIME_EMIT"], emit, rtol=0, atol=1e-9)
        np.testing.assert_array_equal(prf["TIME_EMIT"], extras["TIME_EMIT"])
        np.testing.assert_array_equal(prf["TIME"], extras["TIME"])
        for name in ("LIGHTCURVE_PRF", "LIGHTCURVE_AP", "EXTRAS", "GROUPS"):
            for column in ("TIME", "TIME_EMIT"):
                assert hdul[name].data[column].dtype == np.dtype(">f8")
        assert np.all(prf["TIME_START"] < prf["TIME"]) and np.all(prf["TIME"] < prf["TIME_STOP"])


def test_time_keywords_in_every_table(written):
    _, path = written
    with fits.open(path) as hdul:
        for hdu in hdul:
            assert hdu.header["TIMESYS"] == "TDB" and hdu.header["BJDREFI"] == 2457000
            assert hdu.header["TIMEREF"] == "SOLARSYSTEM" and hdu.header["TIMEUNIT"] == "d"


def test_primary_header_provenance(written):
    ast, path = written
    h = fits.getheader(path)
    assert h["TELESCOP"] == "TESS" and "lightcurve" in h["CREATOR"].lower()
    assert h["OBJECT"] == "95" and h["SECTOR"] == 71 and h["CAMERA"] == 2 and h["CCD"] == 2
    assert h["SETHASH"] == SETTINGS.hash() and h["GRPCNT"] == 2 and h["CUTSIZE"] == 5
    assert h["TESSMAG0"] == TESSMAG_ZERO_POINT and h["GAIAREL"] == "DR3" and h["PRFVER"] == "gaussian-test"
    assert h["BAD_SPOC"] == 17087 and h["SSOBIT0"] == "WCS_FALLBACK" and h["SSOBIT11"] == "ONE_STAR_MASK_ONLY"
    assert h["LTCORR"] is False and "OBS_DISTANCE" in h["TIMEEMIT"]
    assert h["OBS_DIST"] == pytest.approx(1.9)


def test_quality_columns_and_flux_kept_when_not_positive(written):
    _, path = written
    with fits.open(path) as hdul:
        prf, members = hdul["LIGHTCURVE_PRF"].data, hdul["GROUPS"].data
    # the fake cube has no asteroid: some fluxes are <= 0, written as they are, with NaN magnitudes
    assert np.all(np.isfinite(prf["FLUX"]))
    assert np.all(np.isnan(prf["TESSMAG"][prf["FLUX"] <= 0]))
    for row in prf:
        m = members[members["GROUP_ID"] == row["GROUP_ID"]]
        assert row["QUALITY"] == np.bitwise_or.reduce(m["QUALITY"])
        assert row["SSO_QUALITY"] & np.bitwise_or.reduce(m["SSO_QUALITY"]) == np.bitwise_or.reduce(m["SSO_QUALITY"])
    for hdu in ("LIGHTCURVE_PRF", "LIGHTCURVE_AP"):
        t = fits.getdata(path, hdu)
        # random and coherent errors, each its components in quadrature
        np.testing.assert_allclose(t["FLUX_ERR"] ** 2, t["FLUX_ERR_PIX"] ** 2 + t["FLUX_ERR_BKG"] ** 2 + t["FLUX_ERR_EXCESS"] ** 2, rtol=1e-5)
        np.testing.assert_allclose(t["FLUX_ERR_SYS"] ** 2, t["FLUX_ERR_STAR"] ** 2 + t["FLUX_ERR_BKG_SYS"] ** 2, rtol=1e-5)
        np.testing.assert_allclose(t["FLUX_BKG_ERR"] ** 2, t["FLUX_ERR_BKG"] ** 2 + t["FLUX_ERR_BKG_SYS"] ** 2, rtol=1e-5)
    segments = fits.getdata(path, "SEGMENTS")
    assert np.all(segments["EXCESS"][np.isfinite(segments["EXCESS"])] >= 0)


def test_background_channels(written):
    _, path = written
    with fits.open(path) as hdul:
        for name in ("LIGHTCURVE_PRF", "LIGHTCURVE_AP"):
            t = hdul[name].data
            # the fake cube has no noise, so the residual statistics are 0, but they are written
            assert np.all(np.isfinite(t["BKG_MAD"]) & (t["BKG_MAD"] >= 0))
            assert np.all(np.isfinite(t["BKG_CHI2"]) & (t["BKG_CHI2"] >= 0))
            assert np.all((t["BKG_USABLE"] > 0) & (t["BKG_USABLE"] <= 1))


def test_prf_shift_columns(written):
    _, path = written
    with fits.open(path) as hdul:
        # the fake cube has no asteroid: no offset is measured, so the PRF stays at the prediction
        for name in ("LIGHTCURVE_PRF", "LIGHTCURVE_AP"):
            assert not hdul[name].data["PRF_SHIFT1"].any() and not hdul[name].data["PRF_SHIFT2"].any()
        segments = hdul["SEGMENTS"].data
        assert not segments["OFFSET_APPLIED"].any()
        assert hdul[0].header["PRFPOS"] == "segment"
        assert hdul[0].header["RESMODEL"] == "local_pointing" and hdul[0].header["LOCALBKG"] is True


def test_segments_hdu(written):
    ast, path = written
    segments = ast.segments(71, 2, 2)
    table = fits.getdata(path, "SEGMENTS")
    assert list(table["SEGMENT"]) == [s.index for s in segments]
    assert list(table["N_EXP"]) == [s.n_frames for s in segments]
    assert list(table["SOURCE"]) == [s.source for s in segments]


def test_fallback_to_the_spoc_correction(tmp_path):
    def broken(t, ra, dec):
        raise RuntimeError("no kernels")

    ast = pipeline(tmp_path, barycentric=broken)
    with pytest.warns(BarycentricFallbackWarning):
        hdul = ast.lightcurve(71, 2, 2, save=False)
    assert hdul[0].header["TIMECORR"] == "SPOC"
    np.testing.assert_array_equal(hdul["GROUPS"].data["TIMECORR"], hdul["GROUPS"].data["ORIGINAL_TIMECORR"])
    np.testing.assert_array_equal(hdul["GROUPS"].data["TIME"], hdul["GROUPS"].data["ORIGINAL_TIME"])
    assert not ast.lightcurve_path(71, 2, 2).exists()


def test_tess_magnitude():
    flux = 10 ** ((TESSMAG_ZERO_POINT - 12.0) / 2.5)
    mag, err, upper, lower = tess_magnitude(np.array([flux, 0.0, -5.0]), np.array([0.01 * flux, 1.0, 1.0]))
    assert mag[0] == pytest.approx(12.0) and np.isnan(mag[1:]).all()
    assert err[0] == pytest.approx(np.hypot(0.05, 2.5 / np.log(10) * 0.01))
    assert upper[0] > err[0] > lower[0] > 0.05


def test_weighted_ra_across_zero():
    assert weighted_ra(np.array([359.9, 0.1]), np.array([0.5, 0.5])) == pytest.approx(0.0, abs=1e-9)
    assert weighted_ra(np.array([10.0, 20.0]), np.array([0.25, 0.75])) == pytest.approx(17.5)


def test_lightkurve_reads_the_file(written):
    lk = pytest.importorskip("lightkurve")
    _, path = written
    lc = lk.io.tess.read_tess_lightcurve(str(path), quality_bitmask="default")
    assert len(lc) == 3 and lc.time.format == "btjd" and lc.time.scale == "tdb"
    assert "tessmag" in lc.columns and "time_emit" in lc.columns
    assert type(lk.read(str(path))).__name__ == "TessLightCurve"


def test_to_lightkurve_chooses_the_time(written):
    pytest.importorskip("lightkurve")
    _, path = written
    table = fits.getdata(path, "LIGHTCURVE_PRF")
    bary = to_lightkurve(path)
    emit = to_lightkurve(path, time="TIME_EMIT")
    np.testing.assert_allclose(bary.time.value, table["TIME"], rtol=0, atol=1e-9)
    np.testing.assert_allclose(emit.time.value, table["TIME_EMIT"], rtol=0, atol=1e-9)
    np.testing.assert_allclose(emit["time_barycentric"], table["TIME"], rtol=0, atol=1e-9)
    assert {"flux_err_pix", "flux_err_star", "flux_err_bkg", "sso_quality"} <= set(bary.columns)
    ap = to_lightkurve(path, hdu="LIGHTCURVE_AP")
    np.testing.assert_allclose(ap.flux.value, fits.getdata(path, "LIGHTCURVE_AP")["FLUX"], rtol=1e-6)
    with pytest.raises(ValueError):
        to_lightkurve(path, time="CADENCENO")


@pytest.fixture
def tessbin():
    if not BINARIES_SRC.exists():
        pytest.skip(f"binaries source not found at {BINARIES_SRC} (set BINARIES_SRC)")
    pytest.importorskip("lightkurve")
    sys.path.insert(0, str(BINARIES_SRC))
    try:
        from tessbin import extraction, quality
    finally:
        sys.path.remove(str(BINARIES_SRC))
    return extraction, quality


def test_binaries_readers(written, tessbin):
    extraction, quality = tessbin
    _, path = written

    class Obs:  # the attributes the readers use
        lc_path = path
        gaia_path = path.with_name("no-gaia.fits")

    with fits.open(path) as hdul:
        prf, extras = hdul["LIGHTCURVE_PRF"].data, hdul["EXTRAS"].data
    offsets = extraction.light_time_offsets(path, prf["CADENCENO"])
    np.testing.assert_allclose(prf["TIME"] + offsets, prf["TIME_EMIT"], rtol=0, atol=1e-8)
    (sto,) = extraction.extras_columns(path, prf["CADENCENO"], "STO_ANGLE")
    np.testing.assert_allclose(sto, extras["STO_ANGLE"])
    x, y, y_err = extraction.load_lc(Obs())
    good = np.isfinite(prf["TESSMAG"])
    np.testing.assert_allclose(x, prf["TIME_EMIT"][good], rtol=0, atol=1e-8)
    geometry = extraction.observing_geometry(Obs())
    assert len(geometry) == len(prf)
    diagnostics = quality.load_diagnostics(Obs())
    # binaries scores a tess_sso file on its level, errors, and instrumental flags (binaries phase 11c)
    assert set(diagnostics.channels) == {"background", "flux_uncertainty"}
    assert set(diagnostics.flags) == {"tess_sso_flags"}
