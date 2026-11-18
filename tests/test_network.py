"""Tests against JPL, MAST, MIT, and the Gaia archive. Run with: pytest -m network"""

import numpy as np
import pytest

from tess_sso.storage.download import TessCubeSource, fetch_pixels
from tess_sso.storage.store import PixelStore


@pytest.mark.network
def test_real_pixels(tmp_path):
    source = TessCubeSource(71, 2, 2)
    store = PixelStore(tmp_path / "s0071-2-2", 71, 2, 2)
    frames, flux, err = fetch_pixels([(1000, 1000), (1000, 1001)], source, store)
    assert flux.shape == (len(frames), 2)
    assert np.isfinite(flux).mean() > 0.9
    assert frames.provenance["OBJKEY"].endswith("tess-s0071-2-2-cube.fits")
    assert frames.provenance["DATA_REL"] == 100
    assert frames.provenance["CUBEDATE"] == "2023-12-10"


@pytest.mark.network
def test_find_observations_for_95(tmp_path):
    from tess_sso.search import find_observations

    table = find_observations("95", root=tmp_path)
    visits = {(r["sector"], r["camera"], r["ccd"]) for r in table}
    assert {(70, 4, 4), (71, 2, 2), (89, 1, 3), (89, 1, 4)} <= visits
    row = table[(table["sector"] == 71) & (table["camera"] == 2) & (table["ccd"] == 2)][0]
    assert row["on_ccd"] and row["v_known"]
    assert table.meta["FULLNAME"].startswith("95 Arethusa")


@pytest.mark.network
def test_secondary_designation_and_unknown():
    from tess_sso.designation import resolve_designation
    from tess_sso.errors import UnknownDesignationError

    assert resolve_designation("2010 PH5").designation == "722079"
    with pytest.raises(UnknownDesignationError):
        resolve_designation("zzzzqq")


@pytest.mark.network
def test_track_and_plan_for_95(tmp_path):
    """Fetches the sector 71 frame table (not the pixels) and plans the download."""
    import numpy as np

    from tess_sso.asteroid import Asteroid
    from tess_sso.geometry import wcs_disagreement

    ast = Asteroid("95", root=tmp_path)
    track = ast.track(71, 2, 2)
    assert len(track) > 10000
    assert track.on_science.mean() > 0.95
    assert track.wcs_fallback.mean() < 0.1
    assert wcs_disagreement(track, 71, 2, 2) < 0.5
    plan = ast.plan(71, 2, 2)
    assert plan.missing == sorted(plan.strip_pixels) and 20000 < len(plan.strip_pixels) < 30000
    assert 2.0e9 < plan.missing_bytes < 2.5e9
    assert not ast.store(71, 2, 2).files()  # planning downloads no pixels


@pytest.mark.network
def test_segments_for_95_sector_71(tmp_path, recwarn):
    from tess_sso.asteroid import Asteroid
    from tess_sso.segments import DownlinkMismatchWarning

    segments = Asteroid("95", root=tmp_path).segments(71, 2, 2)
    assert len(segments) == 4
    assert [round(s.stop, 1) for s in segments[:3]] == [3240.5, 3247.0, 3253.4]
    assert not [w for w in recwarn if issubclass(w.category, DownlinkMismatchWarning)]


@pytest.mark.network
def test_masks_for_95_short_stretch(tmp_path):
    """Downloads a few hundred pixels (~30 MB), the Gaia stars around them, and the lkprf model."""
    import numpy as np

    from tess_sso.asteroid import Asteroid
    from tess_sso.settings import Settings

    ast = Asteroid("95", root=tmp_path, settings=Settings(cutout_size=5, strip_width=11))
    stretch = (3235.0, 3235.05)
    ast.download(71, 2, 2, time_range=stretch)
    result = ast.masks(71, 2, 2, time_range=stretch)
    first = result.per_segment[0]
    assert len(result.per_segment) == 4
    assert first.asteroid.any()
    # the asteroid (V ~ 12) is masked down to the noise, so in an 11-px strip it covers most of
    # segment 0; the other segments, without the asteroid, keep most of the strip
    assert all(not m.asteroid.any() and m.usable_fraction > 0.2 for m in result.per_segment[1:])
    data_only = first.star_data & ~first.star_gaia
    assert data_only.sum() <= 0.1 * max(first.star_data.sum(), 1)  # Gaia explains the visible stars
    assert result.header["PRFSET"] == "sectors 4+" and result.header["NSTARS"] > 0
    lkprf_dir = tmp_path / "reference" / "lkprf"
    assert lkprf_dir.is_dir() and any(lkprf_dir.iterdir())
    assert (ast.results_dir(71, 2, 2) / "masks.fits").exists()


@pytest.mark.network
def test_scattered_light_for_95_short_stretch(tmp_path):
    """Same stretch and narrow strip as the mask test: segment 0 has too few background pixels."""
    import warnings

    import numpy as np

    from tess_sso.asteroid import Asteroid
    from tess_sso.scattered_light import FewBackgroundPixelsWarning
    from tess_sso.settings import Settings

    ast = Asteroid("95", root=tmp_path, settings=Settings(cutout_size=5, strip_width=11))
    stretch = (3235.0, 3235.05)
    ast.download(71, 2, 2, time_range=stretch)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = ast.scattered_light(71, 2, 2, time_range=stretch)
    assert any(issubclass(w.category, FewBackgroundPixelsWarning) and "segment 0" in str(w.message) for w in caught)
    assert [seg.segment for seg in model.segments] == [1, 2, 3]
    for seg in model.segments:
        assert np.nanmedian(seg.frame_rms) < 2.0  # per-exposure noise is ~1.3 e-/s
    assert (ast.results_dir(71, 2, 2) / "scattered_light.fits").exists()


@pytest.mark.network
def test_stellar_model_for_95_short_stretch(tmp_path):
    """Narrow strip: segment 0 has no scattered-light model, so its stars are not fitted."""
    import warnings

    import numpy as np

    from tess_sso.asteroid import Asteroid, MissingModelWarning
    from tess_sso.settings import Settings

    ast = Asteroid("95", root=tmp_path, settings=Settings(cutout_size=5, strip_width=11))
    stretch = (3235.0, 3235.05)
    ast.download(71, 2, 2, time_range=stretch)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = ast.stellar_model(71, 2, 2, time_range=stretch)
    assert any(issubclass(w.category, MissingModelWarning) for w in caught)
    assert [seg.segment for seg in model.segments] == [1, 2, 3]
    for seg in model.segments:
        assert seg.diagnostics.fitted.any()
        assert np.abs(seg.shifts).max() < 0.5  # pixels
    assert (ast.results_dir(71, 2, 2) / "stellar.fits").exists()


@pytest.mark.network
def test_photometry_for_95_short_stretch(tmp_path):
    """Strip 21 over ~1.2 h so segment 0 has models; 30-minute groups."""
    import numpy as np

    from tess_sso.asteroid import Asteroid
    from tess_sso.settings import Settings

    ast = Asteroid("95", root=tmp_path, settings=Settings(cutout_size=7, strip_width=21, group_minutes=30))
    stretch = (3235.0, 3235.05)
    ast.download(71, 2, 2, time_range=stretch)
    table = ast.photometry(71, 2, 2, time_range=stretch).to_table()
    assert 2 <= len(table) <= 4
    # asteroid 95 at V ~ 12: a few thousand e-/s, measured to better than 1 %
    assert np.all((table["flux"] > 1000) & (table["flux"] < 10000))
    assert np.all(table["flux_err"] < 0.01 * table["flux"])
    assert np.all(np.abs(table["ap_flux"] / table["flux"] - 1) < 0.05)
    assert np.all(np.abs(table["offset_row"]) < 0.5) and np.all(np.abs(table["offset_col"]) < 0.5)
    assert (ast.results_dir(71, 2, 2) / "photometry.ecsv").exists()


@pytest.mark.network
def test_prf_position_for_95(tmp_path):
    """Acceptance criterion 4 (phase 10): median RED_CHI2 <= 3 in the clean field with the default PRF position."""
    import numpy as np

    from tess_sso.asteroid import Asteroid
    from tess_sso.settings import Settings

    ast = Asteroid("95", root=tmp_path)
    stretch = (3235.0, 3235.1)
    ast.download(71, 2, 2, time_range=stretch)
    masks = ast.masks(71, 2, 2, time_range=stretch, save=False)
    scattered = ast.scattered_light(71, 2, 2, time_range=stretch, masks=masks, save=False)
    stellar = ast.stellar_model(71, 2, 2, time_range=stretch, masks=masks, scattered=scattered, save=False)
    tables = {}
    for mode in ("predicted", "segment"):
        ast.settings = Settings(prf_position=mode)
        phot = ast.photometry(71, 2, 2, time_range=stretch, masks=masks, scattered=scattered, stellar=stellar, save=False)
        tables[mode] = phot.to_table()
    assert np.median(tables["segment"]["red_chi2"]) <= 3 < np.median(tables["predicted"]["red_chi2"])
    assert phot.segment_offsets[0].applied
    ratio = np.array(tables["segment"]["flux"]) / np.array(tables["predicted"]["flux"])
    assert np.all(np.abs(ratio - 1) < 0.02)


@pytest.mark.network
def test_light_curve_for_95_short_stretch(tmp_path):
    """The light-curve file for the photometry stretch; the barycentric correction from the SPICE kernels."""
    import numpy as np
    from astropy.io import fits

    from tess_sso.asteroid import Asteroid
    from tess_sso.geometry import BTJD0, average_wcs
    from tess_sso.settings import Settings
    from tess_sso.timing import lkspacecraft_correction

    ast = Asteroid("95", root=tmp_path, settings=Settings(cutout_size=7, strip_width=21, group_minutes=30))
    stretch = (3235.0, 3235.05)
    ast.download(71, 2, 2, time_range=stretch)
    ast.lightcurve(71, 2, 2, time_range=stretch)
    with fits.open(ast.lightcurve_path(71, 2, 2)) as hdul:
        assert hdul[0].header["TIMECORR"] == "lkspacecraft"
        extras = hdul["EXTRAS"].data
        # toward the asteroid, a few seconds from SPOC's CCD-centre value
        assert np.all(np.abs(extras["TIMECORR"] - extras["ORIGINAL_TIMECORR"]) * 86400 < 30)

    # SPOC's BARYCORR is the correction toward the CCD centre: the kernels reproduce it
    frames = ast.store(71, 2, 2).frames()
    ra, dec = average_wcs(frames.table).all_pix2world([[1067.5, 1023.5]], 0)[0]
    idx = np.array([0, len(frames) // 2, len(frames) - 1])
    t_sc = frames.time[idx] - frames.barycorr[idx] + BTJD0
    tc = lkspacecraft_correction(t_sc, np.full(3, ra), np.full(3, dec))
    assert np.all(np.abs(tc - frames.barycorr[idx]) * 86400 < 0.1)

    # the pixel product: same rows, and the asteroid is the brightest thing in the model
    ast.pixel_product(71, 2, 2, time_range=stretch)
    with fits.open(ast.pixel_product_path(71, 2, 2)) as hdul:
        pixels = hdul["PIXELS"].data
        assert len(pixels) == len(extras) and pixels["FLUX"].shape[1:] == (7, 7)
        centre = pixels["ASTEROID_MODEL"][:, 3, 3]
        assert np.all(centre == np.nanmax(pixels["ASTEROID_MODEL"], axis=(1, 2)))


@pytest.mark.network
def test_mast_and_naif_certificates_are_trusted():
    from urllib.request import urlopen

    import tess_sso  # noqa: F401  (uses the system trust store)

    for url in ("https://archive.stsci.edu/missions/tess/models/", "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/"):
        with urlopen(url, timeout=60) as response:
            assert response.status == 200
