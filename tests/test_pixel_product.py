import numpy as np
import pytest
from astropy.io import fits

from test_output import pipeline
from tess_sso.pixel_product import pixel_hdulist
from tess_sso.photometry import Photometry
from tess_sso.quality import PixelMask


@pytest.fixture(scope="module")
def written(tmp_path_factory):
    ast = pipeline(tmp_path_factory.mktemp("tpf"))
    phot = ast.photometry(71, 2, 2, save=False)
    ast.lightcurve(71, 2, 2, photometry=phot)
    ast.pixel_product(71, 2, 2, photometry=phot)
    return ast, ast.pixel_product_path(71, 2, 2), ast.lightcurve_path(71, 2, 2)


def test_file_name_hdus_and_rows(written):
    ast, path, lc_path = written
    assert path.name == "tess-95-s0071-2-2-tpf.fits" and path.parent == ast.results_dir(71, 2, 2)
    with fits.open(path) as hdul, fits.open(lc_path) as lc:
        assert [h.name for h in hdul] == ["PRIMARY", "PIXELS", "APERTURE", "EXTRAS"]
        pixels = hdul["PIXELS"].data
        # one row per group, as in the light-curve file
        np.testing.assert_array_equal(pixels["CADENCENO"], lc["LIGHTCURVE_PRF"].data["CADENCENO"])
        np.testing.assert_array_equal(pixels["TIME"], lc["LIGHTCURVE_PRF"].data["TIME"])
        assert pixels["FLUX"].shape == (3, 5, 5)  # cutout_size 5
        assert hdul["PIXELS"].columns.names[4] == "FLUX"  # column 5, as in SPOC files
        assert hdul[0].header["CREATOR"] == "tess_sso targetpixel" and hdul[0].header["PIXBIT3"] == "APERTURE"
        np.testing.assert_array_equal(hdul["EXTRAS"].data["CORNER1"], lc["EXTRAS"].data["CORNER1"])
        assert hdul["PIXELS"].header["1CRV5P"] == hdul["EXTRAS"].data["CORNER1"][0]
        assert hdul["PIXELS"].header["2CRV5P"] == hdul["EXTRAS"].data["CORNER2"][0]


def test_images_add_up_and_masks_agree(written):
    _, path, _ = written
    with fits.open(path) as hdul:
        d, aperture = hdul["PIXELS"].data, hdul["APERTURE"].data
    used = d["COVERAGE"] > 0
    np.testing.assert_array_equal((d["PIXEL_MASK"] & PixelMask.USED) > 0, used)
    np.testing.assert_allclose((d["FLUX"] - d["STAR_MODEL"] - d["ASTEROID_MODEL"])[used], d["RESIDUAL"][used], atol=1e-3)
    assert np.all(np.isnan(d["FLUX"][~used]))
    assert np.all(d["FLUX_BKG"][used] > 0) and np.all(d["FLUX_BKG_ERR"][used] > 0)
    assert np.all(aperture & 1)
    majority = ((d["PIXEL_MASK"] & PixelMask.APERTURE) > 0).mean(axis=0) >= 0.5
    np.testing.assert_array_equal((aperture & 2) > 0, majority)


def test_lightkurve_reads_plots_and_animates(written):
    lk = pytest.importorskip("lightkurve")
    import matplotlib

    matplotlib.use("Agg")
    _, path, _ = written
    tpf = lk.read(str(path))
    assert type(tpf).__name__ == "TessTargetPixelFile" and tpf.shape == (3, 5, 5)
    tpf.plot(frame=1)
    tpf.animate()
    lc = tpf.to_lightcurve(aperture_mask="pipeline")
    assert len(lc) == 3


def test_needs_the_group_images():
    with pytest.raises(ValueError, match="no group images"):
        pixel_hdulist(Photometry(), None, None, "95", "95 Arethusa", {})
