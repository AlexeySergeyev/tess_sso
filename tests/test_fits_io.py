import numpy as np
import pytest
from astropy.io import fits

from tess_sso.storage.fits_io import read_pixel_file, read_pixel_list, write_pixel_file

PIXELS = np.array([[705, 10], [705, 11], [712, 40]])


def _arrays(nframes=5):
    rng = np.random.default_rng(0)
    flux = rng.normal(100, 5, (nframes, len(PIXELS))).astype(np.float32)
    return flux, np.full_like(flux, 2.5)


def test_roundtrip(tmp_path):
    flux, err = _arrays()
    path = tmp_path / "r0705-0720_1.fits"
    write_pixel_file(path, PIXELS, flux, err, {"SECTOR": 71})
    data = read_pixel_file(path)
    np.testing.assert_array_equal(data.pixels, PIXELS)
    assert data.flux.dtype == np.float32
    np.testing.assert_array_equal(data.flux, flux)
    np.testing.assert_array_equal(data.flux_err, err)
    assert data.header["SECTOR"] == 71 and data.header["NPIX"] == 3 and data.header["NFRAMES"] == 5
    np.testing.assert_array_equal(read_pixel_list(path), PIXELS)
    assert not (tmp_path / "r0705-0720_1.fits.part").exists()


def test_wrong_shape_is_rejected_and_nothing_written(tmp_path):
    flux, err = _arrays()
    path = tmp_path / "x.fits"
    with pytest.raises(ValueError, match="expected"):
        write_pixel_file(path, PIXELS, flux[:, :2], err[:, :2], {})
    assert not path.exists()


def test_no_checksums_are_written(tmp_path):
    flux, err = _arrays()
    path = tmp_path / "x.fits"
    write_pixel_file(path, PIXELS, flux, err, {})
    with fits.open(path) as hdul:
        assert not any("CHECKSUM" in hdu.header or "DATASUM" in hdu.header for hdu in hdul)
