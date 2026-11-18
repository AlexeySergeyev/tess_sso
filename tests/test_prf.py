"""Real lkprf checks.

Offline tests only read model files already present in the lkprf package directory and
skip otherwise. The network test downloads its models into ``tmp_path``.
"""

from pathlib import Path

import numpy as np
import pytest

from tess_sso.prf import Prf, window_origin


def lkprf_dir_with(*files):
    lkprf = pytest.importorskip("lkprf")
    folder = Path(lkprf.__file__).parent / "data"
    if not all((folder / f).exists() for f in files):
        pytest.skip(f"lkprf model files {files} not present")
    return str(folder) + "/"


def test_absolute_normalization_and_centre():
    prf = Prf(71, 2, 2, cache_dir=lkprf_dir_with("tess-prf-cam2-ccd2-sec4.fits"))
    big = prf.evaluate(np.array([1000.0]), np.array([500.0]), (990, 490), (21, 21))[0]
    small = prf.evaluate(np.array([1000.0]), np.array([500.0]), (999, 499), (3, 3))[0]
    assert big.sum() == pytest.approx(1.0, abs=1e-3)
    assert 0.7 < small.sum() < 0.9
    assert np.unravel_index(big.argmax(), big.shape) == (10, 10)
    assert prf.model_set == "sectors 4+"


@pytest.mark.network
def test_sector_1_to_3_model_differs(tmp_path):
    """Downloads both lkprf model sets for camera 2, CCD 2 into tmp_path."""
    folder = str(tmp_path) + "/"
    early = Prf(2, 2, 2, cache_dir=folder)
    late = Prf(71, 2, 2, cache_dir=folder)
    a = late.evaluate(np.array([1000.0]), np.array([500.0]), (990, 490), (21, 21))
    b = early.evaluate(np.array([1000.0]), np.array([500.0]), (990, 490), (21, 21))
    assert early.model_set == "sectors 1-3" and late.model_set == "sectors 4+"
    assert (tmp_path / "tess-prf-cam2-ccd2-sec1.fits").exists()
    assert np.abs(a - b).max() > 1e-3


def test_smear_spreads_flux_but_keeps_total():
    prf = Prf(71, 2, 2, cache_dir=lkprf_dir_with("tess-prf-cam2-ccd2-sec4.fits"))
    still = prf.smeared(1000.0, 500.0, 1000.0, 500.0, (990, 490), (21, 21))
    moving = prf.smeared(1000.0, 497.0, 1000.0, 503.0, (990, 490), (21, 21))
    assert moving.sum() == pytest.approx(still.sum(), abs=1e-3)
    assert moving.max() < still.max()


def test_window_origin():
    assert window_origin(100.4, 200.6, 11) == (95, 196)


def fake_lkprf(monkeypatch, build):
    """lkprf.TESSPRF replaced by ``build(path)``, which acts on the model file."""
    import lkprf

    from tess_sso.prf import model_path

    def tessprf(camera, ccd, sector, cache_dir):
        build(model_path(cache_dir, sector, camera, ccd))
        return object()

    monkeypatch.setattr(lkprf, "TESSPRF", tessprf)


def write_model(path, nhdus):
    from astropy.io import fits

    fits.HDUList([fits.PrimaryHDU(np.zeros((2, 2)))] + [fits.ImageHDU(np.zeros((2, 2))) for _ in range(nhdus - 1)]).writeto(path)


def test_a_failed_build_removes_the_file_it_left(monkeypatch, tmp_path):
    def build(path):
        path.touch()  # lkprf opens the file, then the first download fails
        raise OSError("403 Client Error: Forbidden")

    fake_lkprf(monkeypatch, build)
    with pytest.raises(OSError, match="403") as info:
        Prf(13, 1, 3, cache_dir=str(tmp_path) + "/")
    assert not list(tmp_path.iterdir())
    assert "unfinished file was removed" in "".join(info.value.__notes__)


@pytest.mark.parametrize("nhdus", [0, 10])
def test_an_unfinished_cached_model_is_built_again(monkeypatch, tmp_path, nhdus):
    from tess_sso.prf import MODEL_HDUS, model_path

    path = model_path(str(tmp_path), 13, 1, 3)
    path.touch() if nhdus == 0 else write_model(path, nhdus)
    seen = []

    def build(path):
        seen.append(path.exists())
        write_model(path, MODEL_HDUS)

    fake_lkprf(monkeypatch, build)
    Prf(13, 1, 3, cache_dir=str(tmp_path) + "/")
    assert seen == [False]  # lkprf found no file, so it built one


def test_a_complete_cached_model_is_kept(monkeypatch, tmp_path):
    from tess_sso.prf import MODEL_HDUS, model_path

    path = model_path(str(tmp_path), 13, 1, 3)
    write_model(path, MODEL_HDUS)
    before = path.stat().st_mtime_ns
    fake_lkprf(monkeypatch, lambda path: None)
    Prf(13, 1, 3, cache_dir=str(tmp_path) + "/")
    assert path.stat().st_mtime_ns == before
