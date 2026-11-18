import numpy as np
import pytest

from conftest import pixel_value
from tess_sso.storage.store import MissingPixelsError, PixelStore


def make_store(tmp_path):
    return PixelStore(tmp_path / "95" / "s0071-2-2", 71, 2, 2)


def write(store, pixels, nframes=3):
    flux = np.array([[pixel_value(r, c) for r, c in pixels]] * nframes, dtype=np.float32)
    err = np.repeat(np.arange(nframes, dtype=np.float32)[:, None], len(pixels), axis=1)
    return store.write_block(pixels, flux, err)


def test_layout(tmp_path):
    store = make_store(tmp_path)
    assert store.frames_path == tmp_path / "95" / "s0071-2-2" / "frames.fits"
    assert store.pixels_dir == tmp_path / "95" / "s0071-2-2" / "pixels"


def test_files_and_missing(tmp_path):
    store = make_store(tmp_path)
    assert store.files() == [] and store.n_pixels() == 0
    path = write(store, [(705, 10), (706, 10)])
    assert path.name == "r0705-0720_1.fits"
    (store.pixels_dir / "r0721-0736_1.fits.part").write_bytes(b"partial")
    assert store.files() == [path]
    assert store.missing([(705, 10), (706, 10), (721, 10)]) == [(721, 10)]


def test_new_pixels_of_a_block_go_to_a_new_file(tmp_path):
    store = make_store(tmp_path)
    first = write(store, [(705, 10)])
    before = first.read_bytes()
    second = write(store, [(705, 11)])
    assert second.name == "r0705-0720_2.fits"
    assert first.read_bytes() == before  # never rewritten
    assert store.n_pixels() == 2


def test_one_block_per_write(tmp_path):
    with pytest.raises(ValueError, match="row blocks"):
        write(make_store(tmp_path), [(705, 10), (721, 10)])


def test_read_pixels_in_requested_order_across_files(tmp_path):
    store = make_store(tmp_path)
    write(store, [(705, 10), (705, 12)])
    write(store, [(721, 3)])
    pixels = [(721, 3), (705, 12), (705, 10)]
    flux, err = store.read_pixels(pixels)
    assert flux.shape == (3, 3)
    for i, (row, col) in enumerate(pixels):
        assert np.all(flux[:, i] == pixel_value(row, col))
        np.testing.assert_array_equal(err[:, i], [0, 1, 2])


def test_read_pixels_reports_missing(tmp_path):
    with pytest.raises(MissingPixelsError) as excinfo:
        make_store(tmp_path).read_pixels([(500, 1030)])
    assert excinfo.value.missing == [(500, 1030)]


def test_read_only_some_frames(tmp_path):
    store = make_store(tmp_path)
    pixels = [(705, 10), (705, 11)]
    flux = np.arange(10 * 2, dtype=np.float32).reshape(10, 2)
    store.write_block(pixels, flux, flux + 0.5)
    part, part_err = store.read_pixels(pixels, frames=slice(3, 6))
    np.testing.assert_array_equal(part, flux[3:6])
    np.testing.assert_array_equal(part_err, flux[3:6] + 0.5)
