import numpy as np
import pytest

from conftest import FakeSource, pixel_value
from tess_sso.network import NetworkRetryWarning
from tess_sso.storage.download import download_pixels, ensure_frames, fetch_pixels
from tess_sso.storage.store import PixelStore


def make_store(tmp_path):
    return PixelStore(tmp_path / "95" / "s0071-2-2", 71, 2, 2)


def test_downloads_exactly_the_requested_pixels(tmp_path, fake_source):
    store = make_store(tmp_path)
    pixels = [(500, 1030), (500, 1031), (500, 1035), (2078, 2136)]
    report = download_pixels(store, pixels, fake_source.fetch)
    assert report.downloaded == 4 and report.already_stored == 0 and len(report.files) == 2
    assert sum(run["ncolumns"] for run in fake_source.requested_runs) == 4
    flux, err = store.read_pixels(pixels)
    assert flux.shape == (5, 4)
    for i, (row, col) in enumerate(pixels):
        assert np.all(flux[:, i] == pixel_value(row, col))
    np.testing.assert_array_equal(err[:, 0], np.arange(5))


def test_second_call_fetches_only_new_pixels(tmp_path, fake_source):
    store = make_store(tmp_path)
    download_pixels(store, [(500, 1030)], fake_source.fetch)
    fake_source.requested_runs.clear()
    report = download_pixels(store, [(500, 1030), (500, 1031)], fake_source.fetch)
    assert report.already_stored == 1 and report.downloaded == 1
    assert fake_source.requested_runs == [{"row": 500, "start_column": 1031, "ncolumns": 1}]


def test_a_dropped_block_is_tried_again(tmp_path, fake_source):
    store = make_store(tmp_path)
    calls = {"n": 0}

    def flaky(runs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise ConnectionResetError("connection reset by peer")
        return fake_source.fetch(runs)

    with pytest.warns(NetworkRetryWarning, match="rows 513-528"):
        report = download_pixels(store, [(500, 1030), (520, 1030)], flaky, waits=[0])
    assert calls["n"] == 3 and report.downloaded == 2
    assert store.missing([(500, 1030), (520, 1030)]) == []


def test_a_block_that_keeps_failing_leaves_earlier_blocks(tmp_path, fake_source):
    store = make_store(tmp_path)
    calls = {"n": 0}

    def down_after_first(runs):
        calls["n"] += 1
        if calls["n"] >= 2:
            raise ConnectionError("network down")
        return fake_source.fetch(runs)

    with pytest.warns(NetworkRetryWarning), pytest.raises(ConnectionError):
        download_pixels(store, [(500, 1030), (520, 1030)], down_after_first, attempts=3, waits=[0])
    assert calls["n"] == 4  # one block stored, then three tries of the second
    assert store.missing([(500, 1030), (520, 1030)]) == [(520, 1030)]


def test_a_non_network_error_is_not_retried(tmp_path):
    calls = {"n": 0}

    def broken(runs):
        calls["n"] += 1
        raise ValueError("bad shape")

    with pytest.raises(ValueError):
        download_pixels(make_store(tmp_path), [(500, 1030)], broken, waits=[0])
    assert calls["n"] == 1


class OfflineSource(FakeSource):
    """A source that fails if asked for anything."""

    def __getattribute__(self, name):
        if name in ("fetch", "frames_hdu", "cadenceno", "provenance"):
            raise AssertionError(f"source.{name} used although the files are stored")
        return super().__getattribute__(name)


def test_ensure_frames_writes_once_then_uses_the_file(tmp_path):
    store = make_store(tmp_path)
    frames = ensure_frames(store, FakeSource())
    assert len(frames) == 5
    again = ensure_frames(store, OfflineSource())
    np.testing.assert_array_equal(again.cadenceno, frames.cadenceno)


def test_fetch_pixels_end_to_end(tmp_path, fake_source):
    store = make_store(tmp_path)
    pixels = [(500, 1030), (501, 1031)]
    frames, flux, err = fetch_pixels(pixels, fake_source, store)
    assert len(frames) == 5 and flux.shape == (5, 2)
    assert np.all(flux[:, 1] == pixel_value(501, 1031))
    fake_source.requested_runs.clear()
    fetch_pixels(pixels, fake_source, store)
    assert fake_source.requested_runs == []  # served from storage
