import numpy as np
import pytest

from conftest import make_cube_frames
from tess_sso.storage.frames import read_frames, write_frames


def test_roundtrip(tmp_path):
    path = tmp_path / "frames.fits"
    write_frames(path, make_cube_frames(4), np.arange(10, 14), {"OBJKEY": "k", "SECTOR": 71})
    frames = read_frames(path)
    assert len(frames) == 4
    np.testing.assert_array_equal(frames.cadenceno, [10, 11, 12, 13])
    np.testing.assert_allclose(frames.time, frames.tstart + 100 / 86400)
    assert frames.quality.dtype == np.int64
    assert frames.provenance == {"OBJKEY": "k", "SECTOR": 71}
    assert "FFI_FILE" in frames.table.colnames


def test_cadence_count_must_match(tmp_path):
    with pytest.raises(ValueError, match="cadence numbers"):
        write_frames(tmp_path / "frames.fits", make_cube_frames(4), np.arange(3), {})


def test_unknown_provenance_key_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown provenance"):
        write_frames(tmp_path / "frames.fits", make_cube_frames(2), np.arange(2), {"FOO": 1})


def test_cube_date_is_a_provenance_key(tmp_path):
    path = tmp_path / "frames.fits"
    write_frames(path, make_cube_frames(2), np.arange(2), {"CUBEDATE": "2023-12-10", "DATA_REL": 100})
    assert read_frames(path).provenance == {"DATA_REL": 100, "CUBEDATE": "2023-12-10"}
