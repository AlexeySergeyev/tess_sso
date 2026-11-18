import numpy as np
import pandas as pd
import pytest
from astropy.io import fits
from astropy.table import Table

from tess_sso.storage.ecsv_io import read_legacy_fits, read_table, write_table
from tess_sso.storage.fits_io import CorruptFileError, write_atomic


def sample():
    return Table({
        "time": np.array([2460259.500800725123, np.nan, 1 / 3]),
        "source_id": np.array([4295806720, -1, 2**62], dtype=np.int64),
        "flux": np.array([1.1, np.nan, 3.0], dtype=np.float32),
        "sector": np.array([72, 92, 92], dtype=np.int32),
    })


def test_round_trip_is_exact_with_types_and_metadata(tmp_path):
    path = tmp_path / "t.ecsv"
    write_table(path, sample(), {"DESIG": "3485", "GMAX": 17.5, "STATCORR": True})
    table = read_table(path)
    for name, column in sample().items():
        assert table[name].dtype == column.dtype
        assert np.array_equal(table[name], column, equal_nan=True)
    assert table.meta == {"DESIG": "3485", "GMAX": 17.5, "STATCORR": True}
    assert not (tmp_path / "t.ecsv.part").exists()


def test_pandas_reads_it_as_csv(tmp_path):
    path = tmp_path / "t.ecsv"
    write_table(path, sample())
    df = pd.read_csv(path, comment="#")
    assert list(df.columns) == ["time", "source_id", "flux", "sector"]
    assert df["source_id"].tolist() == [4295806720, -1, 2**62]


def test_an_empty_table_keeps_its_columns(tmp_path):
    path = tmp_path / "t.ecsv"
    write_table(path, Table(names=["sector", "camera"], dtype=[np.int32, np.int32]))
    table = read_table(path)
    assert table.colnames == ["sector", "camera"] and len(table) == 0 and table["sector"].dtype == np.int32


def test_a_truncated_file_is_reported(tmp_path):
    path = tmp_path / "t.ecsv"
    write_table(path, sample())
    path.write_text(path.read_text()[:60])
    with pytest.raises(CorruptFileError):
        read_table(path)


def test_a_legacy_fits_cache_is_read_with_its_cards(tmp_path):
    hdu = fits.table_to_hdu(sample())
    primary = fits.PrimaryHDU()
    primary.header["QUERYDAT"] = "2026-09-24T21:20:00+00:00"
    write_atomic(tmp_path / "t.fits", fits.HDUList([primary, hdu]))
    table = read_legacy_fits(tmp_path / "t.ecsv")
    assert table.meta == {"QUERYDAT": "2026-09-24T21:20:00+00:00"}
    assert np.array_equal(table["time"], sample()["time"], equal_nan=True)
    write_table(tmp_path / "t.ecsv", table)  # native byte order: writable as ECSV
    assert read_legacy_fits(tmp_path / "none.ecsv") is None
