"""Shared test helpers: a fake cloud cube with predictable pixel values."""

from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits


def pixel_value(row: int, col: int) -> float:
    """Flux of a fake pixel: exact in float32 for any CCD pixel."""
    return float(row * 1000 + col)


def make_cube_frames(nframes: int) -> fits.BinTableHDU:
    """A small stand-in for tesscube's per-frame table (last HDU)."""
    tstart = 3234.0 + np.arange(nframes) * (200 / 86400)
    return fits.BinTableHDU.from_columns(
        [
            fits.Column(name="TSTART", format="D", array=tstart),
            fits.Column(name="TSTOP", format="D", array=tstart + 200 / 86400),
            fits.Column(name="DQUALITY", format="J", array=np.zeros(nframes, dtype=np.int32)),
            fits.Column(name="BARYCORR", format="D", array=np.full(nframes, 0.003)),
            fits.Column(name="FFI_FILE", format="40A", array=[f"ffi-{i:05d}.fits" for i in range(nframes)]),
        ]
    )


class FakeSource:
    """Implements the PixelSource protocol without network access."""

    def __init__(self, sector=71, camera=2, ccd=2, nframes=5):
        self.sector, self.camera, self.ccd = sector, camera, ccd
        self.nframes = nframes
        self.requested_runs: list[dict[str, int]] = []

    def fetch(self, runs):
        self.requested_runs.extend(runs)
        out = []
        for run in runs:
            cols = np.arange(run["start_column"], run["start_column"] + run["ncolumns"])
            values = np.zeros((run["ncolumns"], self.nframes, 2), dtype=np.float32)
            values[..., 0] = np.array([pixel_value(run["row"], c) for c in cols])[:, None]
            values[..., 1] = np.arange(self.nframes)[None, :]  # error = frame index
            out.append(values)
        return out

    def frames_hdu(self):
        return make_cube_frames(self.nframes)

    def cadenceno(self):
        return np.arange(100, 100 + self.nframes)

    def provenance(self):
        return {"OBJKEY": "tess/public/mast/tess-s0071-2-2-cube.fits",
                "SECTOR": self.sector, "CAMERA": self.camera, "CCD": self.ccd}


@pytest.fixture(autouse=True)
def no_retry_pauses(monkeypatch):
    """Network retries in tests do not wait."""
    monkeypatch.setattr("tess_sso.network._sleep", lambda seconds: None)


@pytest.fixture
def fake_source():
    return FakeSource()


def make_wcs_frames(nframes: int, start_btjd: float = 3234.0, cadence_s: float = 200.0,
                    sector: int = 71, camera: int = 2, ccd: int = 2,
                    crval_shift_deg: float = 0.0, nan_frames: tuple[int, ...] = ()) -> fits.BinTableHDU:
    """Frame table whose per-frame WCS columns come from the tesswcs model of one CCD."""
    from tesswcs import WCS

    wcs = WCS.from_sector(sector, camera, ccd)
    tstart = start_btjd + np.arange(nframes) * cadence_s / 86400
    cols = {
        "TSTART": tstart,
        "TSTOP": tstart + cadence_s / 86400,
        "DQUALITY": np.zeros(nframes, dtype=np.int32),
        "BARYCORR": np.full(nframes, 0.003),
        "CRVAL1": np.full(nframes, wcs.wcs.crval[0] + crval_shift_deg),
        "CRVAL2": np.full(nframes, wcs.wcs.crval[1]),
        "CRPIX1": np.full(nframes, wcs.wcs.crpix[0]),
        "CRPIX2": np.full(nframes, wcs.wcs.crpix[1]),
    }
    cd = wcs.pixel_scale_matrix
    for (i, j), value in np.ndenumerate(cd):
        cols[f"CD{i + 1}_{j + 1}"] = np.full(nframes, value)
    for name, matrix in (("A", wcs.sip.a), ("B", wcs.sip.b)):
        cols[f"{name}_ORDER"] = np.full(nframes, wcs.sip.a_order, dtype=np.int32)
        for (p, q), value in np.ndenumerate(matrix):
            if p + q >= 2 and p + q <= wcs.sip.a_order:
                cols[f"{name}_{p}_{q}"] = np.full(nframes, value)
    for name in ("CRVAL1", "CRVAL2", "CD1_1", "CD1_2", "CD2_1", "CD2_2"):
        cols[name][list(nan_frames)] = np.nan
    columns = [fits.Column(name=k, format="J" if v.dtype.kind == "i" else "D", array=v) for k, v in cols.items()]
    columns.append(fits.Column(name="CTYPE1", format="12A", array=["RA---TAN-SIP"] * nframes))
    columns.append(fits.Column(name="CTYPE2", format="12A", array=["DEC--TAN-SIP"] * nframes))
    columns.append(fits.Column(name="FFI_FILE", format="40A", array=[f"ffi-{i:05d}.fits" for i in range(nframes)]))
    return fits.BinTableHDU.from_columns(columns)


class WcsFakeSource(FakeSource):
    """FakeSource whose frame table carries a per-frame WCS."""

    def __init__(self, nframes=6, crval_shift_deg=0.0, **kwargs):
        super().__init__(nframes=nframes, **kwargs)
        self.crval_shift_deg = crval_shift_deg

    def frames_hdu(self):
        return make_wcs_frames(self.nframes, crval_shift_deg=self.crval_shift_deg)


class GaussianPrf:
    """Stand-in for tess_sso.prf.Prf: a pixel-integrated Gaussian with unit total flux."""

    lkprf_version = "gaussian-test"
    model_set = "test"

    def __init__(self, sector=71, camera=2, ccd=2, sigma=0.8):
        self.sector, self.camera, self.ccd, self.sigma = sector, camera, ccd, sigma

    def _integrated(self, centre, start, n):
        from scipy.special import erf

        edges = start - 0.5 + np.arange(n + 1)
        cdf = 0.5 * (1 + erf((edges - centre) / (np.sqrt(2) * self.sigma)))
        return np.diff(cdf)

    def evaluate(self, rows, cols, origin, shape):
        rows, cols = np.atleast_1d(rows), np.atleast_1d(cols)
        return np.array([
            np.outer(self._integrated(r, origin[0], shape[0]), self._integrated(c, origin[1], shape[1]))
            for r, c in zip(rows, cols)
        ])

    def smeared(self, row0, col0, row1, col1, origin, shape):
        from tess_sso.prf import Prf

        return Prf.smeared(self, row0, col0, row1, col1, origin, shape)
