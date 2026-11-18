"""Frame table of a stored observation (``frames.fits``).

The file keeps tesscube's per-frame table verbatim (HDU ``CUBE_FRAMES``,
including per-frame WCS keywords) with a ``CADENCENO`` column added, and the
provenance of the cube in the primary header.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from astropy.io import fits
from astropy.table import Table

from tess_sso.storage.fits_io import write_atomic

# Provenance keywords written to the primary header of frames.fits.
PROVENANCE_KEYS = (
    "OBJKEY",  # S3 object key of the cube
    "LASTMOD",  # S3 last-modified date of the cube
    "SECTOR",
    "CAMERA",
    "CCD",
    "CCDROWS",
    "CCDCOLS",
    "DATA_REL",  # SPOC data release, when the cube header has it
    "CUBEDATE",  # creation date of the MAST cube; cubes carry no SPOC pipeline version
    "TESSCUBE",  # tesscube version
    "TESS_SSO",  # tess_sso version
)


@dataclass(frozen=True)
class FrameTable:
    table: Table  # full cube frame table, including per-frame WCS columns
    provenance: dict[str, Any]

    def __len__(self) -> int:
        return len(self.table)

    @property
    def tstart(self) -> np.ndarray:
        return np.asarray(self.table["TSTART"], dtype=np.float64)

    @property
    def tstop(self) -> np.ndarray:
        return np.asarray(self.table["TSTOP"], dtype=np.float64)

    @property
    def time(self) -> np.ndarray:
        """Mid-exposure time in BTJD, as tesscube defines it."""
        return (self.tstart + self.tstop) / 2

    @property
    def quality(self) -> np.ndarray:
        return np.asarray(self.table["DQUALITY"], dtype=np.int64)

    @property
    def barycorr(self) -> np.ndarray:
        return np.asarray(self.table["BARYCORR"], dtype=np.float64)

    @property
    def cadenceno(self) -> np.ndarray:
        return np.asarray(self.table["CADENCENO"], dtype=np.int64)


def write_frames(
    path: Path,
    cube_frames: fits.BinTableHDU,
    cadenceno: np.ndarray,
    provenance: Mapping[str, Any],
) -> None:
    """Write ``frames.fits`` from tesscube's last HDU and cadence numbers."""
    if len(cadenceno) != len(cube_frames.data):
        raise ValueError(
            f"{len(cadenceno)} cadence numbers for {len(cube_frames.data)} frames"
        )
    unknown = set(provenance) - set(PROVENANCE_KEYS)
    if unknown:
        raise ValueError(f"unknown provenance keys: {sorted(unknown)}")
    table = Table(cube_frames.data)
    table["CADENCENO"] = np.asarray(cadenceno, dtype=np.int64)
    frames_hdu = fits.table_to_hdu(table)
    frames_hdu.name = "CUBE_FRAMES"
    primary = fits.PrimaryHDU()
    for key, value in provenance.items():
        primary.header[key] = value
    write_atomic(path, fits.HDUList([primary, frames_hdu]))


def read_frames(path: Path) -> FrameTable:
    with fits.open(path) as hdul:
        provenance = {k: hdul[0].header[k] for k in PROVENANCE_KEYS if k in hdul[0].header}
        table = Table(hdul["CUBE_FRAMES"].data)
    return FrameTable(table=table, provenance=provenance)
