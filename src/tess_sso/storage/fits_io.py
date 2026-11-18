"""Atomic FITS writing and reading for stored files."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from astropy.io import fits


class CorruptFileError(OSError):
    """A stored file cannot be read."""


def write_atomic(path: Path, hdul: fits.HDUList) -> None:
    """Write ``hdul`` to a temporary file, then rename it.

    A file at ``path`` is therefore always complete.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    hdul.writeto(tmp, overwrite=True)
    os.replace(tmp, path)


@dataclass(frozen=True)
class PixelFile:
    pixels: np.ndarray  # (npix, 2): 1-based (row, col)
    flux: np.ndarray  # (nframes, npix), float32
    flux_err: np.ndarray
    header: fits.Header


def write_pixel_file(
    path: Path,
    pixels: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    cards: Mapping[str, Any],
) -> None:
    """Write pixels (npix, 2) with ``flux`` and ``flux_err`` shaped (nframes, npix)."""
    pixels = np.asarray(pixels, dtype=np.int32).reshape(-1, 2)
    for name, array in (("flux", flux), ("flux_err", flux_err)):
        if array.ndim != 2 or array.shape[1] != len(pixels):
            raise ValueError(f"{name} has shape {array.shape}; expected (nframes, {len(pixels)})")
    if flux.shape != flux_err.shape:
        raise ValueError(f"flux {flux.shape} and flux_err {flux_err.shape} differ")
    primary = fits.PrimaryHDU()
    primary.header["NPIX"] = (len(pixels), "number of pixels in this file")
    primary.header["NFRAMES"] = (flux.shape[0], "number of frames")
    for key, value in cards.items():
        primary.header[key] = value
    table = fits.BinTableHDU.from_columns(
        [fits.Column(name="ROW", format="J", array=pixels[:, 0]), fits.Column(name="COLUMN", format="J", array=pixels[:, 1])],
        name="PIXELS",
    )
    hdul = fits.HDUList(
        [
            primary,
            table,
            fits.ImageHDU(np.asarray(flux, dtype=">f4"), name="FLUX"),
            fits.ImageHDU(np.asarray(flux_err, dtype=">f4"), name="FLUX_ERR"),
        ]
    )
    write_atomic(path, hdul)


def read_pixel_list(path: Path) -> np.ndarray:
    """Only the (row, col) list of a pixel file, without reading the fluxes."""
    with fits.open(path) as hdul:
        data = hdul["PIXELS"].data
        return np.column_stack([data["ROW"], data["COLUMN"]]).astype(np.int64)


def read_pixel_file(path: Path, frames: slice | None = None) -> PixelFile:
    """Read one pixel file; with ``frames``, only those rows of FLUX/FLUX_ERR (memory-mapped)."""
    with fits.open(path, memmap=True) as hdul:
        data = hdul["PIXELS"].data
        pixels = np.column_stack([data["ROW"], data["COLUMN"]]).astype(np.int64)
        rows = slice(None) if frames is None else frames
        return PixelFile(
            pixels=pixels,
            flux=np.array(hdul["FLUX"].data[rows], dtype=np.float32),
            flux_err=np.array(hdul["FLUX_ERR"].data[rows], dtype=np.float32),
            header=hdul[0].header.copy(),
        )
