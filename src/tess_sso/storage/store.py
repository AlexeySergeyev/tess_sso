"""Pixel storage of one observation: FITS files with exactly the requested pixels."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from tess_sso import __version__
from tess_sso.storage.blocks import (
    BLOCK_ROWS,
    CCD_NCOLUMNS,
    CCD_NROWS,
    by_block,
    check_pixels,
    file_name,
    parse_file_name,
)
from tess_sso.storage.fits_io import read_pixel_file, read_pixel_list, write_pixel_file
from tess_sso.storage.frames import FrameTable, read_frames


class MissingPixelsError(LookupError):
    """Requested pixels are not stored; ``missing`` lists them."""

    def __init__(self, missing: Sequence[tuple[int, int]]):
        self.missing = list(missing)
        super().__init__(f"{len(self.missing)} pixel(s) not downloaded, e.g. {self.missing[:3]}")


class PixelStore:
    """Frame table and pixel files of one observation, in ``directory``."""

    def __init__(
        self,
        directory: Path,
        sector: int,
        camera: int,
        ccd: int,
        block_rows: int = BLOCK_ROWS,
        ccd_shape: tuple[int, int] = (CCD_NROWS, CCD_NCOLUMNS),
    ):
        self.directory = Path(directory)
        self.sector, self.camera, self.ccd = sector, camera, ccd
        self.block_rows = block_rows
        self.ccd_shape = ccd_shape
        self.pixels_dir = self.directory / "pixels"
        self.frames_path = self.directory / "frames.fits"
        self._coverage: dict[tuple[int, int], tuple[Path, int]] | None = None

    def files(self) -> list[Path]:
        """Complete pixel files; unfinished ``.part`` files are ignored."""
        if not self.pixels_dir.exists():
            return []
        return sorted(p for p in self.pixels_dir.iterdir() if parse_file_name(p.name))

    def coverage(self) -> dict[tuple[int, int], tuple[Path, int]]:
        """Every stored pixel -> (file, column index in that file)."""
        if self._coverage is None:
            index: dict[tuple[int, int], tuple[Path, int]] = {}
            for path in self.files():
                for i, (row, col) in enumerate(read_pixel_list(path)):
                    index.setdefault((int(row), int(col)), (path, i))
            self._coverage = index
        return self._coverage

    def n_pixels(self) -> int:
        return len(self.coverage())

    def missing(self, pixels: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
        pixels = sorted(set((int(r), int(c)) for r, c in pixels))
        check_pixels(pixels, self.ccd_shape)
        stored = self.coverage()
        return [p for p in pixels if p not in stored]

    def _next_path(self, start: int) -> Path:
        parts = [parse_file_name(p.name)[1] for p in self.files() if parse_file_name(p.name)[0] == start]
        return self.pixels_dir / file_name(start, max(parts, default=0) + 1, self.block_rows)

    def write_block(
        self, pixels: Sequence[tuple[int, int]], flux: np.ndarray, flux_err: np.ndarray
    ) -> Path:
        """Store new pixels of one row block in a new file; existing files are never touched."""
        starts = {(row - 1) // self.block_rows * self.block_rows + 1 for row, _ in pixels}
        if len(starts) != 1:
            raise ValueError(f"pixels span {len(starts)} row blocks; write one block at a time")
        start = starts.pop()
        path = self._next_path(start)
        cards = {
            "ROWFIRST": start,
            "ROWLAST": start + self.block_rows - 1,
            "SECTOR": self.sector,
            "CAMERA": self.camera,
            "CCD": self.ccd,
            "TESS_SSO": __version__,
        }
        write_pixel_file(path, np.asarray(pixels), flux, flux_err, cards)
        self._coverage = None
        return path

    def has_frames(self) -> bool:
        return self.frames_path.exists()

    def frames(self) -> FrameTable:
        return read_frames(self.frames_path)

    def read_pixels(
        self, pixels: Sequence[tuple[int, int]], frames: slice | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Flux and error of ``pixels``, each shaped (nframes, npixels), in the given order.

        ``frames`` reads only a contiguous range of frames.
        """
        pixels = [(int(r), int(c)) for r, c in pixels]
        missing = self.missing(pixels)
        if missing:
            raise MissingPixelsError(missing)
        stored = self.coverage()
        wanted: dict[Path, list[tuple[int, int]]] = {}
        for i, p in enumerate(pixels):
            path, j = stored[p]
            wanted.setdefault(path, []).append((i, j))
        flux = err = None
        for path, pairs in wanted.items():
            data = read_pixel_file(path, frames)
            if flux is None:
                flux = np.full((data.flux.shape[0], len(pixels)), np.nan, dtype=np.float32)
                err = np.full_like(flux, np.nan)
            elif data.flux.shape[0] != flux.shape[0]:
                raise ValueError(f"{path.name} has {data.flux.shape[0]} frames; other files have {flux.shape[0]}")
            out_idx = [i for i, _ in pairs]
            in_idx = [j for _, j in pairs]
            flux[:, out_idx] = data.flux[:, in_idx]
            err[:, out_idx] = data.flux_err[:, in_idx]
        if flux is None:
            return np.empty((0, 0), np.float32), np.empty((0, 0), np.float32)
        return flux, err

    def blocks_of(self, pixels: Iterable[tuple[int, int]]) -> dict[int, list[tuple[int, int]]]:
        return by_block(pixels, self.block_rows)
