"""Download pixels from the MAST cloud cubes through tesscube into a PixelStore."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol, Sequence

import numpy as np
from astropy.io import fits

from tess_sso import __version__
from tess_sso._compat import patch_lkspacecraft_listing, take_listing_fallback
from tess_sso.network import ATTEMPTS, RETRY_WAITS_S, socket_timeout, with_retries
from tess_sso.storage.blocks import runs_for
from tess_sso.storage.frames import FrameTable, write_frames
from tess_sso.storage.store import PixelStore

# One array per run, shaped (ncolumns, nframes, 2): flux and flux error.
Fetch = Callable[[list[dict[str, int]]], Sequence[np.ndarray]]

BLOCK_TIMEOUT_S = 900.0  # one block of rows, all its requests together; then it is tried again


class PixelSource(Protocol):
    sector: int
    camera: int
    ccd: int

    def fetch(self, runs: list[dict[str, int]]) -> Sequence[np.ndarray]: ...

    def frames_hdu(self) -> fits.BinTableHDU: ...

    def cadenceno(self) -> np.ndarray: ...

    def provenance(self) -> dict: ...


@dataclass
class DownloadReport:
    files: list[Path] = field(default_factory=list)  # new pixel files
    downloaded: int = 0  # pixels fetched
    already_stored: int = 0  # requested pixels that were already stored


def download_pixels(
    store: PixelStore,
    pixels: Sequence[tuple[int, int]],
    fetch: Fetch,
    attempts: int = ATTEMPTS,
    waits: Sequence[float] = RETRY_WAITS_S,
) -> DownloadReport:
    """Fetch the requested pixels that are not stored yet, one row block per file.

    Each block is written as soon as it arrives, so an interrupted run
    resumes with the first block that is still missing. A block whose fetch
    fails on the network (timeout, reset, truncated response) is tried again,
    up to ``attempts`` times in all, after the pauses in ``waits``.
    """
    requested = set((int(r), int(c)) for r, c in pixels)
    missing = store.missing(requested)
    report = DownloadReport(already_stored=len(requested) - len(missing))
    for start, block in store.blocks_of(missing).items():
        runs = runs_for(block)
        what = f"{store.directory.name}: rows {start}-{start + store.block_rows - 1}"
        arrays = list(with_retries(lambda: fetch(runs), what, attempts, waits))
        if len(arrays) != len(runs):
            raise RuntimeError(f"fetch returned {len(arrays)} rows for {len(runs)} runs")
        order, flux_parts, err_parts = [], [], []
        for run, values in zip(runs, arrays):
            if values.shape[0] != run["ncolumns"] or values.shape[2] != 2:
                raise RuntimeError(f"unexpected shape {values.shape} for run {run}")
            order += [(run["row"], run["start_column"] + k) for k in range(run["ncolumns"])]
            flux_parts.append(values[..., 0].T)  # (nframes, ncolumns)
            err_parts.append(values[..., 1].T)
        report.files.append(store.write_block(order, np.hstack(flux_parts), np.hstack(err_parts)))
        report.downloaded += len(order)
    return report


def ensure_frames(store: PixelStore, source: PixelSource) -> FrameTable:
    """Write ``frames.fits`` on first use; an existing file is used as is, without asking ``source``."""
    if not store.has_frames():
        write_frames(store.frames_path, source.frames_hdu(), source.cadenceno(), source.provenance())
    return store.frames()


def fetch_pixels(
    pixels: Sequence[tuple[int, int]], source: PixelSource, store: PixelStore
) -> tuple[FrameTable, np.ndarray, np.ndarray]:
    """Frame table, flux, and error of ``pixels``, downloading missing ones first."""
    frames = ensure_frames(store, source)
    download_pixels(store, pixels, source.fetch)
    flux, flux_err = store.read_pixels(pixels)
    if flux.shape[0] != len(frames):
        raise ValueError(f"pixel files have {flux.shape[0]} frames; frames.fits has {len(frames)}")
    return frames, flux, flux_err


class TessCubeSource:
    """Pixels, frame table, and provenance of one MAST cloud cube, via tesscube."""

    def __init__(self, sector: int, camera: int, ccd: int):
        self.sector, self.camera, self.ccd = sector, camera, ccd
        self.cube = with_retries(self._open, f"s{sector:04d}-{camera}-{ccd}: opening the MAST cube")
        self._head: dict | None = None

    def _open(self):
        # tesscube builds lkspacecraft's meta kernel on import, reading MAST's
        # listing with no timeout of its own; opening a cube reads its headers from S3
        with socket_timeout():
            patch_lkspacecraft_listing()
            from tesscube import TESSCube

            if message := take_listing_fallback():
                warnings.warn(message, stacklevel=2)
            return TESSCube(sector=self.sector, camera=self.camera, ccd=self.ccd)

    def fetch(self, runs: list[dict[str, int]]) -> Sequence[np.ndarray]:
        """One block's runs; one that takes longer than ``BLOCK_TIMEOUT_S`` raises TimeoutError."""
        import asyncio

        # tesscube's own sync wrapper; it is private, so pin tesscube's version range.
        from tesscube.utils import _sync_call

        return _sync_call(
            lambda: asyncio.wait_for(self.cube.async_get_data_per_rows(runs=runs), BLOCK_TIMEOUT_S)
        )

    def _head_object(self) -> dict:
        if self._head is None:
            import boto3
            from botocore import UNSIGNED
            from botocore.config import Config
            from tesscube import BUCKET_NAME

            s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
            self._head = with_retries(
                lambda: s3.head_object(Bucket=BUCKET_NAME, Key=self.cube.object_key),
                f"{self.cube.object_key}: S3 metadata",
            )
        return self._head

    def frames_hdu(self) -> fits.BinTableHDU:
        return self.cube.last_hdu

    def cadenceno(self) -> np.ndarray:
        return np.asarray(self.cube.cadence_number)

    def provenance(self) -> dict:
        from importlib.metadata import version

        head = self._head_object()
        header = self.cube.primary_hdu.header
        prov = {
            "OBJKEY": self.cube.object_key,
            "LASTMOD": head["LastModified"].isoformat(),
            "SECTOR": self.sector,
            "CAMERA": self.camera,
            "CCD": self.ccd,
            "CCDROWS": self.cube.nrows,
            "CCDCOLS": self.cube.ncolumns,
            "TESSCUBE": version("tesscube"),
            "TESS_SSO": __version__,
        }
        if "DATA_REL" in header:
            prov["DATA_REL"] = header["DATA_REL"]
        if "DATE" in header:
            prov["CUBEDATE"] = header["DATE"]
        return prov
