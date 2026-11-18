"""Instrumental TESS PRF from lkprf, for field stars and the moving asteroid.

Coordinates are 1-based (row, column) with pixel k centred at k.0, the same
convention lkprf uses for ``targets`` and ``origin``. The model for the
right sector is always requested (lkprf has one set for sectors 1-3 and one
for sectors 4+), and its absolute normalization is kept: a PRF summed over
the whole plane is 1, so a fitted amplitude is the total flux.

lkprf downloads its model files on first use. They go to
``<data root>/reference/lkprf/``, never into the installed package.
"""

from __future__ import annotations

import os
from importlib.metadata import version
from pathlib import Path

import numpy as np

from tess_sso.paths import reference_dir

# Motion within an exposure is sampled at this step (pixels).
SMEAR_STEP_PX = 0.1


def prf_cache_dir(root: str | os.PathLike | None = None) -> str:
    """lkprf model directory in ``reference/`` (lkprf wants a trailing slash)."""
    path = reference_dir(root) / "lkprf"
    path.mkdir(parents=True, exist_ok=True)
    return str(path) + os.sep


# lkprf's model file for one CCD holds the PRFs of a 5 x 5 grid of positions, one HDU each.
MODEL_HDUS = 25


def model_path(cache_dir: str, sector: int, camera: int, ccd: int) -> Path:
    """The file lkprf builds for this CCD and model set in ``cache_dir``."""
    return Path(cache_dir) / f"tess-prf-cam{camera}-ccd{ccd}-sec{1 if sector <= 3 else 4}.fits"


def _complete(path: Path) -> bool:
    """Whether the model file at ``path`` holds the whole grid."""
    from astropy.io import fits

    try:
        with fits.open(path) as hdul:
            return len(hdul) == MODEL_HDUS
    except Exception:  # empty, truncated, or not FITS
        return False


def _remove_unfinished(path: Path) -> bool:
    """Remove a model file lkprf left unfinished; True if one was removed."""
    if path.is_file() and not _complete(path):
        path.unlink()
        return True
    return False


class Prf:
    def __init__(self, sector: int, camera: int, ccd: int, cache_dir: str | None = None):
        import lkprf  # deferred: loads model files

        self.sector, self.camera, self.ccd = sector, camera, ccd
        cache_dir = cache_dir or prf_cache_dir()
        # lkprf creates its model file before downloading the PRFs into it and
        # reads any file that exists, so a failed download would poison the cache.
        path = model_path(cache_dir, sector, camera, ccd)
        _remove_unfinished(path)
        try:
            self.model = lkprf.TESSPRF(camera=camera, ccd=ccd, sector=sector, cache_dir=cache_dir)
        except Exception as exc:
            if _remove_unfinished(path):
                exc.add_note(f"lkprf could not build {path.name}; the unfinished file was removed")
            raise
        self.lkprf_version = version("lkprf")
        self.model_set = "sectors 1-3" if sector <= 3 else "sectors 4+"

    def evaluate(
        self, rows: np.ndarray, cols: np.ndarray, origin: tuple[int, int], shape: tuple[int, int]
    ) -> np.ndarray:
        """PRF of each target on the image ``origin`` (1-based row, col of [0, 0]) x ``shape``."""
        targets = [(float(r), float(c)) for r, c in zip(np.atleast_1d(rows), np.atleast_1d(cols))]
        return np.asarray(self.model.evaluate(targets=targets, origin=origin, shape=shape))

    def smeared(
        self, row0: float, col0: float, row1: float, col1: float, origin: tuple[int, int], shape: tuple[int, int]
    ) -> np.ndarray:
        """PRF of a source moving uniformly from (row0, col0) to (row1, col1) during the exposure."""
        length = float(np.hypot(row1 - row0, col1 - col0))
        n = max(1, int(np.ceil(length / SMEAR_STEP_PX)))
        if n == 1:
            rows, cols = np.array([(row0 + row1) / 2]), np.array([(col0 + col1) / 2])
        else:
            frac = (np.arange(n) + 0.5) / n
            rows, cols = row0 + (row1 - row0) * frac, col0 + (col1 - col0) * frac
        return self.evaluate(rows, cols, origin, shape).mean(axis=0)


def window_origin(row: float, col: float, size: int) -> tuple[int, int]:
    """Origin of a ``size`` x ``size`` window centred on the pixel containing (row, col)."""
    return int(np.floor(row + 0.5)) - size // 2, int(np.floor(col + 0.5)) - size // 2


def render_stars(
    prf: Prf,
    rows: np.ndarray,
    cols: np.ndarray,
    fluxes: np.ndarray,
    origin: tuple[int, int],
    shape: tuple[int, int],
    window: int = 15,
) -> np.ndarray:
    """Predicted stellar flux image: each star's PRF over a ``window`` box, added into the image."""
    image = np.zeros(shape, dtype=np.float64)
    r0, c0 = origin
    for row, col, flux in zip(rows, cols, fluxes):
        if not (np.isfinite(row) and np.isfinite(col) and np.isfinite(flux)):
            continue
        wr, wc = window_origin(row, col, window)
        # overlap of the window with the image, in image indices
        i0, j0 = max(wr - r0, 0), max(wc - c0, 0)
        i1, j1 = min(wr - r0 + window, shape[0]), min(wc - c0 + window, shape[1])
        if i0 >= i1 or j0 >= j1:
            continue
        stamp = prf.evaluate(np.array([row]), np.array([col]), (wr, wc), (window, window))[0]
        image[i0:i1, j0:j1] += flux * stamp[i0 - (wr - r0) : i1 - (wr - r0), j0 - (wc - c0) : j1 - (wc - c0)]
    return image
