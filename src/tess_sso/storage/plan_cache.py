"""Persist the most recent pixel plan for a visit, alongside its frame table."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import BadZipFile

import numpy as np

from tess_sso.geometry import Track, track_pixels
from tess_sso.storage.store import PixelStore


def _inventory(store: PixelStore) -> str:
    """Identify the complete FITS blocks without opening their pixel tables."""
    files = []
    for path in store.files():
        try:
            stat = path.stat()
        except FileNotFoundError:  # a block was removed during the directory scan
            continue
        files.append((path.name, stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns))
    return json.dumps([str(store.pixels_dir.resolve()), files])


def _pairs(array: np.ndarray) -> list[tuple[int, int]]:
    if array.shape == (0,):
        return []
    if array.ndim != 2 or array.shape[1] != 2 or array.dtype.kind != "i":
        raise ValueError("Invalid pixel coordinate array in cached plan")
    return list(map(tuple, array.tolist()))


def plan_pixels(track: Track, cutout_size: int, strip_width: int, store: PixelStore):
    """Reuse the strip geometry and its coverage while their respective inputs match."""
    # box_pixels rounds to the nearest pixel. Only on-science centres contribute.
    centres = np.column_stack((track.row[track.on_science], track.col[track.on_science]))
    geometry = hashlib.sha256(np.floor(centres + 0.5).astype("<i4").tobytes()).hexdigest()
    geometry = f"v1:{cutout_size}:{strip_width}:{geometry}"
    inventory = _inventory(store)
    path = store.directory / "pixel_plan.npz"
    strip = None
    try:
        with np.load(path, allow_pickle=False) as cache:
            if cache["geometry"].item() == geometry:
                cutout = _pairs(cache["cutout"])
                strip = _pairs(cache["strip"])
                if cache["inventory"].item() == inventory:
                    return cutout, strip, _pairs(cache["missing"])
    except (OSError, EOFError, TypeError, ValueError, KeyError, BadZipFile):
        strip = None

    if strip is None:
        strip = track_pixels(track, strip_width)
        cutout = track_pixels(track, cutout_size)
    missing = store.missing(strip)

    # A concurrent download can change coverage while the pixel tables are read.
    if _inventory(store) == inventory:
        try:
            with TemporaryDirectory(dir=store.directory) as tmp:
                pending = Path(tmp) / path.name
                np.savez(
                    pending,
                    geometry=geometry,
                    inventory=inventory,
                    cutout=np.asarray(cutout, dtype=np.int32).reshape(-1, 2),
                    strip=np.asarray(strip, dtype=np.int32).reshape(-1, 2),
                    missing=np.asarray(missing, dtype=np.int32).reshape(-1, 2),
                )
                pending.replace(path)
        except OSError:
            pass  # A read-only data directory still supports planning.
    return cutout, strip, missing
