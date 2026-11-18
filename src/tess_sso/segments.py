"""Continuous observing segments of one sector-camera-CCD.

A segment boundary is an actual gap in the frame times: consecutive frames
more than max(0.1 d, 3 cadences) apart, in spacecraft time
(``TIME - BARYCORR``). Flagged frames and momentum dumps are still frames,
so they never split a segment. The public TESS orbit-times table is used
only to cross-check the boundaries.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import requests
from astropy.table import Table
from astropy.time import Time

from tess_sso.storage.frames import FrameTable
from tess_sso.errors import ServiceError
from tess_sso.network import with_retries
from tess_sso.paths import reference_dir

BTJD0 = 2457000.0
MIN_GAP_DAYS = 0.1
GAP_CADENCES = 3
ORBIT_TIMES_URL = "https://tess.mit.edu/public/files/TESS_orbit_times.csv"
# Segment and orbit-table boundaries further apart than this are reported.
CHECK_TOLERANCE_DAYS = 0.05


class DownlinkMismatchWarning(UserWarning):
    """Segments found in the data disagree with the TESS orbit-times table."""


@dataclass(frozen=True)
class Segment:
    index: int
    first: int  # index of the first frame
    last: int  # index of the last frame (inclusive)
    start: float  # spacecraft BTJD (TDB) of the first frame
    stop: float  # spacecraft BTJD (TDB) of the last frame
    source: str  # "gap" or "user"

    @property
    def n_frames(self) -> int:
        return self.last - self.first + 1


def spacecraft_time(frames: FrameTable) -> np.ndarray:
    """Frame mid-times at the spacecraft: barycentric BTJD minus BARYCORR (TDB)."""
    return frames.time - frames.barycorr


def gap_threshold(t: np.ndarray) -> float:
    cadence = float(np.median(np.diff(t))) if len(t) > 1 else 0.0
    return max(MIN_GAP_DAYS, GAP_CADENCES * cadence)


def find_segments(t: np.ndarray) -> list[Segment]:
    """Split sorted frame times at gaps longer than ``gap_threshold``."""
    t = np.asarray(t, dtype=np.float64)
    if len(t) == 0:
        return []
    if not np.all(np.diff(t) > 0):
        raise ValueError("frame times must be strictly increasing")
    breaks = np.flatnonzero(np.diff(t) > gap_threshold(t)) + 1
    firsts = np.concatenate([[0], breaks])
    lasts = np.concatenate([breaks - 1, [len(t) - 1]])
    return [
        Segment(i, int(a), int(b), float(t[a]), float(t[b]), "gap")
        for i, (a, b) in enumerate(zip(firsts, lasts))
    ]


def user_segments(t: np.ndarray, bounds: Sequence[tuple[float, float]]) -> list[Segment]:
    """Segments given as (start, stop) spacecraft-BTJD pairs; every frame must fall in exactly one."""
    t = np.asarray(t, dtype=np.float64)
    member = np.array([(t >= a) & (t <= b) for a, b in bounds])
    if not np.all(member.sum(axis=0) == 1):
        raise ValueError("every frame must fall in exactly one user segment")
    segments = []
    for i, row in enumerate(member):
        idx = np.flatnonzero(row)
        if len(idx) == 0:
            raise ValueError(f"user segment {bounds[i]} contains no frames")
        if idx[-1] - idx[0] + 1 != len(idx):
            raise ValueError(f"user segment {bounds[i]} is not contiguous")
        segments.append(Segment(i, int(idx[0]), int(idx[-1]), float(t[idx[0]]), float(t[idx[-1]]), "user"))
    return sorted(segments, key=lambda s: s.first)


def segment_ids(segments: Sequence[Segment], n_frames: int) -> np.ndarray:
    """Segment index of every frame."""
    ids = np.full(n_frames, -1, dtype=np.int32)
    for seg in segments:
        ids[seg.first : seg.last + 1] = seg.index
    return ids


def _get(url: str) -> requests.Response:
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return response


def fetch_orbit_times() -> str:
    try:
        response = with_retries(lambda: _get(ORBIT_TIMES_URL), "TESS orbit times")
    except requests.RequestException as exc:
        raise ServiceError(f"could not download {ORBIT_TIMES_URL}: {exc}") from exc
    return response.text


def orbit_times_path(root: str | os.PathLike | None) -> Path:
    return reference_dir(root) / "TESS_orbit_times.csv"


def parse_orbit_times(text: str) -> Table:
    """Sector, orbit, and start/stop as spacecraft BTJD (TDB); the file gives UTC."""
    lines = [line for line in text.splitlines() if line.strip() and not line.startswith("#")]
    table = Table.read(lines, format="ascii.csv")
    start = Time(list(table["Start of Orbit"]), format="iso", scale="utc").tdb.jd - BTJD0
    stop = Time(list(table["End of Orbit"]), format="iso", scale="utc").tdb.jd - BTJD0
    return Table({"sector": np.asarray(table["Sector"], int), "orbit": np.asarray(table["Orbit"], int),
                  "start": start, "stop": stop})


def load_orbit_times(
    sector: int,
    root: str | os.PathLike | None = None,
    fetch: Callable[[], str] = fetch_orbit_times,
) -> Table:
    """Orbit-table rows for ``sector``; the cached file is refreshed once if the sector is missing."""
    path = orbit_times_path(root)
    if path.exists():
        table = parse_orbit_times(path.read_text())
        if sector in table["sector"]:
            return table[table["sector"] == sector]
    text = fetch()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    tmp.write_text(f"# downloaded {stamp} from {ORBIT_TIMES_URL}\n{text}")
    os.replace(tmp, path)
    table = parse_orbit_times(path.read_text())
    return table[table["sector"] == sector]


@dataclass(frozen=True)
class SegmentCheck:
    ok: bool
    offsets: list[tuple[float, float]]  # (start, stop) offsets from the matched orbit interval, days
    message: str


def check_segments(segments: Sequence[Segment], orbits: Table, tolerance: float = CHECK_TOLERANCE_DAYS) -> SegmentCheck:
    """Compare segments with orbit-table intervals, matched by largest overlap."""
    if len(orbits) == 0:
        return SegmentCheck(False, [], "the orbit-times table has no rows for this sector")
    offsets = []
    problems = []
    for seg in segments:
        overlap = np.minimum(seg.stop, orbits["stop"]) - np.maximum(seg.start, orbits["start"])
        best = int(np.argmax(overlap))
        if overlap[best] <= 0:
            problems.append(f"segment {seg.index} overlaps no orbit interval")
            offsets.append((np.nan, np.nan))
            continue
        d_start = seg.start - float(orbits["start"][best])
        d_stop = seg.stop - float(orbits["stop"][best])
        offsets.append((d_start, d_stop))
        if abs(d_start) > tolerance or abs(d_stop) > tolerance:
            problems.append(f"segment {seg.index} is off by {d_start:+.3f} d / {d_stop:+.3f} d")
    if len(segments) != len(orbits):
        problems.append(f"{len(segments)} segment(s) in the data, {len(orbits)} in the orbit table")
    return SegmentCheck(not problems, offsets, "; ".join(problems) or "segments match the orbit table")
