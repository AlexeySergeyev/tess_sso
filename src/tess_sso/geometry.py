"""Asteroid position on the CCD in every frame, and the pixels around it.

Pixel coordinates are 1-based (row, column) on the full cube (2078 x 2136),
with pixel k centred at k.0. Science pixels are rows 1-2048, columns 45-2092.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from astropy.table import Table
from astropy.wcs import NoConvergence, WCS
from scipy.interpolate import CubicSpline

from tess_sso.storage.blocks import CCD_NCOLUMNS, CCD_NROWS
from tess_sso.storage.frames import FrameTable
from tess_sso.ephemeris import Ephemeris

BTJD0 = 2457000.0
SCIENCE_ROWS = (1, 2048)
SCIENCE_COLUMNS = (45, 2092)
# Header keywords that make up a frame WCS in the cube frame table.
_WCS_PREFIXES = ("CTYPE1", "CTYPE2", "CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2",
                 "CD1_", "CD2_", "A_", "B_", "AP_", "BP_")


class WcsProjectionWarning(UserWarning):
    """Some sky positions could not be converted to reliable pixel coordinates."""


@dataclass(frozen=True)
class Track:
    """Predicted asteroid position in every frame."""

    time: np.ndarray  # frame mid-time, BTJD
    ra: np.ndarray  # deg, at spacecraft time of the frame
    dec: np.ndarray
    row: np.ndarray  # 1-based, per-frame WCS
    col: np.ndarray
    wcs_fallback: np.ndarray  # True where the frame had no WCS and the average was used
    on_science: np.ndarray  # True where the position is on the science area

    def __len__(self) -> int:
        return len(self.time)

    def within(self, start: float, stop: float) -> "Track":
        """Same frames, with only those between ``start`` and ``stop`` (BTJD) left on the science area."""
        keep = (self.time >= start) & (self.time <= stop)
        return Track(self.time, self.ra, self.dec, self.row, self.col, self.wcs_fallback, self.on_science & keep)


@dataclass(frozen=True)
class FrameGeometry:
    """Observing geometry at each frame's spacecraft time (from the ephemeris)."""

    r: np.ndarray  # distance from the Sun, au
    delta: np.ndarray  # distance from TESS, au
    alpha: np.ndarray  # Sun-target-observer angle, deg
    v: np.ndarray  # predicted V magnitude


def interpolate_geometry(eph: Ephemeris, frames: FrameTable) -> FrameGeometry:
    """Distances, phase angle, and V at each frame's spacecraft time (TDB), by cubic spline."""
    t_sc = frames.time + BTJD0 - frames.barycorr

    def at_frames(values: np.ndarray) -> np.ndarray:
        good = np.isfinite(values)
        if good.sum() < 2:
            return np.full(len(t_sc), np.nan)
        return CubicSpline(eph.time_tdb[good], values[good], extrapolate=False)(t_sc)

    return FrameGeometry(r=at_frames(eph.r), delta=at_frames(eph.delta), alpha=at_frames(eph.alpha), v=at_frames(eph.v))


def interpolate_radec(eph: Ephemeris, frames: FrameTable) -> tuple[np.ndarray, np.ndarray]:
    """RA and Dec at each frame's spacecraft time (TDB), by cubic spline; NaN outside the ephemeris."""
    t_sc = frames.time + BTJD0 - frames.barycorr  # barycentric -> spacecraft time, TDB JD
    ra_unwrapped = np.degrees(np.unwrap(np.radians(eph.ra)))
    ra = CubicSpline(eph.time_tdb, ra_unwrapped, extrapolate=False)(t_sc) % 360.0
    dec = CubicSpline(eph.time_tdb, eph.dec, extrapolate=False)(t_sc)
    return ra, dec


def _wcs_columns(table: Table) -> list[str]:
    return [name for name in table.colnames if name.startswith(_WCS_PREFIXES)]


def _reference_groups(table: Table) -> np.ndarray:
    """Per frame, a label shared by the frames written about the same reference pixel and SIP orders; -1
    without a WCS.

    The FFI headers of one sector need not share a CRPIX (sector 91 moves it by 24 px on both axes between
    orbits): CRVAL, CD and the SIP coefficients only mean something together with their own CRPIX.
    """
    names = [name for name in ("CRPIX1", "CRPIX2", "A_ORDER", "B_ORDER", "AP_ORDER", "BP_ORDER") if name in table.colnames]
    keys = np.column_stack([np.asarray(table[name], float) for name in names])
    crval = np.column_stack([np.asarray(table["CRVAL1"], float), np.asarray(table["CRVAL2"], float)])
    good = np.isfinite(keys).all(axis=1) & np.isfinite(crval).all(axis=1)
    labels = np.full(len(table), -1)
    if good.any():
        labels[good] = np.unique(np.round(keys[good], 6), axis=0, return_inverse=True)[1].ravel()
    return labels


def _main_group(labels: np.ndarray) -> np.ndarray:
    """Frames of the largest reference group (every frame if none has a WCS)."""
    if not (labels >= 0).any():
        return np.ones(len(labels), bool)
    return labels == np.bincount(labels[labels >= 0]).argmax()


def average_wcs(table: Table, rows: np.ndarray | None = None) -> WCS:
    """WCS from the median of every numeric WCS column (and the first non-empty string) over ``rows``.

    By default the rows are the largest group of frames sharing a reference pixel: a median across groups
    would pair one group's CRVAL with another's CRPIX.
    SIP coefficients above the group's declared polynomial order are omitted: mixed-order
    frame tables carry NaN placeholders for those unused terms.
    """
    if rows is None:
        rows = _main_group(_reference_groups(table))
    sip_orders = {
        prefix: int(np.nanmedian(np.asarray(table[f"{prefix}_ORDER"])[rows]))
        for prefix in ("A", "B", "AP", "BP")
        if f"{prefix}_ORDER" in table.colnames
    }
    header = {}
    for name in _wcs_columns(table):
        parts = name.split("_")
        if len(parts) == 3 and parts[0] in ("A", "B", "AP", "BP"):
            if int(parts[1]) + int(parts[2]) > sip_orders.get(parts[0], 0):
                continue
        values = np.asarray(table[name])[rows]
        if values.dtype.kind in "US":
            header[name] = next((str(v) for v in values if str(v).strip()), "")
        else:
            header[name] = float(np.nanmedian(values))
    for key in ("A_ORDER", "B_ORDER", "AP_ORDER", "BP_ORDER"):
        if key in header:
            header[key] = int(header[key])
    header["WCSAXES"] = 2
    header["CTYPE1"] = "RA---TAN-SIP"
    header["CTYPE2"] = "DEC--TAN-SIP"
    return WCS(header, relax=True)


def frame_pixels(table: Table, ra: np.ndarray, dec: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """1-based (row, col) of (ra, dec) in each frame, using that frame's CRPIX, CRVAL and CD.

    The SIP distortion is the average one of the frames sharing the frame's reference pixel. Frames
    without a WCS use the average WCS and are flagged in the third return value.
    Failed WCS inversions leave NaN coordinates (off science) and emit WcsProjectionWarning.
    """
    labels = _reference_groups(table)
    group_wcs = {label: average_wcs(table, labels == label) for label in np.unique(labels[labels >= 0])}
    fallback_wcs = average_wcs(table, _main_group(labels))
    crval1, crval2 = np.asarray(table["CRVAL1"], float), np.asarray(table["CRVAL2"], float)
    cd = np.stack(
        [
            [np.asarray(table["CD1_1"], float), np.asarray(table["CD1_2"], float)],
            [np.asarray(table["CD2_1"], float), np.asarray(table["CD2_2"], float)],
        ]
    ).transpose(2, 0, 1)
    row = np.full(len(ra), np.nan)
    col = np.full(len(ra), np.nan)
    fallback = (labels < 0) | ~np.isfinite(cd).all(axis=(1, 2))
    failed = []
    for i in range(len(ra)):
        if not (np.isfinite(ra[i]) and np.isfinite(dec[i])):
            continue
        if fallback[i]:
            wcs = fallback_wcs
        else:
            wcs = group_wcs[labels[i]]
            wcs.wcs.crval, wcs.wcs.cd = (crval1[i], crval2[i]), cd[i]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                x, y = wcs.all_world2pix([[ra[i], dec[i]]], 0)[0]
            except NoConvergence:
                # SIP inversions can diverge far outside the detector. The best
                # iterate is not a reliable position, even if it looks on science.
                failed.append(i)
                continue
        row[i], col[i] = y + 1, x + 1
    if failed:
        warnings.warn(
            f"WCS inversion did not converge in {len(failed)} frame(s) "
            f"(first frame index: {failed[0]}); pixel coordinates left as NaN "
            "and these frames excluded from the science area",
            WcsProjectionWarning,
            stacklevel=2,
        )
    return row, col, fallback


def frame_wcs(table: Table, frame: int) -> WCS:
    """WCS of one frame: its CRPIX, CRVAL and CD with the average SIP distortion of the frames sharing its
    reference pixel (the average WCS if it has none)."""
    labels = _reference_groups(table)
    cd = np.array([[table["CD1_1"][frame], table["CD1_2"][frame]], [table["CD2_1"][frame], table["CD2_2"][frame]]], float)
    if labels[frame] < 0 or not np.isfinite(cd).all():
        return average_wcs(table, _main_group(labels))
    wcs = average_wcs(table, labels == labels[frame])
    wcs.wcs.crval, wcs.wcs.cd = (float(table["CRVAL1"][frame]), float(table["CRVAL2"][frame])), cd
    return wcs


def pixel_offset_to_sky(
    table: Table, frame: int, ra: float, dec: float, drow: float, dcol: float
) -> tuple[float, float]:
    """(ra, dec) moved by a pixel offset, measured with the WCS of ``frame``."""
    wcs = frame_wcs(table, frame)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        x, y = wcs.all_world2pix([[ra, dec]], 0)[0]
        (ra0, dec0), (ra1, dec1) = wcs.all_pix2world([[x, y], [x + dcol, y + drow]], 0)
    dra = (ra1 - ra0 + 180.0) % 360.0 - 180.0
    return float((ra + dra) % 360.0), float(dec + dec1 - dec0)


def sky_to_pixels(table: Table, ra: np.ndarray, dec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """1-based (row, col) of fixed sources with the average WCS of the frames."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        xy = average_wcs(table).all_world2pix(np.column_stack([ra, dec]), 0) if len(ra) else np.empty((0, 2))
    return xy[:, 1] + 1, xy[:, 0] + 1


def on_science(row: np.ndarray, col: np.ndarray) -> np.ndarray:
    return (
        (row >= SCIENCE_ROWS[0] - 0.5)
        & (row < SCIENCE_ROWS[1] + 0.5)
        & (col >= SCIENCE_COLUMNS[0] - 0.5)
        & (col < SCIENCE_COLUMNS[1] + 0.5)
    )


def make_track(eph: Ephemeris, frames: FrameTable) -> Track:
    ra, dec = interpolate_radec(eph, frames)
    row, col, fallback = frame_pixels(frames.table, ra, dec)
    return Track(
        time=frames.time,
        ra=ra,
        dec=dec,
        row=row,
        col=col,
        wcs_fallback=fallback,
        on_science=on_science(row, col),
    )


def box_pixels(row: float, col: float, size: int) -> list[tuple[int, int]]:
    """``size`` x ``size`` pixels centred on the pixel containing (row, col), clipped to the cube."""
    r0, c0 = int(np.floor(row + 0.5)), int(np.floor(col + 0.5))
    half = size // 2
    return [
        (r, c)
        for r in range(max(1, r0 - half), min(CCD_NROWS, r0 + half) + 1)
        for c in range(max(1, c0 - half), min(CCD_NCOLUMNS, c0 + half) + 1)
    ]


def track_pixels(track: Track, size: int) -> list[tuple[int, int]]:
    """Union of ``size`` x ``size`` boxes around the track, over frames on the science area."""
    pixels: set[tuple[int, int]] = set()
    for row, col, good in zip(track.row, track.col, track.on_science):
        if good:
            pixels.update(box_pixels(row, col, size))
    return sorted(pixels)


def wcs_offsets(track: Track, sector: int, camera: int, ccd: int) -> np.ndarray:
    """Per frame, how far (pixels) the per-frame WCS puts the asteroid from the tesswcs model.

    NaN where nothing is checked: off the science area (off-detector WCS extrapolations
    can disagree substantially even when the solutions agree on the CCD), without a
    position, or on the average WCS already.
    """
    from tesswcs import WCS as TessWCS

    offsets = np.full(len(track), np.nan)
    good = track.on_science & np.isfinite(track.row) & np.isfinite(track.col) & ~track.wcs_fallback
    if not good.any():
        return offsets
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        x, y = TessWCS.from_sector(sector, camera, ccd).world_to_pixel_values(track.ra[good], track.dec[good])
    offsets[good] = np.hypot(np.asarray(y) + 1 - track.row[good], np.asarray(x) + 1 - track.col[good])
    return offsets


def wcs_disagreement(track: Track, sector: int, camera: int, ccd: int) -> float:
    """Largest WCS disagreement (pixels) on the science area; NaN if none can be checked."""
    offsets = wcs_offsets(track, sector, camera, ccd)
    return float(np.nanmax(offsets)) if np.isfinite(offsets).any() else float("nan")


def with_average_wcs(track: Track, table: Table, frames: np.ndarray) -> Track:
    """``track`` with ``frames`` placed by the average WCS of the largest group and flagged as fallback.

    For frames whose own WCS solution is wrong: the same treatment as frames without one.
    """
    row, col, fallback = track.row.copy(), track.col.copy(), track.wcs_fallback.copy()
    wcs = average_wcs(table)
    for i in np.flatnonzero(frames):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                x, y = wcs.all_world2pix([[track.ra[i], track.dec[i]]], 0)[0]
            except NoConvergence:
                x = y = np.nan
        row[i], col[i] = y + 1, x + 1
        fallback[i] = True
    return Track(track.time, track.ra, track.dec, row, col, fallback, on_science(row, col))
