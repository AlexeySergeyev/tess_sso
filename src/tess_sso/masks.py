"""Source masks: stars (Gaia + PRF, OR a data-driven mask) and the moving asteroid.

Everything lives on a ``Grid``: the bounding box of the background strip.
Star masks are constant within a continuous observing segment; the asteroid
footprint is per exposure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from scipy import ndimage

from tess_sso.storage.fits_io import read_pixel_file
from tess_sso.storage.store import PixelStore
from tess_sso.geometry import Track
from tess_sso.prf import Prf, window_origin
from tess_sso.segments import Segment

# Data-driven star mask thresholds (median flux and gradient relative to the strip median).
FLUX_FACTOR = 1.05
GRADIENT_FACTOR = 4.0
# Gaia mask: predicted stellar flux above K_SIGMA x per-exposure background noise. k = 1 masks
# almost everything in dense fields (noise ~1.3 e-/s vs ~13 e-/s of faint-star wings per pixel
# near asteroid 95 in sector 71); 10 leaves about half the strip usable. To be tuned.
K_SIGMA = 10.0
# ...raised per segment where it would leave less than this share of the strip usable: in the
# Galactic plane (35534, sector 39: 180 000 stars) the faint-star wings reach 20-200 x the
# noise everywhere and k = 10 leaves 6 of 37 000 pixels. The background model then absorbs
# the faintest stars' wings, as it does for every star below k elsewhere. To be tuned.
MIN_USABLE_FRACTION = 0.2
# Asteroid footprint: pixels with at least this fraction of the asteroid's flux.
ASTEROID_THRESHOLD = 0.01
# ...or where its predicted flux is at least ASTEROID_K_SIGMA x the per-exposure noise. A 1 %
# threshold alone leaves wings of tens of e-/s around a bright asteroid (95 at V ~ 12).
ASTEROID_K_SIGMA = 1.0
# Rough TESS magnitude of an asteroid from Horizons V (roughly solar colours); a larger offset
# only makes the mask slightly wider.
ASTEROID_V_TO_T = 0.8
ASTEROID_WINDOW = 21


class DenseFieldWarning(UserWarning):
    """The Gaia mask threshold was raised in a segment to leave part of the strip to the background."""


@dataclass(frozen=True)
class Grid:
    origin: tuple[int, int]  # 1-based (row, col) of image[0, 0]
    shape: tuple[int, int]

    @classmethod
    def around(cls, pixels: Sequence[tuple[int, int]]) -> "Grid":
        rows = [p[0] for p in pixels]
        cols = [p[1] for p in pixels]
        return cls((min(rows), min(cols)), (max(rows) - min(rows) + 1, max(cols) - min(cols) + 1))

    def image(self, pixels: Iterable[tuple[int, int]]) -> np.ndarray:
        """Boolean image, True at ``pixels``."""
        out = np.zeros(self.shape, dtype=bool)
        for row, col in pixels:
            out[row - self.origin[0], col - self.origin[1]] = True
        return out


def data_driven_star_mask(
    median_image: np.ndarray, valid: np.ndarray, flux_factor: float = FLUX_FACTOR, gradient_factor: float = GRADIENT_FACTOR
) -> np.ndarray:
    """Bright AND steep pixels of the median image, holes filled (catches sources Gaia misses)."""
    med = np.where(valid, median_image, np.nan)
    level = np.nanmedian(med)
    bright = np.nan_to_num(med, nan=-np.inf) >= flux_factor * level
    gy, gx = np.gradient(med)
    grad = np.hypot(gy, gx)
    steep = np.nan_to_num(grad, nan=-np.inf) >= gradient_factor * np.nanmedian(grad)
    steep = ndimage.binary_fill_holes(steep)
    return bright & steep & valid


def gaia_star_mask(star_image: np.ndarray, noise_image: np.ndarray, k: float = K_SIGMA) -> np.ndarray:
    """Pixels where the predicted stellar flux exceeds ``k`` x the per-exposure background noise."""
    with np.errstate(divide="ignore", invalid="ignore"):  # the ratio, as dense_field_k ranks it
        return star_image / noise_image > k


def dense_field_k(
    star_image: np.ndarray, noise_image: np.ndarray, candidates: np.ndarray, n_strip: int,
    k: float = K_SIGMA, min_fraction: float = MIN_USABLE_FRACTION,
) -> float:
    """``k``, or the smallest larger one that leaves ``min_fraction`` of the strip out of the Gaia mask.

    ``candidates`` are the strip pixels nothing else masks; the ones with the faintest
    predicted stars relative to the noise are given back to the background.
    """
    ratio = np.sort(star_image[candidates] / noise_image[candidates])
    needed = int(np.ceil(min_fraction * n_strip))
    if needed == 0 or np.count_nonzero(ratio <= k) >= needed:
        return k
    return float(ratio[min(needed, len(ratio)) - 1])


def ephemeris_margin_px(sigma_arcsec: float, pixel_arcsec: float = 21.0) -> int:
    """Asteroid-mask margin: the 3-sigma ephemeris uncertainty in whole pixels.

    Rounded to the nearest pixel, so uncertainties below half a pixel add no
    margin (the 1 % PRF footprint already reaches well beyond the core).
    """
    if not np.isfinite(sigma_arcsec) or sigma_arcsec <= 0:
        return 0
    return int(np.floor(sigma_arcsec / pixel_arcsec + 0.5))


def exposure_ends(track: Track, exposure_days: np.ndarray) -> tuple[np.ndarray, ...]:
    """Asteroid position at the start and end of each exposure, from the local track velocity."""
    t = track.time
    v_row = np.gradient(track.row, t) if len(t) > 1 else np.zeros_like(t)
    v_col = np.gradient(track.col, t) if len(t) > 1 else np.zeros_like(t)
    half = exposure_days / 2
    return track.row - v_row * half, track.col - v_col * half, track.row + v_row * half, track.col + v_col * half


def asteroid_thresholds(v_mag: np.ndarray, noise: float, k: float = ASTEROID_K_SIGMA) -> np.ndarray:
    """Per-frame PRF-fraction threshold: min(1 %, k x noise / predicted asteroid flux).

    Frames without a predicted V keep the 1 % threshold.
    """
    from tess_sso.gaia import tess_flux

    predicted = tess_flux(np.asarray(v_mag, dtype=float) - ASTEROID_V_TO_T)
    with np.errstate(invalid="ignore", divide="ignore"):
        noise_fraction = k * noise / predicted
    return np.where(np.isfinite(noise_fraction), np.minimum(ASTEROID_THRESHOLD, noise_fraction), ASTEROID_THRESHOLD)


def asteroid_footprints(
    prf: Prf,
    track: Track,
    exposure_days: np.ndarray,
    grid: Grid,
    threshold: float | np.ndarray = ASTEROID_THRESHOLD,
    margin_px: int = 0,
    window: int = ASTEROID_WINDOW,
) -> list[np.ndarray]:
    """Per frame: flat grid indices where the motion-smeared PRF >= ``threshold``, dilated by ``margin_px``.

    ``threshold`` may be one value or one per frame (see ``asteroid_thresholds``).
    """
    thresholds = np.broadcast_to(np.asarray(threshold, dtype=float), (len(track),))
    r0s, c0s, r1s, c1s = exposure_ends(track, exposure_days)
    out = []
    half = window // 2
    for i in range(len(track)):
        row, col = track.row[i], track.col[i]
        far = not (
            grid.origin[0] - half <= row < grid.origin[0] + grid.shape[0] + half
            and grid.origin[1] - half <= col < grid.origin[1] + grid.shape[1] + half
        )
        if not (track.on_science[i] and np.isfinite(row) and np.isfinite(col)) or far:
            out.append(np.empty(0, dtype=np.int64))
            continue
        ends = (r0s[i], c0s[i], r1s[i], c1s[i])
        if not np.isfinite(ends).all():  # next to a frame without a position: no smear
            ends = (row, col, row, col)
        wr, wc = window_origin(row, col, window)
        stamp = prf.smeared(*ends, (wr, wc), (window, window)) >= thresholds[i]
        if margin_px > 0:
            stamp = ndimage.binary_dilation(stamp, iterations=margin_px)
        ii, jj = np.nonzero(stamp)
        rows, cols = ii + wr - grid.origin[0], jj + wc - grid.origin[1]
        inside = (rows >= 0) & (rows < grid.shape[0]) & (cols >= 0) & (cols < grid.shape[1])
        out.append(np.ravel_multi_index((rows[inside], cols[inside]), grid.shape))
    return out


def segment_medians(
    store: PixelStore, grid: Grid, strip: np.ndarray, segments: Sequence[Segment], bad_frames: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per segment: median flux and median flux error of every strip pixel over good frames.

    Pixel files are read one at a time, so memory stays at one file. Returns
    two arrays shaped (nsegments, *grid.shape), NaN outside the strip.
    """
    median = np.full((len(segments), *grid.shape), np.nan, dtype=np.float32)
    noise = np.full_like(median, np.nan)
    for path in store.files():
        data = read_pixel_file(path)
        i = data.pixels[:, 0] - grid.origin[0]
        j = data.pixels[:, 1] - grid.origin[1]
        inside = (i >= 0) & (i < grid.shape[0]) & (j >= 0) & (j < grid.shape[1])
        inside[inside] &= strip[i[inside], j[inside]]
        if not inside.any():
            continue
        for k, seg in enumerate(segments):
            frames = np.arange(seg.first, seg.last + 1)
            frames = frames[~bad_frames[frames]]
            with np.errstate(all="ignore"):
                median[k, i[inside], j[inside]] = np.nanmedian(data.flux[frames][:, inside], axis=0)
                noise[k, i[inside], j[inside]] = np.nanmedian(data.flux_err[frames][:, inside], axis=0)
    median[:, ~strip] = np.nan
    noise[:, ~strip] = np.nan
    return median, noise


@dataclass(frozen=True)
class SegmentMasks:
    segment: int
    star_gaia: np.ndarray  # bool, grid shape
    star_data: np.ndarray
    asteroid: np.ndarray  # every pixel the asteroid covers in this segment
    strip: np.ndarray
    finite: np.ndarray
    k: float = K_SIGMA  # Gaia mask threshold used, raised in dense fields

    @property
    def star(self) -> np.ndarray:
        return self.star_gaia | self.star_data

    @property
    def one_mask_only(self) -> np.ndarray:
        """In exactly one of the two star masks: a hint of catalogue, WCS, or PRF problems."""
        return (self.star_gaia ^ self.star_data) & self.strip

    @property
    def usable(self) -> np.ndarray:
        """Pixels the scattered-light model may use."""
        return self.strip & self.finite & ~self.star & ~self.asteroid

    @property
    def usable_fraction(self) -> float:
        return float(self.usable.sum() / max(self.strip.sum(), 1))


def build_segment_masks(
    segment: int,
    strip: np.ndarray,
    median_image: np.ndarray,
    noise_image: np.ndarray,
    star_image: np.ndarray,
    footprints: Sequence[np.ndarray],
    k: float = K_SIGMA,
    min_usable: float = MIN_USABLE_FRACTION,
) -> SegmentMasks:
    finite = np.isfinite(median_image) & np.isfinite(noise_image) & strip
    asteroid = np.zeros(strip.shape, dtype=bool)
    if len(footprints):
        idx = np.concatenate(list(footprints)) if any(len(f) for f in footprints) else np.empty(0, int)
        asteroid.flat[idx] = True
    star_data = data_driven_star_mask(median_image, finite)
    noise = np.where(finite, noise_image, np.inf)
    k = dense_field_k(star_image, noise, finite & ~star_data & ~asteroid, int(strip.sum()), k, min_usable)
    return SegmentMasks(
        segment=segment,
        star_gaia=gaia_star_mask(star_image, noise, k) & strip,
        star_data=star_data,
        asteroid=asteroid,
        strip=strip,
        finite=finite,
        k=k,
    )


# Bits of the MASKS image in masks.fits.
BIT_STRIP, BIT_STAR_GAIA, BIT_STAR_DATA, BIT_ASTEROID, BIT_USABLE = 1, 2, 4, 8, 16


@dataclass(frozen=True)
class ObservationMasks:
    grid: Grid
    segments: list[Segment]
    per_segment: list[SegmentMasks]
    footprints: list[np.ndarray]  # per frame, flat grid indices
    median: np.ndarray  # (nsegments, *grid.shape)
    noise: np.ndarray
    star_image: np.ndarray  # predicted flux of modelled (unsaturated) Gaia stars
    header: dict  # provenance and thresholds

    def describe(self) -> str:
        lines = [f"grid {self.grid.shape} at {self.grid.origin}, {len(self.segments)} segment(s)"]
        for seg, m in zip(self.segments, self.per_segment):
            lines.append(
                f"  segment {seg.index}: usable {m.usable_fraction:5.1%} of {int(m.strip.sum())} strip px, k = {m.k:g}; "
                f"stars Gaia {int(m.star_gaia.sum())}, data {int(m.star_data.sum())}, "
                f"Gaia only {int((m.star_gaia & ~m.star_data & m.strip).sum())}, "
                f"data only (missing from Gaia) {int((m.star_data & ~m.star_gaia & m.strip).sum())}; "
                f"asteroid {int(m.asteroid.sum())}"
            )
        return "\n".join(lines)

    def bits(self) -> np.ndarray:
        out = np.zeros((len(self.per_segment), *self.grid.shape), dtype=np.uint8)
        for k, m in enumerate(self.per_segment):
            out[k] = (
                BIT_STRIP * m.strip + BIT_STAR_GAIA * m.star_gaia + BIT_STAR_DATA * m.star_data
                + BIT_ASTEROID * m.asteroid + BIT_USABLE * m.usable
            )
        return out


def write_masks(path, masks: ObservationMasks) -> None:
    from astropy.io import fits

    from tess_sso.storage.fits_io import write_atomic

    primary = fits.PrimaryHDU()
    primary.header["ROW0"] = masks.grid.origin[0]
    primary.header["COL0"] = masks.grid.origin[1]
    primary.header["BITS"] = "1 strip, 2 Gaia star, 4 data star, 8 asteroid, 16 usable"
    for key, value in masks.header.items():
        primary.header[key] = value
    frame = np.concatenate([np.full(len(f), i, dtype=np.int32) for i, f in enumerate(masks.footprints)] or [np.empty(0, np.int32)])
    index = np.concatenate(list(masks.footprints) or [np.empty(0, np.int64)])
    footprints = fits.BinTableHDU.from_columns(
        [fits.Column(name="FRAME", format="J", array=frame), fits.Column(name="INDEX", format="K", array=index)],
        name="FOOTPRINTS",
    )
    write_atomic(
        path,
        fits.HDUList(
            [
                primary,
                fits.ImageHDU(masks.bits(), name="MASKS"),
                fits.ImageHDU(masks.median, name="MEDIAN"),
                fits.ImageHDU(masks.noise, name="NOISE"),
                fits.ImageHDU(masks.star_image.astype(np.float32), name="STARMODEL"),
                footprints,
            ]
        ),
    )
