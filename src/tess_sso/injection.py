"""Injection of a known signal into the pixels (phase 10c).

The signal is a factor A(t) on the asteroid's own flux: every frame gets the
image (A(t) - 1) x F(t) x PRF(t) added to its pixels as they are read, where
F(t) is the asteroid's flux measured without injection and PRF(t) its
motion-smeared PRF at the shifted position the photometry uses. Every step
(masks, scattered light, stars, photometry) sees the injected pixels, and
the ratio of the injected to the plain photometry recovers A(t) whatever
the asteroid does itself.

The photometry is linear in the data, so with the masks and models fixed
the injected part would come back exactly; what the injection tests is
whether the models, the flags, and the timing change it: a scattered-light
or stellar model that absorbs the asteroid's variation, flags raised by it,
a signal put at the wrong time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

BOX = 15  # pixels on a side of the injected image
# The validation's signal (spec: Validation): a sinusoid and 1-hour box dips,
# one at each bright-star crossing and others every 1.5 days away from them.
AMPLITUDE = 0.05
PERIOD_DAYS = 0.25
DEPTH = 0.02
DIP_DAYS = 1 / 24
DIP_SPACING_DAYS = 1.5


def sinusoid(amplitude: float, period_days: float, phase: float = 0.0) -> Callable[[np.ndarray], np.ndarray]:
    """A(t) = 1 + amplitude sin(2 pi t / period + phase)."""
    return lambda t: 1.0 + amplitude * np.sin(2 * np.pi * np.asarray(t, float) / period_days + phase)


def dips(windows: Sequence[tuple[float, float]], depth: float) -> Callable[[np.ndarray], np.ndarray]:
    """A(t) = 1 - depth inside each (start, stop) window (BTJD), 1 elsewhere."""
    def factor(t):
        t = np.asarray(t, float)
        out = np.ones_like(t)
        for lo, hi in windows:
            out[(t >= lo) & (t <= hi)] = 1.0 - depth
        return out

    return factor


@dataclass(frozen=True)
class InjectedSignal:
    """A(t) = (1 + amplitude sin(2 pi t / period)) x (1 - depth inside each dip window); t in SPOC BTJD."""

    amplitude: float = AMPLITUDE
    period: float = PERIOD_DAYS
    depth: float = DEPTH
    crossing: tuple[tuple[float, float], ...] = ()  # dip windows at the bright-star crossings
    isolated: tuple[tuple[float, float], ...] = ()  # dip windows away from them

    @property
    def windows(self) -> list[tuple[float, float]]:
        return list(self.crossing) + list(self.isolated)

    def __call__(self, t: np.ndarray) -> np.ndarray:
        return sinusoid(self.amplitude, self.period)(t) * dips(self.windows, self.depth)(t)

    @property
    def label(self) -> str:
        return (f"sin {self.amplitude:g} {self.period * 24:g} h x {len(self.windows)} dips {self.depth:g}")

    @classmethod
    def for_lightcurve(cls, lightcurve, dip_days: float = DIP_DAYS, spacing: float = DIP_SPACING_DAYS, **kw):
        """Dips from a plain light-curve file: one centred on each crossing's brightest star point (crossings as
        in :func:`tess_sso.validation.crossing_windows`), and one every ``spacing`` days of each segment at least
        0.1 d from the crossings."""
        from astropy.io import fits

        from tess_sso.validation import crossing_windows, in_windows

        hdul = lightcurve if isinstance(lightcurve, fits.HDUList) else fits.open(lightcurve)
        d = hdul["LIGHTCURVE_PRF"].data
        t, star = np.asarray(d["TIME"], float), np.asarray(d["STAR_FLUX"], float)
        t_spoc = np.asarray(hdul["EXTRAS"].data["ORIGINAL_TIME"], float)
        crossings = crossing_windows(t, star)
        at = []
        for lo, hi in crossings:
            inside = np.flatnonzero((t >= lo) & (t <= hi))
            k = inside[np.argmax(np.abs(star[inside]))]
            at.append((t_spoc[k] - dip_days / 2, t_spoc[k] + dip_days / 2))
        near = [(lo - 0.1, hi + 0.1) for lo, hi in crossings]
        isolated = []
        for seg in np.unique(d["SEGMENT"]):
            ts = t_spoc[d["SEGMENT"] == seg]
            for c in np.arange(ts.min() + 0.5, ts.max() - 0.5, spacing):
                w = (c - dip_days / 2, c + dip_days / 2)
                if not in_windows(np.array(w), near).any():
                    isolated.append(w)
        return cls(crossing=tuple(at), isolated=tuple(isolated), **kw)


@dataclass
class Injection:
    """Per-frame images to add: frame index -> (origin (row, col), image (BOX, BOX)); ``label`` names the signal."""

    images: dict[int, tuple[tuple[int, int], np.ndarray]] = field(default_factory=dict)
    label: str = ""

    @classmethod
    def build(
        cls,
        factor: Callable[[np.ndarray], np.ndarray],
        times: np.ndarray,
        flux: np.ndarray,
        ends: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        shifts: np.ndarray,
        prf,
        label: str = "",
    ) -> "Injection":
        """``times`` (BTJD), ``flux`` (e-/s, NaN: nothing injected), the track ``ends`` (row0, col0,
        row1, col1), and the PRF ``shifts`` (row, col) are per frame; ``prf`` has ``smeared``."""
        amplitude = (np.asarray(factor(times), float) - 1.0) * np.asarray(flux, float)
        r0, c0, r1, c1 = (np.asarray(v, float) for v in ends)
        images = {}
        for f in np.flatnonzero(np.isfinite(amplitude) & (amplitude != 0) & np.isfinite(r0) & np.isfinite(c1)):
            dr, dc = shifts[f]
            centre = (0.5 * (r0[f] + r1[f]) + dr, 0.5 * (c0[f] + c1[f]) + dc)
            origin = (int(np.floor(centre[0] + 0.5)) - BOX // 2, int(np.floor(centre[1] + 0.5)) - BOX // 2)
            image = prf.smeared(r0[f] + dr, c0[f] + dc, r1[f] + dr, c1[f] + dc, origin, (BOX, BOX))
            images[int(f)] = (origin, amplitude[f] * image)
        return cls(images, label)

    def add(self, flux: np.ndarray, pixels: Sequence[tuple[int, int]], first_frame: int) -> np.ndarray:
        """``flux`` (nframes, npixels) of frames first_frame... with the injected images added."""
        if not self.images:
            return flux
        pixels = np.asarray(pixels, int).reshape(-1, 2)
        out = flux.copy()
        for k in range(flux.shape[0]):
            entry = self.images.get(first_frame + k)
            if entry is None:
                continue
            (r0, c0), image = entry
            i, j = pixels[:, 0] - r0, pixels[:, 1] - c0
            inside = (i >= 0) & (i < BOX) & (j >= 0) & (j < BOX)
            out[k, inside] += image[i[inside], j[inside]]
        return out


class InjectedStore:
    """A PixelStore whose pixels come back with an injection added; everything else is the store's."""

    def __init__(self, store, injection: Injection):
        self._store = store
        self._injection = injection

    def __getattr__(self, name):
        return getattr(self._store, name)

    def read_pixels(self, pixels, frames: slice | None = None):
        flux, err = self._store.read_pixels(pixels, frames=frames)
        first = 0 if frames is None or frames.start is None else frames.start
        return self._injection.add(flux, [(int(r), int(c)) for r, c in pixels], first).astype(flux.dtype), err


def injection_for(asteroid, sector: int, camera: int, ccd: int, factor, plain_lightcurve, label: str = "") -> Injection:
    """The injection of ``factor`` into an observation, from its plain single-exposure light-curve file.

    The file gives each exposure's flux and PRF shift (matched to frames by
    CADENCENO); frames it lacks get nothing injected. ``label`` defaults to
    the factor's own, if it has one.
    """
    from astropy.io import fits

    from tess_sso.masks import exposure_ends

    frames = asteroid.store(sector, camera, ccd).frames()
    lc = fits.getdata(plain_lightcurve, "LIGHTCURVE_PRF")
    if np.any(lc["N_EXP"] > 1):
        raise ValueError("the injection needs a single-exposure light curve: one flux per frame")
    index = {int(c): i for i, c in enumerate(frames.cadenceno)}
    flux = np.full(len(frames), np.nan)
    shifts = np.zeros((len(frames), 2))
    for c, f, dcol, drow in zip(lc["CADENCENO"], lc["FLUX"], lc["PRF_SHIFT1"], lc["PRF_SHIFT2"]):
        k = index.get(int(c))
        if k is not None:
            flux[k], shifts[k] = f, (drow, dcol)
    track = asteroid.track(sector, camera, ccd)
    ends = exposure_ends(track, frames.tstop - frames.tstart)
    label = label or getattr(factor, "label", "")
    return Injection.build(factor, frames.time, flux, ends, shifts, asteroid._prf(sector, camera, ccd), label)
