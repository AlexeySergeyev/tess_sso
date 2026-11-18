"""Time columns of the output: barycentric correction toward the asteroid and emission time.

SPOC timestamps (``TIME`` of the frame table) are corrected to the solar
system barycentre for the centre of the CCD. For a light curve of the
asteroid the correction is recomputed for the asteroid's direction:
``TIME = t_sc + TIMECORR``, where ``t_sc = ORIGINAL_TIME - ORIGINAL_TIMECORR``
is the time at the spacecraft (TDB). The emission time at the asteroid
removes the correction again and the TESS-to-asteroid light time:
``TIME_EMIT = TIME - TIMECORR - OBS_DISTANCE / c``.
"""

from __future__ import annotations

import warnings
from typing import Callable

import numpy as np
from astropy import constants, units

from tess_sso._compat import patch_lkspacecraft_listing
from tess_sso.geometry import BTJD0
from tess_sso.network import socket_timeout, with_retries

# light travel time over one au, in days
LIGHT_TIME_AU_D = float((units.au / constants.c).to(units.day).value)

# (spacecraft time as TDB JD, RA deg, Dec deg), all per frame -> correction in days
BarycentricCorrection = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]


class BarycentricFallbackWarning(UserWarning):
    """The asteroid-direction correction was unavailable; the SPOC correction is used."""


def lkspacecraft_correction(t_sc_jd: np.ndarray, ra: np.ndarray, dec: np.ndarray) -> np.ndarray:
    """Barycentric correction toward (ra, dec) from the TESS SPICE kernels, in days.

    lkspacecraft keeps its kernels in the astropy download cache (not in the
    environment) and downloads missing ones on first use.
    """
    from astropy.time import Time
    from lkspacecraft import TESSSpacecraft

    patch_lkspacecraft_listing()

    def spacecraft():
        with warnings.catch_warnings(), socket_timeout():  # it reads MAST's listing with no timeout of its own
            warnings.simplefilter("ignore")
            return TESSSpacecraft()

    tess = with_retries(spacecraft, "lkspacecraft: TESS SPICE kernels")
    out = np.full(len(t_sc_jd), np.nan)
    for i in np.flatnonzero(np.isfinite(t_sc_jd) & np.isfinite(ra) & np.isfinite(dec)):
        # one direction per time: an array call would return every time for every direction
        seconds = tess.get_barycentric_time_correction(
            time=Time(t_sc_jd[i], format="jd", scale="tdb"), ra=float(ra[i]), dec=float(dec[i])
        )
        out[i] = float(np.ravel(seconds)[0]) / 86400.0
    return out


def asteroid_timecorr(
    spacecraft_btjd: np.ndarray,
    ra: np.ndarray,
    dec: np.ndarray,
    spoc_timecorr: np.ndarray,
    correction: BarycentricCorrection = lkspacecraft_correction,
) -> tuple[np.ndarray, str]:
    """Per-frame correction toward the asteroid, and its source ("lkspacecraft" or "SPOC").

    Where the asteroid's position is unknown (NaN) or the correction fails,
    the SPOC value is used; if every frame falls back, a warning says so.
    """
    try:
        timecorr = np.asarray(correction(spacecraft_btjd + BTJD0, ra, dec), dtype=np.float64)
    except Exception as exc:  # missing kernels, lkspacecraft not installed, no network
        warnings.warn(
            f"barycentric correction toward the asteroid failed ({exc!r}); using the SPOC correction",
            BarycentricFallbackWarning,
            stacklevel=2,
        )
        return np.asarray(spoc_timecorr, dtype=np.float64).copy(), "SPOC"
    missing = ~np.isfinite(timecorr)
    timecorr[missing] = spoc_timecorr[missing]
    return timecorr, "SPOC" if missing.all() else "lkspacecraft"


def emission_time(time: np.ndarray, timecorr: np.ndarray, obs_distance_au: np.ndarray) -> np.ndarray:
    """TIME_EMIT = TIME - TIMECORR - OBS_DISTANCE / c, in BTJD days (TDB)."""
    return np.asarray(time, np.float64) - np.asarray(timecorr, np.float64) - np.asarray(obs_distance_au, np.float64) * LIGHT_TIME_AU_D
