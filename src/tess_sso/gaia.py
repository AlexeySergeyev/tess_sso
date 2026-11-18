"""Gaia DR3 stars around an asteroid track: query, cache, epoch propagation, TESS flux.

The catalogue release is part of the cache key and of the provenance, so a
DR4 adapter can be added later without mixing releases.
"""

from __future__ import annotations

import hashlib
import os
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import requests
from astropy.table import Table

from tess_sso.storage.ecsv_io import read_legacy_fits, read_table, write_table
from tess_sso.errors import ServiceError
from tess_sso.network import NetworkRetryWarning, is_transient, with_retries

RELEASE = "DR3"
GAIA_TAP_URL = "https://gea.esac.esa.int/tap-server/tap"
GAIA_TAP_MIRROR_URL = "https://gaia.ari.uni-heidelberg.de/tap"
GAIA_TABLE = "gaiadr3.gaia_source_lite"
GAIA_EPOCH = 2016.0  # Julian year of DR3 positions
TESS_ZERO_POINT = 20.44  # TESS magnitude of 1 e-/s
PIXEL_ARCSEC = 21.0
COLUMNS = ("source_id", "ra", "dec", "pmra", "pmdec", "phot_g_mean_mag", "bp_rp")

GaiaFetch = Callable[[str], Table]


def tess_magnitude(g: np.ndarray, bp_rp: np.ndarray) -> np.ndarray:
    """TESS magnitude from Gaia G and BP-RP (TIC v8, Stassun et al. 2019, eq. 1).

    Colours are clipped to the calibrated range -0.2..3.0; without a colour,
    T = G - 0.430.
    """
    g = np.asarray(g, dtype=np.float64)
    c = np.clip(np.asarray(bp_rp, dtype=np.float64), -0.2, 3.0)
    t = g - 0.00522555 * c**3 + 0.0891337 * c**2 - 0.633923 * c + 0.0324473
    return np.where(np.isfinite(c), t, g - 0.430)


def tess_flux(tmag: np.ndarray) -> np.ndarray:
    """Expected flux in e-/s for a TESS magnitude."""
    return 10.0 ** (-0.4 * (np.asarray(tmag, dtype=np.float64) - TESS_ZERO_POINT))


def propagate(ra, dec, pmra, pmdec, epoch: float) -> tuple[np.ndarray, np.ndarray]:
    """Linear proper-motion propagation from the Gaia epoch to ``epoch`` (Julian year).

    ``pmra`` includes cos(dec), in mas/yr as Gaia gives it; missing proper
    motions count as zero.
    """
    dt = epoch - GAIA_EPOCH
    pmra = np.nan_to_num(np.asarray(pmra, dtype=np.float64))
    pmdec = np.nan_to_num(np.asarray(pmdec, dtype=np.float64))
    dec = np.asarray(dec, dtype=np.float64)
    new_dec = dec + pmdec * dt / 3.6e6
    new_ra = np.asarray(ra, dtype=np.float64) + pmra * dt / 3.6e6 / np.cos(np.radians(dec))
    return new_ra % 360.0, new_dec


def angular_separation(ra1, dec1, ra2, dec2) -> float:
    """Small-angle separation in degrees (good to well under a pixel over a few degrees)."""
    dra = ((ra2 - ra1 + 180.0) % 360.0 - 180.0) * np.cos(np.radians((dec1 + dec2) / 2))
    return float(np.hypot(dra, dec2 - dec1))


def cones_along_track(ra: np.ndarray, dec: np.ndarray, half_width_px: float) -> list[tuple[float, float, float]]:
    """Circles (ra, dec, radius_deg) covering a band of ``half_width_px`` around the track.

    Radius = half-width x sqrt(2); centres are at most one half-width apart,
    which covers the band with room to spare.
    """
    half_width = half_width_px * PIXEL_ARCSEC / 3600.0
    radius = half_width * np.sqrt(2)
    good = np.isfinite(ra) & np.isfinite(dec)
    ra, dec = np.asarray(ra, float)[good], np.asarray(dec, float)[good]
    cones: list[tuple[float, float, float]] = []
    for r, d in zip(ra, dec):
        if cones and angular_separation(cones[-1][0], cones[-1][1], r, d) < half_width:
            continue
        cones.append((float(r), float(d), float(radius)))
    if len(ra) and angular_separation(cones[-1][0], cones[-1][1], ra[-1], dec[-1]) > 0:
        cones.append((float(ra[-1]), float(dec[-1]), float(radius)))
    return cones


def build_adql(cones: Sequence[tuple[float, float, float]], g_max: float) -> str:
    circles = " OR ".join(
        f"1=CONTAINS(POINT(ra, dec), CIRCLE({r:.6f}, {d:.6f}, {rad:.6f}))" for r, d, rad in cones
    )
    return (
        f"SELECT {', '.join(COLUMNS)} FROM {GAIA_TABLE}\n"
        f"WHERE ({circles})\nAND phot_g_mean_mag <= {g_max:.2f}"
    )


class _GaiaSession(requests.Session):
    def request(self, method, url, **kwargs):
        # Bound both connection time and the wait for a synchronous catalogue query.
        kwargs.setdefault("timeout", (15, 300))
        return super().request(method, url, **kwargs)


def _query_gaia(url: str, adql: str) -> Table:
    import pyvo

    with _GaiaSession() as session:
        service = pyvo.dal.TAPService(url, session=session)
        return service.run_sync(adql, maxrec=1_000_000).to_table()


def gaia_tap_fetch(adql: str) -> Table:
    """Fetch DR3 from ESA, using the Heidelberg mirror when ESA is unavailable."""
    from pyvo.dal.exceptions import DALFormatError, DALServiceError

    try:
        return _query_gaia(GAIA_TAP_URL, adql)
    except Exception as exc:
        http_status = exc.code if isinstance(exc, DALServiceError) else None
        unavailable = http_status in (408, 429) or (http_status is not None and http_status >= 500)
        cause = exc.cause if isinstance(exc, DALServiceError) and exc.cause else exc
        if not (unavailable or isinstance(exc, DALFormatError) or is_transient(cause)):
            raise ServiceError(f"Gaia TAP query failed at {GAIA_TAP_URL}: {exc}") from exc
        primary_error = str(exc)

    warnings.warn(f"Gaia TAP at {GAIA_TAP_URL} failed ({primary_error}). Trying {GAIA_TAP_MIRROR_URL}.",
                  NetworkRetryWarning, stacklevel=2)
    # Heidelberg uses ADQL 2.0 geometry. Translate the generated cones on the wire
    # so the same DR3 query still has the same cache key on either service.
    mirror_adql = adql.replace("POINT(ra, dec), CIRCLE(", "POINT('ICRS', ra, dec), CIRCLE('ICRS', ")
    try:
        return with_retries(lambda: _query_gaia(GAIA_TAP_MIRROR_URL, mirror_adql), "Gaia TAP mirror")
    except Exception as exc:
        raise ServiceError(f"Gaia TAP failed at {GAIA_TAP_URL} ({primary_error}) "
                           f"and at {GAIA_TAP_MIRROR_URL} ({exc})") from exc


@dataclass(frozen=True)
class StarCatalog:
    source_id: np.ndarray
    ra: np.ndarray  # deg, at the observation epoch
    dec: np.ndarray
    g: np.ndarray
    bp_rp: np.ndarray
    tmag: np.ndarray
    flux: np.ndarray  # expected TESS flux, e-/s (a prior, not a measurement)
    has_pm: np.ndarray  # False: two-parameter solution, position not propagated
    release: str = RELEASE

    def __len__(self) -> int:
        return len(self.source_id)


def gaia_path(directory: Path, adql: str) -> Path:
    """``<observation folder>/gaia_<release>_<query hash>.ecsv``."""
    digest = hashlib.sha256(adql.encode()).hexdigest()[:12]
    return Path(directory) / f"gaia_{RELEASE.lower()}_{digest}.ecsv"


def load_stars(
    directory: Path,
    cones: Sequence[tuple[float, float, float]],
    g_max: float,
    epoch: float,
    fetch: GaiaFetch = gaia_tap_fetch,
) -> StarCatalog:
    """Stars inside ``cones`` down to ``g_max``, stored by the exact query, propagated to ``epoch``."""
    adql = build_adql(cones, g_max)
    path = gaia_path(directory, adql)
    table = read_table(path) if path.exists() else read_legacy_fits(path)
    if table is not None:
        if not path.exists():
            write_table(path, table)
    else:
        raw = fetch(adql)
        table = Table({name: np.ma.filled(np.ma.asarray(raw[name], dtype=np.float64 if name != "source_id" else np.int64),
                                          np.nan if name != "source_id" else -1) for name in COLUMNS})
        _, first = np.unique(table["source_id"], return_index=True)
        table = table[np.sort(first)]
        write_table(path, table, {"RELEASE": RELEASE, "TABLE": GAIA_TABLE, "GMAX": float(g_max), "NCONES": len(cones),
                                  "QUERYDAT": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    pmra, pmdec = np.asarray(table["pmra"], float), np.asarray(table["pmdec"], float)
    ra, dec = propagate(table["ra"], table["dec"], pmra, pmdec, epoch)
    g, bp_rp = np.asarray(table["phot_g_mean_mag"], float), np.asarray(table["bp_rp"], float)
    tmag = tess_magnitude(g, bp_rp)
    return StarCatalog(
        source_id=np.asarray(table["source_id"], np.int64), ra=ra, dec=dec, g=g, bp_rp=bp_rp,
        tmag=tmag, flux=tess_flux(tmag), has_pm=np.isfinite(pmra) & np.isfinite(pmdec),
    )
