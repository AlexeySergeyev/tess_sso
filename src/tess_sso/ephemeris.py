"""Horizons ephemerides of an asteroid as seen from TESS, cached on disk."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import requests
from astropy.table import Table
from astropy.time import Time

from tess_sso.storage.ecsv_io import read_legacy_fits, read_table, write_table
from tess_sso.designation import Target
from tess_sso.errors import ServiceError, UnknownDesignationError
from tess_sso.network import with_retries
from tess_sso.paths import asteroid_dir

HORIZONS_TESS = "@-95"  # TESS spacecraft as the observer
# 1: astrometric RA/Dec, 9: V magnitude, 19: r, 20: delta, 21: one-way light time,
# 24: phase angle, 36: 3-sigma position uncertainty
QUANTITIES = "1,9,19,20,21,24,36"
COLUMNS = ("datetime_jd", "RA", "DEC", "V", "r", "delta", "lighttime", "alpha", "RA_3sigma", "DEC_3sigma")

# (designation, start JD UTC, stop JD UTC, step) -> table with COLUMNS
HorizonsFetch = Callable[[str, float, float, str], Table]


@dataclass(frozen=True)
class Ephemeris:
    time_utc: np.ndarray  # JD, UTC (Horizons observer-table times)
    time_tdb: np.ndarray  # JD, TDB
    ra: np.ndarray  # deg
    dec: np.ndarray  # deg
    v: np.ndarray  # predicted V magnitude; NaN where Horizons has none
    r: np.ndarray  # heliocentric distance, au
    delta: np.ndarray  # distance from TESS, au
    lighttime: np.ndarray  # one-way light time, days
    alpha: np.ndarray  # phase angle, deg
    ra_3sigma: np.ndarray  # 3-sigma position uncertainty, arcsec; NaN if unknown
    dec_3sigma: np.ndarray

    @classmethod
    def from_table(cls, table: Table) -> "Ephemeris":
        def col(name: str) -> np.ndarray:
            if name not in table.colnames:  # e.g. caches written before a column was added
                return np.full(len(table), np.nan)
            return np.ma.filled(np.ma.asarray(table[name], dtype=np.float64), np.nan)

        utc = col("datetime_jd")
        return cls(
            time_utc=utc,
            time_tdb=Time(utc, format="jd", scale="utc").tdb.jd,
            ra=col("RA"),
            dec=col("DEC"),
            v=col("V"),
            r=col("r"),
            delta=col("delta"),
            lighttime=col("lighttime") / 1440.0,
            alpha=col("alpha"),
            ra_3sigma=col("RA_3sigma"),
            dec_3sigma=col("DEC_3sigma"),
        )


def classify_horizons_error(exc: Exception) -> Exception:
    """Map an astroquery/Horizons failure to ``UnknownDesignationError`` or ``ServiceError``."""
    text = str(exc)
    if "No matches found" in text or "Unknown target" in text:
        return UnknownDesignationError(f"Horizons does not know the target: {text[:200]}")
    return ServiceError(f"Horizons query failed: {text[:200]}")


def horizons_fetch(designation: str, start_jd: float, stop_jd: float, step: str) -> Table:
    """Query Horizons for ``designation`` seen from TESS between two UTC Julian dates."""
    from astroquery.jplhorizons import Horizons  # deferred: slow import

    epochs = {
        "start": Time(start_jd, format="jd", scale="utc").iso[:16],
        "stop": Time(stop_jd, format="jd", scale="utc").iso[:16],
        "step": step,
    }
    try:
        query = Horizons(id=designation, id_type="smallbody", location=HORIZONS_TESS, epochs=epochs)
        return with_retries(lambda: query.ephemerides(quantities=QUANTITIES), f"Horizons ephemeris of {designation}")
    except (ValueError, requests.RequestException, ConnectionError) as exc:
        raise classify_horizons_error(exc) from exc


def ephemeris_path(
    target: Target, start_jd: float, stop_jd: float, step: str, root: str | os.PathLike | None
) -> Path:
    return asteroid_dir(target.dir_name, root) / "ephemerides" / f"horizons_{start_jd:.5f}_{stop_jd:.5f}_{step}.ecsv"


def load_ephemeris(
    target: Target,
    start_jd: float,
    stop_jd: float,
    step: str = "1h",
    root: str | os.PathLike | None = None,
    refresh: bool = False,
    fetch: HorizonsFetch = horizons_fetch,
) -> Ephemeris:
    """Ephemeris from the cache, or from Horizons when absent or ``refresh`` is set."""
    path = ephemeris_path(target, start_jd, stop_jd, step, root)
    if not refresh:
        table = read_table(path) if path.exists() else read_legacy_fits(path)
        if table is not None:
            if not path.exists():
                write_table(path, table)
            return Ephemeris.from_table(table)
    table = fetch(target.designation, start_jd, stop_jd, step)
    plain = Table(
        {
            name: np.ma.filled(np.ma.asarray(table[name], dtype=np.float64), np.nan)
            for name in COLUMNS
            if name in table.colnames
        }
    )
    write_table(path, plain, {"DESIG": target.designation, "LOCATION": HORIZONS_TESS, "STEP": step,
                              "QUERYDAT": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    return Ephemeris.from_table(plain)
