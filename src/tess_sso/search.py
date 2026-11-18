"""Find the TESS observations of an asteroid and summarize them in a table.

MAST's TESScut gives the preliminary list of sector-camera-CCD visits.
Horizons gives the trajectory seen from TESS; tesswcs checks that the
trajectory really falls on each listed CCD and gives the time on it.
"""

from __future__ import annotations

import os
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import requests
from astropy.table import Table

from tess_sso.storage.ecsv_io import read_legacy_fits, read_table, write_table
from tess_sso.storage.store import PixelStore
from tess_sso.designation import Target, load_target, resolve_designation
from tess_sso.ephemeris import HorizonsFetch, horizons_fetch, load_ephemeris
from tess_sso.errors import ServiceError, UnknownDesignationError
from tess_sso.network import with_retries
from tess_sso.paths import asteroid_dir, observation_dir

BTJD0 = 2457000.0
# Science pixels of a TESS CCD, 1-indexed.
SCIENCE_ROWS = (1, 2048)
SCIENCE_COLUMNS = (45, 2092)

Observation = tuple[int, int, int]  # sector, camera, ccd
SectorsFetch = Callable[[str], list[Observation]]

COLUMNS = {
    "sector": int,
    "camera": int,
    "ccd": int,
    "start_btjd": float,  # first time on the CCD's science area (TDB, approximate BTJD)
    "stop_btjd": float,
    "hours_on_ccd": float,
    "cadence_s": int,
    "v_min": float,  # predicted brightness range on the CCD
    "v_max": float,
    "v_band": str,
    "v_known": bool,
    "on_ccd": bool,  # False: TESScut lists it, but the trajectory misses the CCD
    "local_pixels": int,  # pixels already stored for this asteroid on this CCD
}


def cadence_seconds(sector: int) -> int:
    """FFI cadence: 1800 s in sectors 1-26, 600 s in 27-55, 200 s from 56."""
    if sector <= 26:
        return 1800
    if sector <= 55:
        return 600
    return 200


def sector_jd_range(sector: int) -> tuple[float, float]:
    """Start and end of a sector (JD) from the tesswcs pointing table."""
    from tesswcs import pointings

    row = pointings[pointings["Sector"] == sector]
    if len(row) == 0:
        raise ValueError(f"sector {sector} is not in the tesswcs pointing table")
    return float(row["Start"][0]), float(row["End"][0])


def trajectory_on_ccd(
    sector: int, camera: int, ccd: int, ra: np.ndarray, dec: np.ndarray
) -> np.ndarray:
    """Which trajectory points fall on the science area of the CCD (tesswcs WCS)."""
    from tesswcs import WCS

    wcs = WCS.from_sector(sector, camera, ccd)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        x, y = wcs.world_to_pixel_values(np.asarray(ra), np.asarray(dec))
    row, col = np.asarray(y) + 1, np.asarray(x) + 1
    return (
        (row >= SCIENCE_ROWS[0])
        & (row <= SCIENCE_ROWS[1])
        & (col >= SCIENCE_COLUMNS[0])
        & (col <= SCIENCE_COLUMNS[1])
    )


def tesscut_sectors(designation: str) -> list[Observation]:
    """Sector-camera-CCD visits from MAST TESScut; empty when there are none."""
    from astroquery.exceptions import InvalidQueryError, NoResultsWarning, RemoteServiceError
    from astroquery.mast import Tesscut

    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            # mt_type="smallbody" stops MAST from matching a planetary moon first.
            table = with_retries(
                lambda: Tesscut.get_sectors(objectname=designation, moving_target=True, mt_type="smallbody"),
                f"TESScut sectors of {designation}",
            )
    except InvalidQueryError as exc:
        raise UnknownDesignationError(f"MAST does not know the moving target {designation!r}") from exc
    except ValueError as exc:
        # An empty answer arrives as a ValueError from building a zero-row table.
        if any(issubclass(w.category, NoResultsWarning) for w in caught) or (
            "must match number of columns" in str(exc)
        ):
            return []
        raise ServiceError(f"TESScut query for {designation!r} failed: {exc}") from exc
    except (requests.RequestException, RemoteServiceError, ConnectionError) as exc:
        raise ServiceError(f"TESScut query for {designation!r} failed: {exc}") from exc
    if table is None:
        return []
    return sorted((int(r["sector"]), int(r["camera"]), int(r["ccd"])) for r in table)


def observations_path(target: Target, root: str | os.PathLike | None) -> Path:
    return asteroid_dir(target.dir_name, root) / "observations.ecsv"


def load_sectors(
    target: Target,
    root: str | os.PathLike | None = None,
    refresh: bool = False,
    fetch: SectorsFetch = tesscut_sectors,
) -> list[Observation]:
    """Cached TESScut visit list; ``refresh`` queries MAST again."""
    path = observations_path(target, root)
    if not refresh:
        table = read_table(path) if path.exists() else read_legacy_fits(path)
        if table is not None:
            if not path.exists():
                write_table(path, table)
            return [(int(s), int(c), int(d)) for s, c, d in zip(table["sector"], table["camera"], table["ccd"])]
    visits = fetch(target.designation)
    table = Table(
        {
            "sector": np.array([v[0] for v in visits], dtype=np.int32),
            "camera": np.array([v[1] for v in visits], dtype=np.int32),
            "ccd": np.array([v[2] for v in visits], dtype=np.int32),
        }
    )
    write_table(path, table, {"DESIG": target.designation,
                              "QUERYDAT": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    return visits


def empty_table() -> Table:
    return Table(names=list(COLUMNS), dtype=list(COLUMNS.values()))


def find_observations(
    name: str | int,
    root: str | os.PathLike | None = None,
    refresh: bool = False,
    step: str = "1h",
    resolve: Callable[[str | int], Target] = resolve_designation,
    sectors: SectorsFetch = tesscut_sectors,
    horizons: HorizonsFetch = horizons_fetch,
) -> Table:
    """One row per sector-camera-CCD visit. Downloads no images.

    ``table.meta`` carries the resolved designation and full name. An
    asteroid without TESS coverage gives an empty table.
    """
    target = load_target(name, root, resolve)
    visits = load_sectors(target, root, refresh, sectors)
    table = empty_table()
    table.meta["DESIG"] = target.designation
    table.meta["FULLNAME"] = target.fullname
    for sector in sorted({v[0] for v in visits}):
        start, stop = sector_jd_range(sector)
        eph = load_ephemeris(target, start, stop, step, root, refresh, horizons)
        step_hours = float(np.median(np.diff(eph.time_utc))) * 24 if len(eph.time_utc) > 1 else 0.0
        for _, camera, ccd in (v for v in visits if v[0] == sector):
            on = trajectory_on_ccd(sector, camera, ccd, eph.ra, eph.dec)
            v_on = eph.v[on]
            v_known = bool(np.isfinite(v_on).any())
            table.add_row(
                {
                    "sector": sector,
                    "camera": camera,
                    "ccd": ccd,
                    "start_btjd": eph.time_tdb[on].min() - BTJD0 if on.any() else np.nan,
                    "stop_btjd": eph.time_tdb[on].max() - BTJD0 if on.any() else np.nan,
                    "hours_on_ccd": on.sum() * step_hours,
                    "cadence_s": cadence_seconds(sector),
                    "v_min": np.nanmin(v_on) if v_known else np.nan,
                    "v_max": np.nanmax(v_on) if v_known else np.nan,
                    "v_band": "V",
                    "v_known": v_known,
                    "on_ccd": bool(on.any()),
                    "local_pixels": PixelStore(observation_dir(target.dir_name, sector, camera, ccd, root), sector, camera, ccd).n_pixels(),
                }
            )
    return table


def filter_by_brightness(table: Table, v_limit: float, keep_unknown: bool = True) -> Table:
    """Rows whose predicted V reaches ``v_limit`` or brighter; unknown V kept by default."""
    keep = np.asarray(table["v_known"]) & (np.asarray(table["v_min"]) <= v_limit)
    if keep_unknown:
        keep |= ~np.asarray(table["v_known"])
    return table[keep]


def format_observations(table: Table) -> str:
    """Short text summary with the resolved designation first."""
    lines = [f"{table.meta.get('FULLNAME', table.meta.get('DESIG', '?'))}: {len(table)} observation(s)"]
    for row in table:
        v = f"V {row['v_min']:.1f}-{row['v_max']:.1f}" if row["v_known"] else "V unknown"
        where = "" if row["on_ccd"] else "  [trajectory misses the CCD]"
        lines.append(
            f"  s{row['sector']:04d}-{row['camera']}-{row['ccd']}  {row['hours_on_ccd']:6.1f} h  "
            f"{row['cadence_s']:4d} s  {v}  stored px {row['local_pixels']}{where}"
        )
    return "\n".join(lines)
