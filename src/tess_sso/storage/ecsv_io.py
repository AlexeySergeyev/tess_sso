"""Atomic ECSV writing and reading for stored tables.

ECSV is comma-separated text with a YAML header in ``#`` comments: the
header keeps each column's type and the table's metadata (query date,
release, settings), and ``pandas.read_csv(path, comment="#")`` reads the
values. Float64 values round-trip exactly.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

from astropy.table import Table

from tess_sso.storage.fits_io import CorruptFileError


def write_table(path: Path, table: Table, meta: Mapping[str, Any] | None = None) -> None:
    """Write ``table`` (with ``meta`` added to its metadata) to a temporary file, then rename it.

    A file at ``path`` is therefore always complete.
    """
    table = Table(table, copy=False)
    table.meta = {**table.meta, **(meta or {})}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    table.write(tmp, format="ascii.ecsv", delimiter=",", overwrite=True)
    os.replace(tmp, path)


def read_table(path: Path) -> Table:
    """Read an ECSV table, raising ``CorruptFileError`` when it cannot be parsed."""
    try:
        return Table.read(path, format="ascii.ecsv")
    except (ValueError, TypeError, IndexError, KeyError) as exc:
        raise CorruptFileError(f"{path}: {exc}") from exc


def read_legacy_fits(path: Path) -> Table | None:
    """The first table and primary-header cards of a cache written as FITS before ECSV, or None.

    Lets a cached query be kept, query date included, instead of repeated;
    the FITS file itself is left in place.
    """
    legacy = path.with_suffix(".fits")
    if not legacy.exists():
        return None
    from astropy.io import fits

    with fits.open(legacy) as hdul:
        table = Table(hdul[1].data)
        table.meta = {key: value for key, value in hdul[0].header.items()
                      if key not in ("SIMPLE", "BITPIX", "NAXIS", "EXTEND", "CHECKSUM", "DATASUM", "COMMENT", "HISTORY")}
    for name in table.colnames:  # FITS tables are big-endian; ECSV wants native types
        table[name] = table[name].astype(table[name].dtype.newbyteorder("="))
    return table
