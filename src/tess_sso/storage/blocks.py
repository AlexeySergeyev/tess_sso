"""Row blocks and download runs.

Pixel files group pixels by blocks of ``BLOCK_ROWS`` CCD rows. Rows and
columns are 1-based, as in TESS and tesscube.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Iterable

BLOCK_ROWS = 16
# Full cube size (science and non-science pixels), as in the MAST cloud cubes.
CCD_NROWS = 2078
CCD_NCOLUMNS = 2136

_NAME_RE = re.compile(r"^r(\d{4})-(\d{4})_(\d+)\.fits$")


def block_start(row: int, block_rows: int = BLOCK_ROWS) -> int:
    """First row of the block containing ``row``."""
    return (row - 1) // block_rows * block_rows + 1


def check_pixels(pixels: Iterable[tuple[int, int]], ccd_shape: tuple[int, int] = (CCD_NROWS, CCD_NCOLUMNS)) -> None:
    nrows, ncols = ccd_shape
    for row, col in pixels:
        if not (1 <= row <= nrows and 1 <= col <= ncols):
            raise ValueError(f"pixel ({row}, {col}) is outside the CCD (1..{nrows}, 1..{ncols})")


def by_block(pixels: Iterable[tuple[int, int]], block_rows: int = BLOCK_ROWS) -> dict[int, list[tuple[int, int]]]:
    """Pixels grouped by block start row, each group sorted by (row, col)."""
    groups: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for row, col in pixels:
        groups[block_start(row, block_rows)].append((row, col))
    return {start: sorted(set(group)) for start, group in sorted(groups.items())}


def runs_for(pixels: Iterable[tuple[int, int]]) -> list[dict[str, int]]:
    """tesscube download runs covering exactly ``pixels``: one per contiguous column range per row."""
    runs: list[dict[str, int]] = []
    for row, col in sorted(set(pixels)):
        last = runs[-1] if runs else None
        if last and last["row"] == row and last["start_column"] + last["ncolumns"] == col:
            last["ncolumns"] += 1
        else:
            runs.append({"row": row, "start_column": col, "ncolumns": 1})
    return runs


def file_name(start: int, part: int, block_rows: int = BLOCK_ROWS) -> str:
    """``r0705-0720_1.fits``: rows 705-720, first file of that block."""
    return f"r{start:04d}-{start + block_rows - 1:04d}_{part}.fits"


def parse_file_name(name: str) -> tuple[int, int] | None:
    """(block start row, part) of a pixel file name; ``None`` for anything else."""
    match = _NAME_RE.match(name)
    return (int(match.group(1)), int(match.group(3))) if match else None
