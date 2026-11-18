"""Where tess_sso keeps its data.

The data live in the working project, found the way binaries finds its root:
walking up from the working directory to the first folder that contains
``data/``, ``pyproject.toml``, or ``.git``. The data directory is
``<project>/data/tess_sso/`` unless a root is passed explicitly or set in the
``TESS_SSO_DATA`` environment variable.

Inside it, each asteroid has its own folder with everything downloaded or
computed for it; ``reference/`` holds the few small tables that do not depend
on the asteroid.
"""

from __future__ import annotations

import os
from pathlib import Path

ENV_VAR = "TESS_SSO_DATA"
ROOT_MARKERS = ("data", "pyproject.toml", ".git")


def find_project_root(start: str | os.PathLike | None = None) -> Path:
    """First folder at or above ``start`` holding a root marker; ``start`` itself if none."""
    start = Path.cwd() if start is None else Path(start)
    start = start.resolve()
    for folder in (start, *start.parents):
        if any((folder / marker).exists() for marker in ROOT_MARKERS):
            return folder
    return start


def data_root(root: str | os.PathLike | None = None) -> Path:
    """Data directory: explicit ``root``, then ``$TESS_SSO_DATA``, then ``<project>/data/tess_sso``."""
    if root is not None:
        return Path(root)
    env = os.environ.get(ENV_VAR)
    if env:
        return Path(env)
    return find_project_root() / "data" / "tess_sso"


def observation_key(sector: int, camera: int, ccd: int) -> str:
    """Observation key, e.g. ``s0071-2-2``."""
    return f"s{sector:04d}-{camera}-{ccd}"


def asteroid_dir(slug: str, root: str | os.PathLike | None = None) -> Path:
    """Folder of one asteroid; ``slug`` is its designation with underscores."""
    return data_root(root) / slug


def observation_dir(
    slug: str, sector: int, camera: int, ccd: int, root: str | os.PathLike | None = None
) -> Path:
    return asteroid_dir(slug, root) / observation_key(sector, camera, ccd)


def reference_dir(root: str | os.PathLike | None = None) -> Path:
    """Small asteroid-independent tables: the TESS orbit times and lkprf models."""
    return data_root(root) / "reference"
