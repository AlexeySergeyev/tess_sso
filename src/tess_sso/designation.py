"""Resolve an asteroid number, name, or designation with the JPL Small-Body Database."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import requests

from tess_sso.errors import AmbiguousDesignationError, ServiceError, UnknownDesignationError
from tess_sso.network import with_retries
from tess_sso.paths import reference_dir

SBDB_URL = "https://ssd-api.jpl.nasa.gov/sbdb.api"


@dataclass(frozen=True)
class Target:
    """A resolved asteroid."""

    designation: str  # JPL primary designation: "95", "722079", "2000 PH5"
    fullname: str  # e.g. "95 Arethusa (A867 WA)"

    @property
    def dir_name(self) -> str:
        """File-system form of the designation: spaces become underscores."""
        return "_".join(self.designation.split())


def normalize_name(name: str | int) -> str:
    """``"2000_PH5"`` and ``" 2000  PH5 "`` both become ``"2000 PH5"``."""
    return " ".join(str(name).replace("_", " ").split())


def resolve_designation(
    name: str | int, get: Callable = requests.get, timeout: float = 60
) -> Target:
    """The object JPL knows under ``name``; secondary designations map to the primary one."""
    query = normalize_name(name)
    if not query:
        raise UnknownDesignationError("empty designation")
    try:
        def lookup():
            response = get(SBDB_URL, params={"sstr": query}, timeout=timeout)
            response.raise_for_status()
            return response

        payload = with_retries(lookup, f"JPL SBDB lookup of {query!r}").json()
    except (requests.RequestException, ValueError) as exc:
        raise ServiceError(f"JPL SBDB lookup of {query!r} failed: {exc}") from exc
    if payload.get("code") == "300" or "list" in payload:
        candidates = [
            str(item.get("name") or item.get("pdes")) for item in payload.get("list", [])
        ]
        raise AmbiguousDesignationError(query, candidates)
    obj = payload.get("object") or {}
    if not obj.get("des"):
        raise UnknownDesignationError(
            f"JPL SBDB does not know {query!r}: {payload.get('message', 'no object returned')}"
        )
    return Target(str(obj["des"]), str(obj.get("fullname") or obj["des"]))


def designations_path(root: str | os.PathLike | None = None) -> Path:
    """``<data>/reference/designations.json``: every name looked up, with what JPL answered."""
    return reference_dir(root) / "designations.json"


def load_target(
    name: str | int,
    root: str | os.PathLike | None = None,
    resolve: Callable[[str | int], Target] = resolve_designation,
) -> Target:
    """The target for ``name``, from the names looked up before, else from ``resolve`` (JPL), then kept.

    A name is looked up once, so a stored asteroid is processed without the
    network. Ambiguous and unknown names are not kept.
    """
    query = normalize_name(name)
    path = designations_path(root)
    known = json.loads(path.read_text()) if path.exists() else {}
    if query in known:
        return Target(known[query]["designation"], known[query]["fullname"])
    target = resolve(name)
    known[query] = {"designation": target.designation, "fullname": target.fullname}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(json.dumps(known, indent=1, sort_keys=True, ensure_ascii=False) + "\n")
    os.replace(tmp, path)
    return target
