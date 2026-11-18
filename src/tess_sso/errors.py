"""Exceptions shared across tess_sso."""

from __future__ import annotations


class TessSsoError(Exception):
    """Base class for tess_sso errors."""


class UnknownDesignationError(TessSsoError, LookupError):
    """No solar-system object matches the given name or designation."""


class AmbiguousDesignationError(TessSsoError, LookupError):
    """The name matches several objects; ``candidates`` lists them."""

    def __init__(self, query: str, candidates: list[str]):
        self.query = query
        self.candidates = candidates
        shown = ", ".join(candidates[:10]) + (" ..." if len(candidates) > 10 else "")
        super().__init__(f"{query!r} matches {len(candidates)} objects: {shown}")


class ServiceError(TessSsoError, ConnectionError):
    """A remote service (JPL, MAST) failed or could not be reached."""
