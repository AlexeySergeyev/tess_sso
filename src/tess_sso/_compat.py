"""Workarounds for dependencies, each removable once the dependency is fixed."""

from __future__ import annotations

TESS_MODELS_URL = "https://archive.stsci.edu/missions/tess/models/"

# Why the last listing came from the cache, until take_listing_fallback() reports it.
_listing_fallback: str | None = None


def take_listing_fallback() -> str | None:
    """The message of a listing rebuilt from the cache since the last call, or None.

    tesscube builds lkspacecraft's meta kernel with every warning silenced, so
    the caller re-issues this after importing it.
    """
    global _listing_fallback
    message, _listing_fallback = _listing_fallback, None
    return message


class _ListingPandas:
    """pandas for lkspacecraft.utils, with MAST's TESS models listing in the layout it parses.

    The listing now writes ``<a href="X"> X</a>``; lkspacecraft 1.3.1 reads
    the kernel names with the leading space, and building its meta kernel fails with
    ``KeyError: 'TESS_EPH_DEF_2018004_01.bsp'``, on importing tesscube as well.
    """

    def __init__(self, pandas):
        self._pandas = pandas

    def __getattr__(self, name):
        return getattr(self._pandas, name)

    def read_csv(self, path, *args, **kwargs):
        listing = isinstance(path, str) and path.startswith(TESS_MODELS_URL)
        try:
            df = self._pandas.read_csv(path, *args, **kwargs)
        except OSError as exc:  # urllib's HTTPError and URLError are OSErrors
            if not listing:
                raise
            df = self._cached_listing(exc)
        if listing and 0 in df:
            df[0] = df[0].astype(str).str.replace('"> ', '">', regex=False)
        return df

    def _cached_listing(self, exc: OSError):
        """The listing rebuilt from the kernels already in lkspacecraft's cache.

        MAST can refuse the directory listing (HTTP 403) while the kernels
        themselves are cached; lkspacecraft fetches the listing on every
        import, so without it tesscube cannot be imported at all.
        """
        global _listing_fallback
        from astropy.utils.data import cache_contents

        names = sorted(
            url[len(TESS_MODELS_URL):]
            for url in cache_contents(pkgname="lkspacecraft")
            if url.startswith(TESS_MODELS_URL) and "TESS_EPH_DEF" in url
        )
        if not names:
            raise exc
        # lkspacecraft 1.3.1 drops these two from the listing by name
        names = sorted(set(names) | {"TESS_EPH_DEF_2018004_01.bsp", "TESS_EPH_DEF_2018080_01.bsp"})
        _listing_fallback = (
            f"{TESS_MODELS_URL} unavailable ({exc}); using the {len(names)} TESS ephemeris "
            "kernels already cached. Kernels published since the last successful listing are missing."
        )
        return self._pandas.DataFrame({0: [f'<li><a href="{n}">{n}</a></li>' for n in names]})


def patch_lkspacecraft_listing() -> None:
    """Call before importing tesscube or creating a TESSSpacecraft; idempotent.

    Nothing to do from lkspacecraft 1.3.3 on, which reads the kernel names
    from the ``href`` attributes and no longer imports pandas in its utils.
    """
    try:
        import lkspacecraft.utils as lk_utils
    except ImportError:
        return
    if not hasattr(lk_utils, "pd"):
        return
    if not isinstance(lk_utils.pd, _ListingPandas):
        lk_utils.pd = _ListingPandas(lk_utils.pd)
