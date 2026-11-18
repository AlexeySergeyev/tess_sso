import pandas as pd
import pytest

import lkspacecraft.utils as lk_utils
from tess_sso._compat import TESS_MODELS_URL, patch_lkspacecraft_listing, take_listing_fallback

needs_pandas_listing = pytest.mark.skipif(
    not hasattr(lk_utils, "pd"), reason="lkspacecraft >= 1.3.3 parses the listing itself"
)

NEW = '<li><a href="TESS_EPH_DEF_2018004_01.bsp"> TESS_EPH_DEF_2018004_01.bsp</a></li>'
OLD = '<li><a href="TESS_EPH_DEF_2018080_01.bsp">TESS_EPH_DEF_2018080_01.bsp</a></li>'
PARENT = '<li><a href="/missions/tess/"> Parent Directory</a></li>'


def kernel_names(df):
    """lkspacecraft 1.3.1's parsing of the listing (get_tess_bsp)."""
    return [line.split(">")[2].split("<")[0] for line in df[0] if "TESS_EPH_DEF" in line]


@needs_pandas_listing
def test_the_listing_reaches_lkspacecraft_without_the_space(monkeypatch):
    monkeypatch.setattr(lk_utils, "pd", lk_utils.pd)  # restored after the test
    monkeypatch.setattr(pd, "read_csv", lambda *args, **kwargs: pd.DataFrame({0: [PARENT, NEW, OLD]}))
    patch_lkspacecraft_listing()
    patch_lkspacecraft_listing()  # idempotent
    names = kernel_names(lk_utils.pd.read_csv(TESS_MODELS_URL, header=None))
    assert names == ["TESS_EPH_DEF_2018004_01.bsp", "TESS_EPH_DEF_2018080_01.bsp"]


@needs_pandas_listing
def test_other_reads_are_untouched(monkeypatch):
    monkeypatch.setattr(lk_utils, "pd", lk_utils.pd)
    monkeypatch.setattr(pd, "read_csv", lambda *args, **kwargs: pd.DataFrame({0: [NEW]}))
    patch_lkspacecraft_listing()
    assert lk_utils.pd.read_csv("https://example.org/table.csv", header=None)[0][0] == NEW
    assert lk_utils.pd.DataFrame is pd.DataFrame


@needs_pandas_listing
def test_a_refused_listing_falls_back_to_the_cached_kernels(monkeypatch):
    import urllib.error

    import astropy.utils.data

    def refuse(*args, **kwargs):
        raise urllib.error.HTTPError(TESS_MODELS_URL, 403, "Forbidden", {}, None)

    cached = {
        TESS_MODELS_URL + "TESS_EPH_DEF_2025174_21.bsp": "/cache/a",
        TESS_MODELS_URL + "tess_20_year_long_predictive.bsp": "/cache/b",
        "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/naif0012.tls": "/cache/c",
    }
    monkeypatch.setattr(lk_utils, "pd", lk_utils.pd)
    monkeypatch.setattr(pd, "read_csv", refuse)
    monkeypatch.setattr(astropy.utils.data, "cache_contents", lambda pkgname=None: cached)
    patch_lkspacecraft_listing()
    names = kernel_names(lk_utils.pd.read_csv(TESS_MODELS_URL, header=None))
    assert "already cached" in take_listing_fallback()
    assert take_listing_fallback() is None  # reported once
    assert names == [
        "TESS_EPH_DEF_2018004_01.bsp",
        "TESS_EPH_DEF_2018080_01.bsp",
        "TESS_EPH_DEF_2025174_21.bsp",
    ]
    with pytest.raises(urllib.error.HTTPError):
        lk_utils.pd.read_csv("https://example.org/table.csv")


@needs_pandas_listing
def test_a_refused_listing_with_an_empty_cache_still_raises(monkeypatch):
    import urllib.error

    import astropy.utils.data

    def refuse(*args, **kwargs):
        raise urllib.error.HTTPError(TESS_MODELS_URL, 403, "Forbidden", {}, None)

    monkeypatch.setattr(lk_utils, "pd", lk_utils.pd)
    monkeypatch.setattr(pd, "read_csv", refuse)
    monkeypatch.setattr(astropy.utils.data, "cache_contents", lambda pkgname=None: {})
    patch_lkspacecraft_listing()
    with pytest.raises(urllib.error.HTTPError):
        lk_utils.pd.read_csv(TESS_MODELS_URL, header=None)


def test_nothing_to_patch_without_pandas_in_lkspacecraft(monkeypatch):
    import lkspacecraft.utils

    if hasattr(lkspacecraft.utils, "pd"):
        monkeypatch.delattr(lkspacecraft.utils, "pd")
    patch_lkspacecraft_listing()
    assert not hasattr(lkspacecraft.utils, "pd")
