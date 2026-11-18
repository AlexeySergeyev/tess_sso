import numpy as np
import pytest

from tess_sso.quality import SPOC_BAD_BITMASK, SsoQuality, describe, spoc_bad


def test_spoc_bitmask_value():
    assert SPOC_BAD_BITMASK == 17087


def test_spoc_bitmask_matches_lightkurve():
    lk_utils = pytest.importorskip("lightkurve.utils")
    assert SPOC_BAD_BITMASK == lk_utils.TessQualityFlags.DEFAULT_BITMASK


def test_spoc_bad():
    # 32 = momentum dump (bad), 4096 = Straylight2 (not bad by default), 0 = clean
    assert spoc_bad(np.array([0, 32, 4096, 4096 | 1])).tolist() == [False, True, False, True]


def test_sso_bits_are_distinct_single_bits_in_int32():
    values = [int(flag) for flag in SsoQuality]
    assert len(set(values)) == len(values)
    assert all(v & (v - 1) == 0 for v in values)
    assert max(values) < 2**31


def test_describe():
    value = SsoQuality.WCS_FALLBACK | SsoQuality.SATURATED_STAR
    assert describe(value) == ["WCS_FALLBACK", "SATURATED_STAR"]
    assert describe(0) == []
