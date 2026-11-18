import numpy as np
import pytest

from tess_sso.timing import (
    LIGHT_TIME_AU_D,
    BarycentricFallbackWarning,
    asteroid_timecorr,
    emission_time,
)


def test_light_time_of_one_au():
    assert LIGHT_TIME_AU_D * 86400 == pytest.approx(499.004784, abs=1e-6)


def test_emission_time_removes_the_correction_and_the_light_time():
    time = np.array([3235.0, 3235.1])
    got = emission_time(time, np.array([0.004, 0.005]), np.array([1.8, 1.9]))
    np.testing.assert_allclose(got, time - [0.004, 0.005] - np.array([1.8, 1.9]) * LIGHT_TIME_AU_D, rtol=0, atol=1e-12)
    assert got.dtype == np.float64


def test_correction_gets_spacecraft_time_as_tdb_jd():
    seen = {}

    def correction(t_jd, ra, dec):
        seen["t"] = t_jd
        return np.full(len(t_jd), 0.004)

    tc, source = asteroid_timecorr(np.array([3235.0]), np.array([10.0]), np.array([5.0]), np.array([0.003]), correction)
    assert seen["t"][0] == pytest.approx(2460235.0)
    assert tc[0] == 0.004 and source == "lkspacecraft"


def test_frames_without_a_position_keep_the_spoc_correction():
    def correction(t_jd, ra, dec):
        return np.where(np.isfinite(ra), 0.004, np.nan)

    tc, source = asteroid_timecorr(np.zeros(3), np.array([1.0, np.nan, 1.0]), np.zeros(3), np.full(3, 0.003), correction)
    np.testing.assert_array_equal(tc, [0.004, 0.003, 0.004])
    assert source == "lkspacecraft"


def test_failure_falls_back_to_spoc_with_a_warning():
    def correction(t_jd, ra, dec):
        raise RuntimeError("no kernels")

    with pytest.warns(BarycentricFallbackWarning, match="no kernels"):
        tc, source = asteroid_timecorr(np.zeros(2), np.zeros(2), np.zeros(2), np.array([0.003, 0.0031]), correction)
    np.testing.assert_array_equal(tc, [0.003, 0.0031])
    assert source == "SPOC"
