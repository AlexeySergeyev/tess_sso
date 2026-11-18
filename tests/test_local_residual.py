import numpy as np
import pytest

from tess_sso.local_residual import MIN_POINTS, fit_pixel


def series(seed=0):
    rng = np.random.default_rng(seed)
    t = 3240.0 + np.arange(0, 2.4, 200 / 86400)
    drift = 3.0 + 2.0 * (t - 3241.2) - 1.5 * (t - 3241.2) ** 2  # what the models leave at this pixel
    resid = drift + rng.normal(0, 1.0, t.size)
    passage = np.abs(t - 3241.2) < 0.2
    resid[passage] += 40.0  # the asteroid on the pixel
    return t, drift, resid, np.ones(t.size), passage


def test_the_residual_under_the_asteroid_is_the_pixels_own_drift():
    t, drift, resid, err, passage = series()
    pred = fit_pixel(t, resid, err, ~passage, t[passage])
    assert np.abs(pred - drift[passage]).max() < 0.3  # not the asteroid's 40 e-/s


def test_one_sided_data_give_a_level_not_an_extrapolation():
    t, drift, resid, err, passage = series()
    after = t > 3241.4
    pred = fit_pixel(t, resid, err, after, t[passage])
    level = np.mean(resid[after])
    assert np.allclose(pred, pred[0]) and pred[0] == pytest.approx(level, abs=0.2)


def test_too_few_frames_keep_the_static_residual():
    t, drift, resid, err, passage = series()
    few = np.zeros(t.size, bool)
    few[: MIN_POINTS - 1] = True
    assert fit_pixel(t, resid, err, few, t[passage]) is None


def test_outliers_are_clipped():
    t, drift, resid, err, passage = series(seed=1)
    resid[~passage & (np.arange(t.size) % 50 == 0)] += 100.0  # cosmic rays
    pred = fit_pixel(t, resid, err, ~passage, t[passage])
    assert np.abs(pred - drift[passage]).max() < 0.4


def jittered(seed=0):
    """A pixel on a bright star: the stellar model's error follows the pointing, a 4.3 h sawtooth."""
    rng = np.random.default_rng(seed)
    t = 3240.0 + np.arange(0, 2.4, 600 / 86400)
    phase = ((t - t[0]) * 24 / 4.3) % 1
    shift = np.column_stack([0.02 * phase - 0.01, -0.005 * phase])  # (row, column) in pixels
    # the model's error at a ~1300 e-/s star: mostly linear in the shift (3486, sector 45: 20 e-/s over 0.16 px)
    star_error = -30.0 + 1500.0 * shift[:, 0] + 2000.0 * shift[:, 0] ** 2
    resid = star_error + rng.normal(0, 1.0, t.size)
    passage = np.abs(t - 3241.2) < 0.1
    resid[passage] += 40.0
    return t, star_error, resid, np.ones(t.size), passage, shift


def test_the_pointing_shift_explains_the_pulses_a_time_polynomial_misses():
    t, star_error, resid, err, passage, shift = jittered()
    in_time = fit_pixel(t, resid, err, ~passage, t[passage])
    with_pointing = fit_pixel(t, resid, err, ~passage, t[passage], shift, shift[passage])
    assert np.abs(in_time - star_error[passage]).max() > 5  # a polynomial in time sees only the level
    assert np.abs(with_pointing - star_error[passage]).max() < 0.6


def test_without_a_pointing_signal_the_time_polynomial_is_kept():
    t, drift, resid, err, passage = series()
    shift = np.random.default_rng(3).normal(0, 0.01, (t.size, 2))
    in_time = fit_pixel(t, resid, err, ~passage, t[passage])
    np.testing.assert_array_equal(fit_pixel(t, resid, err, ~passage, t[passage], shift, shift[passage]), in_time)


def test_the_pointing_is_not_extrapolated():
    t, star_error, resid, err, passage, shift = jittered()
    far = shift[passage] + 1.0  # far outside what the fit saw
    pred = fit_pixel(t, resid, err, ~passage, t[passage], shift, far)
    assert np.all(np.isfinite(pred)) and np.abs(pred).max() < np.abs(resid[~passage]).max() + 5


def test_a_slow_pointing_drift_is_left_to_the_time_polynomial():
    """Where the shift only drifts with time (35534, sector 55), it adds nothing but its own noise."""
    t, drift, resid, err, passage = series(seed=4)
    rng = np.random.default_rng(5)
    true = np.column_stack([0.30 + 0.01 * (t - t[0]), np.full(t.size, -0.02)])
    shift = true + rng.normal(0, 0.001, (t.size, 2))  # as measured from the stars, frame by frame
    resid = resid + 300.0 * (true[:, 0] - 0.30)  # a bright star following the real pointing: a drift in time
    pred = fit_pixel(t, resid, err, ~passage, t[passage], shift, shift[passage])
    smooth = fit_pixel(t, resid, err, ~passage, t[passage])
    np.testing.assert_array_equal(pred, smooth)  # the pointing terms add only noise here: not used
