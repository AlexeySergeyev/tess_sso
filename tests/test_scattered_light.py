import numpy as np
import pytest

from tess_sso.scattered_light import (
    FewBackgroundPixelsWarning,
    ScatteredLightModel,
    SplineBasis,
    fit_segment,
    read_model,
    warn_if_poorly_constrained,
    write_model,
)

ROWS, COLS = np.mgrid[100:140, 200:260]
PIXELS = np.column_stack([ROWS.ravel(), COLS.ravel()])
NT = 200
T = np.linspace(0, 1, NT)
A = 1 + 0.5 * T**2
B = np.sin(6 * T)


def truth(pixels=PIXELS):
    r, c = pixels[:, 0] - 100.0, pixels[:, 1] - 200.0
    return A[:, None] * (100 + 0.5 * r + 0.3 * c)[None] + B[:, None] * (10 * np.sin(c / 15))[None]


def masked_pixels():
    r, c = PIXELS[:, 0] - 100.0, PIXELS[:, 1] - 200.0
    masked = np.abs(r - c * 40 / 60) < 2  # asteroid track
    for r0, c0 in [(110, 220), (125, 240), (130, 210)]:  # stars
        masked |= np.hypot(PIXELS[:, 0] - r0, PIXELS[:, 1] - c0) < 3
    return masked


def fit(seed=1, good=None, usable=None):
    rng = np.random.default_rng(seed)
    data = truth() + rng.normal(0, 1, (NT, len(PIXELS)))
    usable = ~masked_pixels() if usable is None else usable
    good = np.ones(NT, bool) if good is None else good
    return fit_segment(0, np.arange(NT), good, data[:, usable], PIXELS[usable], PIXELS), data


def test_basis_partition_of_unity_and_support():
    basis = SplineBasis.for_pixels(PIXELS)
    x = basis.design(PIXELS)
    np.testing.assert_allclose(x.sum(axis=1), 1.0, atol=1e-9)
    corner = SplineBasis.for_pixels(PIXELS[PIXELS[:, 0] < 105])
    assert len(corner.columns) < len(basis.columns)  # functions without support are dropped


def test_recovers_smooth_background_under_masks():
    seg, _ = fit()
    err = seg.evaluate(PIXELS) - truth()
    masked = masked_pixels()
    assert np.sqrt(np.mean(err[:, masked] ** 2)) < 0.2  # noise is 1.0 per pixel
    assert np.nanmedian(seg.frame_rms) == pytest.approx(1.0, abs=0.1)
    assert seg.static_rms < 0.2


def test_frames_left_out_of_the_pca_still_get_a_model():
    good = np.ones(NT, bool)
    good[50:60] = False
    seg, _ = fit(good=good)
    assert np.isfinite(seg.temporal).all()
    err = seg.evaluate(PIXELS, np.arange(50, 60)) - truth()[50:60]
    assert np.sqrt(np.mean(err**2)) < 0.2


def test_outlying_pixels_are_clipped():
    rng = np.random.default_rng(2)
    data = truth() + rng.normal(0, 1, (NT, len(PIXELS)))
    usable = ~masked_pixels()
    star = np.hypot(PIXELS[:, 0] - 105, PIXELS[:, 1] - 250) < 2  # an unmasked star
    data[:, star] += 300 * A[:, None]
    seg = fit_segment(0, np.arange(NT), np.ones(NT, bool), data[:, usable], PIXELS[usable], PIXELS)
    assert seg.n_clipped >= star.sum()
    err = seg.evaluate(PIXELS) - truth()
    assert np.sqrt(np.mean(err[:, masked_pixels()] ** 2)) < 0.3


def test_uncertainties_are_calibrated_within_a_factor_of_two():
    pulls = []
    for seed in range(3):
        seg, _ = fit(seed=seed)
        err = seg.evaluate(PIXELS) - truth()
        pulls.append(np.std(err[:, masked_pixels()] / np.sqrt(seg.variance(PIXELS)[:, masked_pixels()])))
    assert 0.5 < np.mean(pulls) < 2.0


def test_pixel_covariance():
    seg, _ = fit()
    cov = seg.pixel_covariance(PIXELS[:6], frame=10)
    np.testing.assert_allclose(cov, cov.T)
    assert np.linalg.eigvalsh(cov).min() > -1e-9
    np.testing.assert_allclose(np.diag(cov), seg.variance(PIXELS[:6], np.array([10]))[0], rtol=1e-8)
    assert cov[0, 1] > 0.5 * np.sqrt(cov[0, 0] * cov[1, 1])  # neighbours are strongly correlated


def test_gradient():
    seg, _ = fit()
    expected = np.hypot(0.5 * A[0], 0.3 * A[0] + B[0] * 10 / 15 * np.cos(30 / 15))
    assert seg.gradient(120.0, 230.0, np.array([0]))[0] == pytest.approx(expected, rel=0.05)


def test_poorly_constrained_region_warns():
    usable = ~masked_pixels() & (PIXELS[:, 1] < 225)  # background only on the left
    seg, _ = fit(usable=usable)
    right = PIXELS[PIXELS[:, 1] > 250]
    assert not seg.constrained(right).any()
    assert seg.constrained(PIXELS[PIXELS[:, 1] < 210]).all()
    with pytest.warns(FewBackgroundPixelsWarning, match="widen the strip"):
        warn_if_poorly_constrained(ScatteredLightModel([seg]), right, "test cutout")


def test_model_evaluates_by_absolute_frame_and_roundtrips(tmp_path):
    seg, _ = fit()
    seg.frames = np.arange(1000, 1000 + NT)
    model = ScatteredLightModel([seg])
    assert model.segment_of(1005) is seg
    values = model.evaluate(PIXELS[:10], [1000, 1199, 5000])
    assert np.isnan(values[2]).all() and np.isfinite(values[:2]).all()
    path = tmp_path / "scattered_light.fits"
    write_model(path, model, {"NCOMP": 8})
    again = read_model(path)
    np.testing.assert_allclose(again.evaluate(PIXELS[:10], [1000, 1199]), values[:2])
    np.testing.assert_allclose(again.segments[0].variance(PIXELS[:10]), seg.variance(PIXELS[:10]))


def test_residual_statistics_per_frame():
    rng = np.random.default_rng(3)
    usable = ~masked_pixels()
    data = truth() + rng.normal(0, 2.0, (NT, len(PIXELS)))
    err = np.full(data.shape, 2.0)
    seg = fit_segment(0, np.arange(NT), np.ones(NT, bool), data[:, usable], PIXELS[usable], PIXELS,
                      flux_err=err[:, usable])
    # Gaussian noise: MAD = 0.6745 sigma; chi2 per pixel ~ 1 with the right errors
    assert np.nanmedian(seg.frame_mad) == pytest.approx(0.6745 * 2.0, rel=0.1)
    assert np.nanmedian(seg.frame_chi2) == pytest.approx(1.0, abs=0.15)
    assert len(seg.fit_pixels) == seg.n_usable
    assert fit()[0].frame_chi2 is None  # no errors given


def test_usable_fraction():
    seg, _ = fit()
    assert seg.usable_fraction(seg.fit_pixels[:10]) == 1.0
    masked = PIXELS[masked_pixels()]
    assert seg.usable_fraction(masked[:10]) == 0.0
    assert seg.usable_fraction(np.concatenate([seg.fit_pixels[:3], masked[:1]])) == pytest.approx(0.75)
    assert np.isnan(seg.usable_fraction(np.empty((0, 2), int)))


def test_residual_statistics_roundtrip(tmp_path):
    seg, _ = fit()
    seg.frame_chi2 = np.linspace(0.5, 1.5, NT)
    path = tmp_path / "scattered_light.fits"
    write_model(path, ScatteredLightModel([seg]), {})
    again = read_model(path).segments[0]
    np.testing.assert_allclose(again.frame_mad, seg.frame_mad)
    np.testing.assert_allclose(again.frame_chi2, seg.frame_chi2)
    np.testing.assert_array_equal(again.fit_pixels, seg.fit_pixels)


def test_variance_terms_rebuild_the_variance():
    seg, _ = fit()
    per_component, fitted_sq = seg.variance_terms(PIXELS[:20])
    rebuilt = seg.temporal[7] ** 2 @ per_component + seg.frame_rms[7] ** 2 * fitted_sq
    np.testing.assert_allclose(rebuilt, seg.variance(PIXELS[:20], np.array([7]))[0])
