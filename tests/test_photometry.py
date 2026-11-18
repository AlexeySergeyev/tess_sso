import numpy as np
import pytest

from conftest import GaussianPrf
from tess_sso.photometry import (
    GroupData,
    Photometry,
    aperture_photometry,
    background_variance,
    background_variance_parts,
    fit_offset,
    group_images,
    groups_path,
    joint_prf_fit,
    prf_weights,
    make_groups,
    member_weights,
    prf_fraction,
    prf_shifts,
    segment_offset,
    star_variance,
    variability_impact,
    weighted_mean,
    white_excess,
    write_photometry,
)
from tess_sso.storage.ecsv_io import read_table

CADENCE = 200 / 86400
PRF = GaussianPrf(sigma=0.9)
SIZE = 11


def test_groups_by_duration_respect_segments_and_rejections():
    times = np.arange(20) * CADENCE
    segments = np.array([0] * 12 + [1] * 8)
    accepted = np.ones(20, bool)
    accepted[3] = False
    groups = make_groups(times, segments, accepted, CADENCE, duration_days=5 * CADENCE)
    assert [g.frames.tolist() for g in groups] == [[0, 1, 2, 4], [5, 6, 7, 8, 9], [10, 11], [12, 13, 14, 15, 16], [17, 18, 19]]
    assert [g.segment for g in groups] == [0, 0, 0, 1, 1]
    assert groups[0].coverage == pytest.approx(0.8)
    assert [g.incomplete for g in groups] == [False, False, True, False, True]


def test_groups_by_count_and_single_exposures():
    times = np.arange(7) * CADENCE
    seg = np.zeros(7, int)
    ok = np.ones(7, bool)
    assert [len(g.frames) for g in make_groups(times, seg, ok, CADENCE, count=3)] == [3, 3, 1]
    singles = make_groups(times, seg, ok, CADENCE)
    assert len(singles) == 7 and all(g.expected == 1 and not g.incomplete for g in singles)
    with pytest.raises(ValueError):
        make_groups(times, seg, ok, CADENCE, duration_days=1.0, count=2)


def _prf_stack(nexp, dr=0.0, dc=0.0):
    return np.stack([
        PRF.evaluate(np.array([6.0 + 0.1 * t + dr]), np.array([6.0 - 0.05 * t + dc]), (1, 1), (SIZE, SIZE))[0]
        for t in range(nexp)
    ])


def group(nexp=4, flux=1000.0, sigma=3.0, offset=(0.0, 0.0), seed=0, background=0.0):
    rng = np.random.default_rng(seed)
    prf = _prf_stack(nexp)
    h = 0.05
    prf_dr = (_prf_stack(nexp, dr=h) - _prf_stack(nexp, dr=-h)) / (2 * h)
    prf_dc = (_prf_stack(nexp, dc=h) - _prf_stack(nexp, dc=-h)) / (2 * h)
    truth = _prf_stack(nexp, *offset)
    data = flux * truth + background + rng.normal(0, sigma, prf.shape)
    return GroupData(np.arange(nexp), np.ones((nexp, 2), int), data, np.full(prf.shape, 1 / sigma**2),
                     np.full(prf.shape, sigma), prf, prf_dr, prf_dc)


def test_joint_fit_is_unbiased_with_calibrated_errors():
    pulls = []
    for seed in range(200):
        flux, err, a, chi2 = joint_prf_fit(group(seed=seed))
        pulls.append((flux - 1000.0) / err)
    assert abs(np.mean(pulls)) < 0.2 and 0.85 < np.std(pulls) < 1.15
    assert 0.8 < chi2 < 1.3


def test_one_exposure_group_equals_a_single_exposure_fit():
    g = group(nexp=1)
    flux, err, a, _ = joint_prf_fit(g)
    manual = (g.weight * g.prf * g.data).sum() / (g.weight * g.prf**2).sum()
    assert flux == pytest.approx(manual)


def test_aperture_masked_pixel_lowers_the_fraction_not_the_flux():
    clean = aperture_photometry(group(sigma=1e-6))
    g = group(sigma=1e-6)
    g.weight[:, 6, 6] = 0.0  # the brightest pixel is masked in every exposure
    masked = aperture_photometry(g)
    assert masked.fraction < clean.fraction and masked.masked_fraction > 0
    assert masked.flux == pytest.approx(1000.0, rel=0.01)
    assert clean.flux == pytest.approx(1000.0, rel=0.01)
    assert masked.coverage[6, 6] == 0 and clean.coverage[6, 6] == 4


def test_position_offset():
    dr, dc, _, _ = fit_offset(group(sigma=1e-6, offset=(0.1, -0.08)))
    assert dr == pytest.approx(0.1, abs=0.01) and dc == pytest.approx(-0.08, abs=0.01)
    assert fit_offset(group(sigma=1e-6))[0] == pytest.approx(0.0, abs=1e-6)
    no_derivatives = group()
    no_derivatives.prf_dr = None
    assert np.isnan(fit_offset(no_derivatives)[0])


def test_position_offset_errors_are_calibrated():
    pulls = []
    for seed in range(200):
        dr, dc, dr_err, dc_err = fit_offset(group(flux=300.0, seed=seed, offset=(0.1, -0.08)))
        pulls += [(dr - 0.1) / dr_err, (dc + 0.08) / dc_err]
    assert abs(np.mean(pulls)) < 0.2 and 0.85 < np.std(pulls) < 1.15


def test_prf_fraction_counts_only_usable_pixels():
    g = group(sigma=1e-6)
    weights = member_weights(g)
    full = prf_fraction(g, weights)
    assert full == pytest.approx(float(g.prf[0].sum()), rel=0.02)
    g.weight[:, 6, 6] = 0.0
    assert prf_fraction(g, weights) == pytest.approx(full - float(np.sum(weights * g.prf[:, 6, 6])))


def test_stellar_error_adds_coherently():
    one = group(nexp=1, sigma=1e-6)
    two = group(nexp=2, sigma=1e-6)
    two.prf[1], two.data[1] = one.prf[0], one.data[0]
    two.prf[0] = one.prf[0]
    _, _, a1, _ = joint_prf_fit(one)
    _, _, a2, _ = joint_prf_fit(two)
    design = np.ones((1, 1, SIZE, SIZE)) * one.prf[0][None, None]
    cov = np.array([[4.0]])
    v1 = star_variance(a1, design, cov)
    v2 = star_variance(a2, np.concatenate([design, design]), cov)
    assert v2 == pytest.approx(v1)  # the same star error in both exposures does not average down


def test_background_variance_matches_the_model_for_one_pixel():
    from test_scattered_light import PIXELS, fit

    seg, _ = fit()
    a = np.zeros((1, 1, 1))
    a[0, 0, 0] = 1.0  # the estimate is just pixel 0 in frame 10
    pixels = PIXELS[:1][None]
    v = background_variance(a, pixels, np.array([10]), seg)
    assert v == pytest.approx(seg.variance(PIXELS[:1], np.array([10]))[0, 0], rel=1e-6)
    spatial, temporal = background_variance_parts(a, pixels, np.array([10]), seg)
    assert spatial + temporal == pytest.approx(v) and spatial > 0 and temporal > 0


def test_member_weights_and_output(tmp_path):
    g = group()
    w = member_weights(g)
    assert w.sum() == pytest.approx(1.0) and np.all(w > 0)
    phot = Photometry(tstart=np.arange(10.0), tstop=np.arange(10.0) + 0.1)
    assert len(phot.to_table()) == 0
    path = tmp_path / "photometry.ecsv"
    write_photometry(path, phot, {"APTHR": 0.02})
    assert read_table(path).meta["APTHR"] == 0.02
    assert groups_path(path) == tmp_path / "photometry_groups.ecsv"
    assert read_table(groups_path(path)).colnames == ["group", "frame", "weight"]


def test_weighted_mean_skips_missing_values():
    assert weighted_mean([1.0, np.nan, 3.0], np.array([0.25, 0.5, 0.25])) == pytest.approx(2.0)
    assert weighted_mean([np.nan, 5.0], np.array([1.0, 0.0])) != weighted_mean([np.nan, 5.0], np.array([1.0, 0.0]))


def test_group_images():
    g = group(sigma=1e-3)
    flux, *_ = joint_prf_fit(g)
    ones = np.ones_like(g.data)
    g.weight[:, 0, 0] = 0.0  # a corner pixel masked in every exposure
    images = group_images(g, flux, background=5 * ones, background_sigma=0.5 * ones, stars=2 * ones)
    assert images["COVERAGE"][0, 0] == 0 and images["COVERAGE"][6, 6] == 4
    assert np.isnan(images["FLUX"][0, 0]) and np.isnan(images["RESIDUAL"][0, 0])
    used = images["COVERAGE"] > 0
    np.testing.assert_allclose(images["FLUX_BKG"][used], 5.0)
    np.testing.assert_allclose(images["FLUX_BKG_ERR"][used], 0.5)  # coherent: not averaged down
    np.testing.assert_allclose(images["STAR_MODEL"][used], 2.0)
    # FLUX keeps the stars; the models account for it down to the noise
    np.testing.assert_allclose((images["FLUX"] - images["STAR_MODEL"] - images["ASTEROID_MODEL"])[used],
                               images["RESIDUAL"][used], atol=1e-4)
    assert np.nanmax(np.abs(images["RESIDUAL"])) < 0.01
    assert images["ASTEROID_MODEL"][used].sum() == pytest.approx(1000.0, rel=0.02)


def test_segment_offset_recovers_a_constant_offset():
    rng = np.random.default_rng(4)
    errors = np.full((40, 2), 0.01)
    offsets = np.array([0.07, -0.06]) + rng.normal(0, 0.01, (40, 2))
    so = segment_offset(3, offsets, errors)
    assert so.segment == 3 and so.applied and so.n_groups == 40
    assert so.row == pytest.approx(0.07, abs=0.005) and so.col == pytest.approx(-0.06, abs=0.005)
    assert so.row_err == pytest.approx(0.01 / np.sqrt(40), rel=0.3)


def test_segment_offset_of_noise_is_not_applied():
    rng = np.random.default_rng(5)
    so = segment_offset(0, rng.normal(0, 0.2, (30, 2)), np.full((30, 2), 0.2))
    assert not so.applied


def test_segment_offset_clips_outliers_and_inflates_errors():
    rng = np.random.default_rng(6)
    offsets = np.array([0.05, 0.05]) + rng.normal(0, 0.03, (50, 2))  # scatter 3x the stated errors
    offsets[0] = (2.0, -2.0)  # a group on a bright star
    so = segment_offset(0, offsets, np.full((50, 2), 0.01))
    assert so.n_groups == 49
    assert so.row_err == pytest.approx(0.03 / np.sqrt(49), rel=0.3)  # from the scatter, not the stated errors


def test_segment_offset_without_measurements():
    so = segment_offset(1, np.full((3, 2), np.nan), np.full((3, 2), np.nan))
    assert not so.applied and so.n_groups == 0 and so.row == 0.0


def test_prf_shifts_by_mode():
    segments = np.array([0, 0, 1, 1])
    offsets = np.array([[0.1, 0.0], [0.1, 0.0], [0.5, 0.5], [np.nan, np.nan]])
    errors = np.array([[0.01, 0.01], [0.01, 0.01], [1.0, 1.0], [np.nan, np.nan]])
    shifts, per_segment = prf_shifts("segment", segments, offsets, errors)
    np.testing.assert_allclose(shifts, [[0.1, 0.0], [0.1, 0.0], [0, 0], [0, 0]])  # segment 1 is not significant
    assert [so.applied for so in per_segment] == [True, False]
    shifts, _ = prf_shifts("group", segments, offsets, errors)
    np.testing.assert_allclose(shifts, [[0.1, 0.0], [0.1, 0.0], [0.5, 0.5], [0, 0]])
    shifts, _ = prf_shifts("predicted", segments, offsets, errors)
    assert not shifts.any()
    with pytest.raises(ValueError):
        prf_shifts("fitted", segments, offsets, errors)


def test_variability_impact_projects_the_step_on_the_flux():
    g = group(nexp=2, sigma=1e-3)
    _, _, a, _ = joint_prf_fit(g)
    # star 0 sits exactly on the asteroid, star 1 far away in the corner, star 2 has no step
    far = np.zeros_like(g.prf)
    far[:, 0, 0] = 1.0
    design = np.stack([g.prf, far, g.prf], axis=1)  # (nexp, ncomp, size, size)
    impact = variability_impact(a, design, np.array([40.0, 40.0, np.nan]))
    assert impact[0] == pytest.approx(40.0, rel=1e-6)  # all of it goes into the flux
    assert impact[1] < 0.5 and impact[2] == 0.0


CAD = 200 / 86400


def test_white_excess_is_zero_when_the_errors_explain_the_scatter():
    rng = np.random.default_rng(8)
    t = np.arange(2000) * CAD
    signal = 3000 * (1 + 0.15 * np.sin(2 * np.pi * t / (8.7 / 24)))  # a fast rotator: 8.7 h, +-15 %
    err = np.full(t.size, 7.0)
    excess = white_excess(t, signal + rng.normal(0, 7.0, t.size), err, CAD)
    assert np.hypot(7.0, excess) < 7.0 * 1.06  # the rotation does not leak in; noise changes the error < 6 %


def test_white_excess_recovers_extra_noise_and_ignores_a_short_event():
    rng = np.random.default_rng(9)
    t = np.arange(3000) * CAD
    flux = 3000 + rng.normal(0, 5.0, t.size) + rng.normal(0, 4.0, t.size)  # stated 5, true 5 (+) 4
    flux[1000:1010] -= 300  # a 30-minute dip
    assert white_excess(t, flux, np.full(t.size, 5.0), CAD) == pytest.approx(4.0, rel=0.15)


def test_white_excess_needs_consecutive_exposures():
    t = np.arange(20) * 10 * CAD  # every tenth exposure: no triples
    assert np.isnan(white_excess(t, np.ones(20), np.ones(20), CAD))


def test_random_and_coherent_errors():
    from tess_sso.photometry import ApertureResult, Group, GroupResult

    ap = ApertureResult(1.0, 3.0, 0.9, 0.0, np.zeros((1, 1, 1)), np.zeros((1, 1)), np.zeros((1, 1)), np.ones((1, 1)), np.ones((1, 1), bool))
    r = GroupResult(Group(0, 0, np.array([0]), 1, False), 0.0, np.ones(1), 10.0, err_pix=3.0, err_star=12.0, err_bkg=4.0,
                    red_chi2=1.0, offset=(0.0, 0.0), aperture=ap, ap_err_star=12.0, ap_err_bkg=4.0,
                    err_bkg_sys=5.0, err_excess=12.0)
    assert r.flux_err == pytest.approx(13.0)  # sqrt(3^2 + 4^2 + 12^2): pixel, temporal background, excess
    assert r.flux_err_sys == pytest.approx(13.0)  # sqrt(12^2 + 5^2): stars, spatial background


def test_a_local_background_in_the_fit_absorbs_a_level_the_models_left():
    plain = [joint_prf_fit(group(seed=seed, background=4.0)) for seed in range(100)]
    local = [joint_prf_fit(group(seed=seed, background=4.0), local_background=True) for seed in range(100)]
    # a level of 4 e-/s per pixel over the box goes into the flux without the term
    assert np.mean([f for f, *_ in plain]) - 1000.0 > 30
    pulls = [(f - 1000.0) / e for f, e, _, _ in local]
    assert abs(np.mean(pulls)) < 0.25 and 0.85 < np.std(pulls) < 1.15
    flux, err, a, chi2 = local[0]
    assert a.sum() == pytest.approx(0.0, abs=1e-9)  # a level projects to nothing
    assert err > plain[0][1]  # at a price in noise
    g = group()
    assert (prf_weights(g) * g.prf).sum() == pytest.approx(1.0)  # the channels still measure what lies under it
