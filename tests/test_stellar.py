import numpy as np
import pytest

from conftest import GaussianPrf
from tess_sso.stellar import (
    CHI2_LIMIT,
    CORE_FRACTION,
    BlockCovariance,
    Components,
    StellarModel,
    Templates,
    fit_region,
    fit_segment,
    read_stellar,
    select_components,
    templates,
    write_stellar,
)

PRF = GaussianPrf(sigma=0.9)
ROWS, COLS = np.mgrid[1:31, 1:41]
PIXELS = np.column_stack([ROWS.ravel(), COLS.ravel()])
STAR_R = np.array([15.2, 8.3, 22.1, 12.7, 25.4])
STAR_C = np.array([20.1, 10.6, 30.2, 33.5, 8.8])
STAR_F = np.array([2000.0, 800.0, 1500.0, 600.0, 1200.0])
NF = 400
TIMES = np.arange(NF) * 200 / 86400
SIGMA = 2.0


def components(prior_scale=1.1):
    return Components(
        [np.array([r]) for r in STAR_R], [np.array([c]) for c in STAR_C], [np.array([1.0])] * 5,
        [np.array([i]) for i in range(5)], STAR_F * prior_scale,
    )


def scene(ast_rows, ast_cols, ast_flux, star_flux=None, shifts=None, seed=0):
    """Stars + a moving asteroid + noise; returns data, weight (asteroid footprint zeroed), footprint."""
    rng = np.random.default_rng(seed)
    nf = len(ast_rows)
    shifts = np.zeros((nf, 2)) if shifts is None else shifts
    star_flux = np.broadcast_to(STAR_F, (nf, 5)) if star_flux is None else star_flux
    stars = np.stack([
        (star_flux[t][:, None] * PRF.evaluate(STAR_R + shifts[t, 0], STAR_C + shifts[t, 1], (1, 1), (30, 40)).reshape(5, -1)).sum(0)
        for t in range(nf)
    ])
    ast_psf = np.stack([PRF.evaluate(np.array([ast_rows[t]]), np.array([ast_cols[t]]), (1, 1), (30, 40))[0].ravel() for t in range(nf)])
    footprint = ast_psf >= 0.01
    data = stars + ast_flux[:, None] * ast_psf + rng.normal(0, SIGMA, stars.shape)
    weight = np.where(footprint, 0.0, 1 / SIGMA**2)
    return data, weight, footprint


def fit(data, weight, footprint, comps=None):
    comps = comps or components()
    h = templates(PRF, comps, PIXELS)
    nf = len(data)
    times = np.arange(nf) * 200 / 86400
    return fit_segment(0, np.arange(nf), times, h, comps.prior_flux, data, weight, footprint), h


def test_fluxes_and_pointing_shifts_recovered_despite_a_variable_asteroid():
    shifts = np.column_stack([0.05 * np.sin(np.arange(NF) / 40), 0.03 * np.cos(np.arange(NF) / 25)])
    ast_flux = 500 * (1 + 0.3 * np.sin(np.arange(NF) / 30))  # varies while crossing star 0
    data, weight, footprint = scene(np.full(NF, 15.5), np.linspace(2, 38, NF), ast_flux, shifts=shifts)
    seg, _ = fit(data, weight, footprint)
    err = np.sqrt(seg.covariance.diag())
    assert np.all(np.abs(seg.fluxes - STAR_F) < 4 * err)
    assert np.all(np.abs(seg.fluxes - STAR_F) / STAR_F < 0.002)
    assert np.sqrt(np.mean((seg.shifts - shifts) ** 2)) < 0.01
    d = seg.diagnostics
    assert d.n_crossing[0] > 0 and d.fitted.all()
    assert not d.poorly_constrained.any() and not d.variable.any()
    assert np.all(d.chi2 < 1.5)


def test_star_hidden_most_of_the_time_falls_back_to_prior_and_is_flagged():
    rows = np.full(NF, STAR_R[2]); cols = np.full(NF, STAR_C[2])  # asteroid parked on star 2
    rows[:10] = 1.0; cols[:10] = 1.0  # except for 10 frames far away
    data, weight, footprint = scene(rows, cols, np.full(NF, 300.0))
    seg, _ = fit(data, weight, footprint)
    d = seg.diagnostics
    assert d.n_clean[2] < 20 and d.poorly_constrained[2]
    assert not d.poorly_constrained[[0, 1, 3, 4]].any()


def test_star_that_changes_across_the_crossing_is_flagged_variable():
    # 4.6 days, so the step's significance can be calibrated on other epochs of the same data
    nf = 2000
    cols = np.full(nf, 1.0)
    cols[900:1100] = np.linspace(2, 38, 200)  # the asteroid crosses star 0 around frame 1000
    star_flux = np.tile(STAR_F, (nf, 1))
    star_flux[1000:, 0] *= 1.02  # +2 % after the crossing
    data, weight, footprint = scene(np.full(nf, 15.5), cols, np.full(nf, 500.0), star_flux=star_flux)
    seg, _ = fit(data, weight, footprint)
    assert seg.diagnostics.variable[0] and seg.diagnostics.variability_z[0] > 3
    assert seg.diagnostics.variability_step[0] == pytest.approx(0.02 * STAR_F[0], rel=0.25)
    steady, _ = fit(*scene(np.full(nf, 15.5), cols, np.full(nf, 500.0), seed=1))
    assert not steady.diagnostics.variable.any()


def test_variability_step_skips_frames_with_a_poor_background():
    nf = 2000
    cols = np.full(nf, 1.0)
    cols[900:1100] = np.linspace(2, 38, 200)  # the asteroid crosses star 0 around frame 1000
    star_flux = np.tile(STAR_F, (nf, 1))
    star_flux[700:900, 0] *= 1.05  # a background misfit just before the crossing (the start of an orbit)
    data, weight, footprint = scene(np.full(nf, 15.5), cols, np.full(nf, 500.0), star_flux=star_flux)
    comps = components()
    h = templates(PRF, comps, PIXELS)
    times = np.arange(nf) * 200 / 86400
    quiet = np.ones(nf, bool)
    quiet[700:900] = False
    plain = fit_segment(0, np.arange(nf), times, h, comps.prior_flux, data, weight, footprint)
    calm = fit_segment(0, np.arange(nf), times, h, comps.prior_flux, data, weight, footprint, quiet)
    assert plain.diagnostics.variable[0] and not calm.diagnostics.variable[0]


def test_variability_z_is_calibrated_across_stars():
    from tess_sso.stellar import _ensemble_z

    rng = np.random.default_rng(7)
    z = 3.0 * rng.normal(size=40)  # a drift common to all stars makes every |z| three times too large
    z[0] = 40.0  # one star that really changed
    tested = np.ones(40, bool)
    calibrated = _ensemble_z(z, tested)
    assert np.median(np.abs(calibrated[1:])) == pytest.approx(0.67, abs=0.2)
    assert calibrated[0] > 3 and np.sum(np.abs(calibrated[1:]) > 3) <= 1
    few = np.zeros(40, bool)
    few[:5] = True
    np.testing.assert_array_equal(_ensemble_z(z, few), z)  # too few stars: the per-epoch scale stays


def test_model_error_fraction_is_small_for_a_good_model():
    from tess_sso.stellar import model_error_fraction

    data, weight, footprint = scene(np.full(NF, 15.5), np.linspace(2, 38, NF), np.full(NF, 500.0))
    seg, _ = fit(data, weight, footprint)
    # the synthetic stars are exact: the model error is the noise of hourly means, well under 1 %
    assert 0 < seg.model_error_fraction < 0.005
    assert seg.model_error_fraction == pytest.approx(model_error_fraction(seg.diagnostics, seg.fluxes))


def test_normal_equations_match_the_dense_sums():
    from tess_sso.stellar import _apply, _block, _moments, _rhs

    rng = np.random.default_rng(11)
    dense = rng.normal(size=(3, 6, 50)) * (rng.random((3, 6, 50)) < 0.2)  # sparse templates
    h = Templates.from_dense(dense)
    weight = rng.random((30, 50))
    data = rng.normal(size=(30, 50))
    shifts = rng.normal(0, 0.05, (30, 2))
    e = np.column_stack([np.ones(30), shifts])
    design = np.einsum("ti,iks->tks", e, dense)  # (frames, components, pixels)
    a_dense = np.einsum("tks,ts,tls->kl", design, weight, design)
    b_dense = np.einsum("tks,ts,ts->k", design, weight, data)
    m, r = _moments(weight, data, shifts)
    np.testing.assert_allclose(_block(h, m, np.arange(6)), a_dense, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(_block(h, m, np.array([1, 4])), a_dense[np.ix_([1, 4], [1, 4])], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(_rhs(h, r), b_dense, rtol=1e-10, atol=1e-10)
    x = rng.normal(size=6)
    np.testing.assert_allclose(_apply(h, m, x), a_dense @ x, rtol=1e-10, atol=1e-10)


def test_edge_star_is_held_at_its_prior():
    comps = components()
    comps.rows.append(np.array([-4.0])); comps.cols.append(np.array([20.0]))  # core outside the image
    comps.ratios.append(np.array([1.0])); comps.source_ids.append(np.array([9]))
    comps.prior_flux = np.append(comps.prior_flux, 5000.0)
    data, weight, footprint = scene(np.full(NF, 15.5), np.linspace(2, 38, NF), np.full(NF, 500.0))
    seg, _ = fit(data, weight, footprint, comps)
    assert not seg.diagnostics.fitted[5]
    assert seg.fluxes[5] == pytest.approx(5000.0, abs=0.1)
    assert not seg.diagnostics.poorly_constrained[5]


def test_select_components():
    region = np.zeros((40, 40), bool)
    region[10:20, 10:20] = True
    rows = np.array([15.0, 15.3, 35.0, 12.0, 22.0])
    cols = np.array([15.0, 15.1, 35.0, 30.0, 12.0])
    tmag = np.array([10.0, 12.0, 10.0, 6.0, 11.0])
    flux = 10 ** (-0.4 * (tmag - 20.44))
    comps, near_saturated = select_components(np.arange(5), rows, cols, tmag, flux, region, (1, 1))
    assert len(comps) == 2  # the close pair merged; the far star and the saturated one left out
    merged = [i for i in range(2) if len(comps.rows[i]) == 2][0]
    assert comps.ratios[merged].sum() == pytest.approx(1.0)
    assert comps.prior_flux[merged] == pytest.approx(flux[0] + flux[1])
    assert near_saturated[11, 29] and not near_saturated[11, 5]  # around the T = 6 star


def test_fit_region():
    strip = np.ones((20, 20), bool)
    strip[:, 15:] = False
    region = fit_region(np.array([[10, 10]]), strip, (1, 1))
    assert region[9, 9] and region[9, 12] and not region[9, 13]
    assert region.sum() == 49


def test_model_evaluate_variance_and_roundtrip(tmp_path):
    data, weight, footprint = scene(np.full(NF, 15.5), np.linspace(2, 38, NF), np.full(NF, 500.0))
    seg, h = fit(data, weight, footprint)
    seg.frames = np.arange(100, 100 + NF)
    model = StellarModel(components(), PIXELS, h, [seg])
    values = model.evaluate([100, 150, 5000])
    assert np.isfinite(values[:2]).all() and np.isnan(values[2]).all()
    assert (model.variance([100])[0] >= 0).all()
    path = tmp_path / "stellar.fits"
    write_stellar(path, model, {"NPIX": len(PIXELS)})
    again = read_stellar(path, PRF)
    np.testing.assert_allclose(again.evaluate([100, 150]), values[:2])
    assert again.segments[0].diagnostics.fitted.all()
    np.testing.assert_array_equal(again.segments[0].diagnostics.variability_step, model.segments[0].diagnostics.variability_step)
    np.testing.assert_array_equal(again.segments[0].diagnostics.model_error, model.segments[0].diagnostics.model_error)
    assert again.segments[0].model_error_fraction == pytest.approx(model.segments[0].model_error_fraction)
    assert CHI2_LIMIT == 10.0


def test_templates_are_the_prf_and_its_derivatives():
    comps = components()
    h = templates(PRF, comps, PIXELS)
    assert isinstance(h, Templates) and h.shape == (5, len(PIXELS))
    dense = h.toarray()
    full = PRF.evaluate(STAR_R, STAR_C, (1, 1), (30, 40)).reshape(5, -1)  # unwindowed: the 15-px window loses nothing
    np.testing.assert_allclose(dense[0], full, atol=1e-10)
    step = 1e-4
    d_row = (PRF.evaluate(STAR_R + step, STAR_C, (1, 1), (30, 40)) - PRF.evaluate(STAR_R - step, STAR_C, (1, 1), (30, 40)))
    np.testing.assert_allclose(dense[1], d_row.reshape(5, -1) / (2 * step), atol=2e-3 * np.abs(dense[1]).max())
    assert h[0].nnz <= 5 * 15 * 15  # stored sparse


def test_templates_block_touching_and_project():
    rng = np.random.default_rng(3)
    dense = rng.random((3, 6, 40)) * (rng.random((3, 6, 40)) < 0.3)
    dense[0, 2, 5] = 0.5 * CORE_FRACTION
    h = Templates.from_dense(dense)
    stars, pixels, shift = np.array([4, 1]), np.array([7, 0, 33]), np.array([0.03, -0.02])
    expected = dense[0][np.ix_(stars, pixels)] + shift[0] * dense[1][np.ix_(stars, pixels)] + shift[1] * dense[2][np.ix_(stars, pixels)]
    np.testing.assert_allclose(h.block(stars, pixels, shift), expected)
    np.testing.assert_array_equal(h.touching(pixels), (dense[0][:, pixels] != 0).any(axis=1))
    np.testing.assert_array_equal(h.touching(pixels, CORE_FRACTION), (dense[0][:, pixels] >= CORE_FRACTION).any(axis=1))
    flux = rng.random(6)
    np.testing.assert_allclose(h.project(flux), np.einsum("k,ikp->ip", flux, dense))


def test_block_covariance():
    a = np.array([[2.0, 0.5], [0.5, 1.0]])
    b = np.array([[3.0, 0.1, 0.2], [0.1, 4.0, 0.3], [0.2, 0.3, 5.0]])
    cov = BlockCovariance(5, [(np.array([0, 2]), a), (np.array([1, 3, 4]), b)])
    np.testing.assert_allclose(cov.diag(), [2.0, 3.0, 1.0, 4.0, 5.0])
    np.testing.assert_allclose(cov.sub(np.array([4, 0, 2, 3])), [[5.0, 0, 0, 0.3], [0, 2.0, 0.5, 0], [0, 0.5, 1.0, 0], [0.3, 0, 0, 4.0]])
    np.testing.assert_allclose(cov.apply(np.ones(5)), [2.5, 3.3, 1.5, 4.4, 5.5])


def test_tiles_cover_every_component_once():
    from tess_sso.stellar import make_tiles

    rng = np.random.default_rng(5)
    centres = np.column_stack([rng.uniform(0, 30, 400), rng.uniform(0, 200, 400)])
    assert [len(c) for c, _ in make_tiles(centres, tile_components=500)] == [400]  # few enough: one tile, the whole fit
    tiles = make_tiles(centres, tile_components=150, margin=7)
    cores = np.concatenate([core for core, _ in tiles])
    assert len(tiles) > 1 and np.array_equal(np.sort(cores), np.arange(400))
    assert max(len(fit) for _, fit in tiles) <= 150
    for core, fit in tiles:
        assert np.isin(core, fit).all()
        near = np.abs(centres[:, None, :] - centres[None, core, :]).max(axis=2).min(axis=1) <= 7  # within 7 px of a core star
        assert np.isin(np.flatnonzero(near), fit).all()


def test_tiled_fit_matches_the_whole_fit():
    shifts = np.column_stack([0.05 * np.sin(np.arange(NF) / 40), 0.03 * np.cos(np.arange(NF) / 25)])
    data, weight, footprint = scene(np.full(NF, 15.5), np.linspace(2, 38, NF), np.full(NF, 500.0), shifts=shifts)
    comps = components()
    h = templates(PRF, comps, PIXELS)
    whole = fit_segment(0, np.arange(NF), TIMES, h, comps.prior_flux, data, weight, footprint)
    centres = np.array([comps.centre(i) for i in range(len(comps))])
    tiled = fit_segment(0, np.arange(NF), TIMES, h, comps.prior_flux, data, weight, footprint,
                        centres=centres, tile_components=2)
    assert len(whole.covariance.blocks) == 1 and len(tiled.covariance.blocks) > 1
    np.testing.assert_allclose(tiled.fluxes, whole.fluxes, rtol=1e-7)
    np.testing.assert_allclose(tiled.shifts, whole.shifts, atol=1e-9)
    # the stars are well apart: what a tile leaves out barely changes their errors
    np.testing.assert_allclose(tiled.covariance.diag(), whole.covariance.diag(), rtol=1e-3)


def test_tiled_covariance_roundtrips_and_old_files_still_read(tmp_path):
    from astropy.io import fits

    data, weight, footprint = scene(np.full(NF, 15.5), np.linspace(2, 38, NF), np.full(NF, 500.0))
    comps = components()
    h = templates(PRF, comps, PIXELS)
    centres = np.array([comps.centre(i) for i in range(len(comps))])
    seg = fit_segment(0, np.arange(NF), TIMES, h, comps.prior_flux, data, weight, footprint, centres=centres, tile_components=2)
    path = tmp_path / "stellar.fits"
    write_stellar(path, StellarModel(comps, PIXELS, h, [seg]), {})
    again = read_stellar(path, PRF).segments[0].covariance
    np.testing.assert_allclose(again.sub(np.arange(5)), seg.covariance.sub(np.arange(5)))
    # a file written before the tiles: one dense ST0_COV image
    with fits.open(path) as hdul:
        hdus = [hdu for hdu in hdul if hdu.name != "ST0_COVIDX"]
        hdus = [fits.ImageHDU(seg.covariance.sub(np.arange(5)), name="ST0_COV") if hdu.name == "ST0_COV" else hdu.copy()
                for hdu in hdus]
        fits.HDUList(hdus).writeto(tmp_path / "old.fits")
    old = read_stellar(tmp_path / "old.fits", PRF).segments[0].covariance
    np.testing.assert_allclose(old.sub(np.arange(5)), seg.covariance.sub(np.arange(5)))


def test_select_components_merges_pairs_as_the_brute_force_search():
    rng = np.random.default_rng(2)
    n = 3000  # crowded: ~2 stars per pixel, many within PAIR_PX of each other
    rows, cols = rng.uniform(1, 40, n), rng.uniform(1, 40, n)
    tmag = rng.uniform(8, 18, n)
    flux = 10 ** (-0.4 * (tmag - 20.44))
    region = np.zeros((40, 40), bool)
    region[10:30, 10:30] = True
    comps, _ = select_components(np.arange(n), rows, cols, tmag, flux, region, (1, 1))
    # the reference: each star, brightest first, joins the first group whose leader is within PAIR_PX
    from scipy import ndimage

    from tess_sso.stellar import PAIR_PX, SQUARE, WINDOW

    reach = ndimage.binary_dilation(region, SQUARE, iterations=WINDOW // 2)
    groups = []
    for i in np.argsort(-flux):
        r, c = int(np.floor(rows[i] + 0.5)) - 1, int(np.floor(cols[i] + 0.5)) - 1
        if not (0 <= r < 40 and 0 <= c < 40 and reach[r, c]):
            continue
        for group in groups:
            if np.hypot(rows[i] - rows[group[0]], cols[i] - cols[group[0]]) < PAIR_PX:
                group.append(i)
                break
        else:
            groups.append([i])
    assert sum(len(g) > 1 for g in groups) > 50  # the test does exercise merging
    assert [list(s) for s in comps.source_ids] == groups


def test_dense_field_epsilon_leaves_out_stars_that_disagree_with_gaia():
    from tess_sso.stellar import DENSE_GAIA_FACTOR, model_error_fraction

    data, weight, footprint = scene(np.full(NF, 15.5), np.linspace(2, 38, NF), np.full(NF, 500.0))
    comps = components()
    comps.prior_flux[0] = 10 * STAR_F[0]  # Gaia far off for star 0: the fit does not describe what Gaia lists
    h = templates(PRF, comps, PIXELS)
    plain = fit_segment(0, np.arange(NF), TIMES, h, comps.prior_flux, data, weight, footprint)
    dense = fit_segment(0, np.arange(NF), TIMES, h, comps.prior_flux, data, weight, footprint, dense=True)
    assert not plain.dense and dense.dense
    np.testing.assert_allclose(dense.fluxes, plain.fluxes)
    assert dense.fluxes[0] / comps.prior_flux[0] < 1 / DENSE_GAIA_FACTOR
    assert plain.model_error_fraction == pytest.approx(model_error_fraction(plain.diagnostics, plain.fluxes))
    without_0 = model_error_fraction(dense.diagnostics, dense.fluxes, comps.prior_flux, DENSE_GAIA_FACTOR)
    assert dense.model_error_fraction == pytest.approx(without_0)
    assert without_0 != pytest.approx(plain.model_error_fraction)


def test_dense_flag_roundtrips(tmp_path):
    data, weight, footprint = scene(np.full(NF, 15.5), np.linspace(2, 38, NF), np.full(NF, 500.0))
    comps = components()
    h = templates(PRF, comps, PIXELS)
    seg = fit_segment(0, np.arange(NF), TIMES, h, comps.prior_flux, data, weight, footprint, dense=True)
    write_stellar(tmp_path / "stellar.fits", StellarModel(comps, PIXELS, h, [seg]), {})
    assert read_stellar(tmp_path / "stellar.fits", PRF).segments[0].dense

