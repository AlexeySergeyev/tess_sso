import numpy as np
import pytest
from astropy.io import fits

from tess_sso import validation as V
from tess_sso.injection import InjectedSignal, dips, sinusoid

CAD = 200 / 86400
PERIOD = 8.7 / 24


def rotating(n=700, seed=0, noise=7.0):
    rng = np.random.default_rng(seed)
    t = 3235.0 + np.arange(n) * CAD
    signal = 3000 * (1 + 0.12 * np.sin(4 * np.pi * t / PERIOD) + 0.04 * np.sin(2 * np.pi * t / PERIOD + 1.0))
    return t, signal, signal + rng.normal(0, noise, n), np.full(n, noise)


def test_second_difference_pulls_ignore_the_rotation():
    t, _, flux, err = rotating()
    assert V.robust_rms(V.second_difference_pulls(t, flux, err, CAD)) == pytest.approx(1.0, abs=0.08)
    assert V.robust_rms(V.second_difference_pulls(t, flux, 2 * err, CAD)) == pytest.approx(0.5, abs=0.05)
    gappy = np.delete(np.arange(len(t)), np.arange(0, len(t), 2))  # every other exposure: no consecutive triples
    assert len(V.second_difference_pulls(t[gappy], flux[gappy], err[gappy], CAD)) == 0


def test_robust_rms_leaves_out_outliers():
    pulls = np.random.default_rng(1).normal(size=2000)
    pulls[:5] = 100.0
    assert V.robust_rms(pulls) == pytest.approx(1.0, abs=0.05)


def test_fourier_model_and_period():
    t, signal, flux, err = rotating()
    period = V.model_period(t, flux, err)
    assert period / PERIOD == pytest.approx(round(period / PERIOD), abs=0.02)  # the period or a multiple
    model = V.fourier_model(t, flux, err, period)
    assert np.std(model - signal) < 2.0


def test_fourier_model_chooses_the_harmonics():
    rng = np.random.default_rng(3)
    t = 3235.0 + np.arange(650) * CAD
    phase = 2 * np.pi * t / PERIOD
    sharp = 3000 * (1 - 0.3 * np.abs(np.cos(phase)) ** 0.3)  # an elongated body: sharp, deep minima
    flux = sharp + rng.normal(0, 7.0, t.size)
    err = np.full(t.size, 7.0)
    chosen = V.fourier_model(t, flux, err, PERIOD)
    few = V.fourier_model(t, flux, err, PERIOD, n_harmonics=2)
    assert np.std(chosen - sharp) < 0.5 * np.std(few - sharp)
    assert V.fourier_harmonics(t, flux, err, PERIOD) > 2
    smooth = 3000 * (1 + 0.1 * np.sin(phase)) + rng.normal(0, 7.0, t.size)
    assert V.fourier_harmonics(t, smooth, err, PERIOD) <= 2  # noise is not fitted with extra harmonics


def test_local_model_follows_a_changing_amplitude():
    rng = np.random.default_rng(12)
    t = 3235.0 + np.arange(4000) * CAD  # 9 days
    amplitude = 0.10 + 0.08 * (t - t[0]) / np.ptp(t)  # the light curve changes over the segment
    signal = 3000 * (1 + amplitude * np.sin(4 * np.pi * t / PERIOD))
    err = np.full(t.size, 7.0)
    flux = signal + rng.normal(0, 7.0, t.size)
    whole = V.fourier_model(t, flux, err, PERIOD)
    local = V.local_fourier_model(t, flux, err, PERIOD)
    assert np.std(local - signal) < 0.2 * np.std(whole - signal)
    assert np.std(local - signal) < 3.0
    fit = np.ones(t.size, bool)
    fit[2000:2100] = False  # a window left out is predicted from the other rotations in its chunk
    gap = V.local_fourier_model(t, flux, err, PERIOD, fit=fit)
    assert np.std((gap - signal)[2000:2100]) < 5.0


def test_significant_period_and_the_fallback_to_a_trend():
    t, signal, flux, err = rotating()
    period = V.significant_period(t, flux, err)
    assert period is not None and period / PERIOD == pytest.approx(round(period / PERIOD), abs=0.02)
    noise = 3000 + np.random.default_rng(5).normal(0, 7.0, t.size)
    assert V.significant_period(t, noise, err) is None
    trend = V.light_curve_model(t, noise, err)
    assert np.std(trend - 3000) < 1.0  # a smooth trend, not a fit to the noise


def test_crossing_check_needs_fitted_points_on_both_sides_and_scales_with_the_model():
    t, signal, flux, err = rotating()
    fit = np.ones(len(t), bool)
    edge = (t[0], t[20])  # at the start: nothing fitted before it
    fit[:21] = False
    model = V.fourier_model(t, flux, err, PERIOD, fit=fit)
    check = V.crossing_check(t, flux + 100, err, np.zeros(len(t), int), edge, model, fit=fit)
    assert not check.testable and check.passed  # not testable: reported, not failed
    inner = (t[300], t[340])
    fit = ~((t >= inner[0]) & (t <= inner[1]))
    rough = V.fourier_model(t, flux, err, PERIOD, n_harmonics=1, fit=fit)  # a model that misses the shape
    check = V.crossing_check(t, flux, err, np.zeros(len(t), int), inner, rough, fit=fit)
    assert check.testable and check.model_scale > 1.5 and check.passed
    segment = np.where(t < t[320], 0, 1)  # a segment boundary inside the window's margin
    fit = ~((t >= inner[0]) & (t <= inner[1]))
    across = V.crossing_check(t, flux, err, np.zeros(len(t), int), (t[321], t[340]), rough, fit=fit, segment=segment)
    assert not across.testable  # the points before it belong to another segment


def test_crossing_check():
    t, signal, flux, err = rotating()
    window = (t[300], t[340])
    inside = (t >= window[0]) & (t <= window[1])
    bumped = flux + np.where(inside, 400.0, 0.0)
    flags = np.zeros(len(t), int)
    model = V.fourier_model(t, bumped, err, PERIOD, fit=~inside)
    missed = V.crossing_check(t, bumped, err, flags, window, model)
    assert not missed.passed and missed.n_unflagged_outliers == inside.sum() and missed.max_deviation > 350
    flags[inside] = 32
    caught = V.crossing_check(t, bumped, err, flags, window, model)
    assert caught.passed and caught.n_flagged == inside.sum()


def synthetic_products(n=40, size=11, leak=0.05, level=3.0, stars_coef=1.0, fraction=0.8, seed=2):
    """A tess_sso pixel product, a reference target-pixel file, and its light curve, with a known background difference."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    asteroid = 3000 * np.exp(-((yy - 5) ** 2 + (xx - 5) ** 2) / 2.0) / (2 * np.pi)
    star = 800 * np.exp(-((yy - 1) ** 2 + (xx - 8) ** 2) / 2.0) / (2 * np.pi)
    ours_bkg = 400 + 2.0 * xx
    residual = rng.normal(0, 0.5, (n, size, size))
    flux = asteroid + star + residual  # data minus our background
    ref_bkg = ours_bkg + leak * asteroid + stars_coef * star + level
    data = flux + ours_bkg
    aperture = (asteroid >= 0.01 * asteroid.max()).astype(np.int32)
    cad = np.arange(n) + 100

    def columns(named):
        return [fits.Column(name=k, format=f"{size * size}{'I' if k == 'PIXEL_MASK' else 'E'}", dim=f"({size},{size})",
                            array=v) for k, v in named.items()]

    def stack(a):
        return np.repeat(a[None].astype(float), n, axis=0)

    ours = fits.HDUList([
        fits.PrimaryHDU(),
        fits.BinTableHDU.from_columns(
            [fits.Column(name="TIME", format="D", array=3235 + cad * CAD)] + columns({
                "FLUX": flux, "FLUX_BKG": stack(ours_bkg), "STAR_MODEL": stack(star), "ASTEROID_MODEL": stack(asteroid),
                "RESIDUAL": residual, "PIXEL_MASK": np.full((n, size, size), 4, np.int16)}), name="PIXELS"),
        fits.BinTableHDU.from_columns([fits.Column(name="CADENCENO", format="J", array=cad),
                                       fits.Column(name="CORNER1", format="J", array=np.full(n, 20)),
                                       fits.Column(name="CORNER2", format="J", array=np.full(n, 30))], name="EXTRAS"),
    ])
    ref = fits.HDUList([
        fits.PrimaryHDU(),
        fits.BinTableHDU.from_columns(columns({"FLUX": data - ref_bkg, "FLUX_BKG": np.repeat(ref_bkg[None], n, axis=0)}),
                                      name="PIXELS"),
        fits.BinTableHDU.from_columns([fits.Column(name="CADENCENO", format="J", array=cad),
                                       fits.Column(name="CORNER1", format="J", array=np.full(n, 20)),
                                       fits.Column(name="CORNER2", format="J", array=np.full(n, 30)),
                                       fits.Column(name="APERTURE", format=f"{size * size}J", dim=f"({size},{size})",
                                                   array=np.repeat(aperture[None], n, axis=0))], name="EXTRAS"),
    ])
    ref_flux = np.array([np.sum((data[k] - ref_bkg)[aperture > 0]) for k in range(n)]) / fraction
    ref_lc = fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns([
        fits.Column(name="CADENCENO", format="J", array=cad), fits.Column(name="FLUX", format="E", array=ref_flux),
        fits.Column(name="FLUX_FRACTION", format="E", array=np.full(n, fraction))])])
    return ours, ref, ref_lc


def test_background_decomposition_recovers_the_terms(tmp_path):
    ours, ref, _ = synthetic_products()
    table = V.background_decomposition(ours, ref)
    assert len(table) == 40
    assert np.median(table["ASTEROID"]) == pytest.approx(0.05, abs=0.002)
    assert np.median(table["STARS"]) == pytest.approx(1.0, abs=0.002)
    assert np.median(table["LEVEL"]) == pytest.approx(3.0, abs=0.05)


def test_reference_flux_budget_predicts_their_flux(tmp_path):
    ours, ref, ref_lc = synthetic_products()
    path = tmp_path / "ref-lc.fits"
    ref_lc.writeto(path)
    budget = V.reference_flux_budget(ours, ref, path)
    assert np.median(budget["PREDICTED"] / budget["ACTUAL"]) == pytest.approx(1.0, abs=0.002)
    assert np.median(budget["LEAK"]) == pytest.approx(0.05, abs=0.002)
    assert np.median(budget["LEVEL"]) > 0


def test_compare_with_reference(tmp_path):
    t, _, flux, err = rotating()
    cad = np.arange(len(t))

    def write(name, f, hdu_name="LIGHTCURVE_PRF"):
        cols = [fits.Column(name="TIME", format="D", array=t), fits.Column(name="CADENCENO", format="J", array=cad),
                fits.Column(name="FLUX", format="E", array=f), fits.Column(name="FLUX_ERR", format="E", array=err)]
        hdul = fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns(cols, name=hdu_name)])
        hdul.writeto(tmp_path / name)
        return tmp_path / name

    ours = write("ours.fits", flux)
    same = V.compare_with_reference(ours, write("ref.fits", 0.9 * flux, "LIGHTCURVE_AP"))
    assert same.agree and same.flux_ratio == pytest.approx(1 / 0.9, rel=1e-3) and same.n_common == len(t)
    trend = V.compare_with_reference(ours, write("trend.fits", flux * (1 + 0.05 * (t - t[0]) / np.ptp(t)), "LIGHTCURVE_AP"))
    assert not trend.agree


def test_crossing_windows():
    t = 3235.0 + np.arange(1000) * CAD
    star = np.full(1000, 100.0)
    star[300:320] = 5000.0
    star[322:330] = 3000.0  # the same crossing, merged
    star[800:805] = 800.0
    windows = V.crossing_windows(t, star)
    assert len(windows) == 2
    assert windows[0][0] == pytest.approx(t[300] - 0.05) and windows[0][1] == pytest.approx(t[329] + 0.05)
    inside = V.in_windows(t, windows)
    assert inside[300] and inside[805] and not inside[500]


def lightcurve_with_groups(member_times, weights, group_ids, flux=None, err=None, flags=None):
    rows = np.unique(group_ids)
    n = len(rows)
    lc = fits.BinTableHDU.from_columns([
        fits.Column(name="TIME", format="D", array=np.array([np.average(member_times[group_ids == g]) for g in rows])),
        fits.Column(name="GROUP_ID", format="J", array=rows),
        fits.Column(name="FLUX", format="D", array=np.ones(n) if flux is None else flux),
        fits.Column(name="FLUX_ERR", format="D", array=np.full(n, 0.01) if err is None else err),
        fits.Column(name="SSO_QUALITY", format="J", array=np.zeros(n, int) if flags is None else flags),
    ], name="LIGHTCURVE_PRF")
    groups = fits.BinTableHDU.from_columns([
        fits.Column(name="GROUP_ID", format="J", array=group_ids),
        fits.Column(name="ORIGINAL_TIME", format="D", array=member_times),
        fits.Column(name="WEIGHT", format="D", array=weights),
    ], name="GROUPS")
    return fits.HDUList([fits.PrimaryHDU(), lc, groups])


def test_expected_factor_averages_the_members_with_their_weights():
    times = np.array([0.0, 0.25, 0.5, 0.75])
    hdul = lightcurve_with_groups(times, np.array([0.5, 0.5, 0.25, 0.75]), np.array([0, 0, 1, 1]))
    factor = sinusoid(0.1, 1.0)
    assert V.expected_factor(hdul, factor) == pytest.approx([1.05, 0.25 * 1.0 + 0.75 * 0.9])


def test_recovered_factor_matches_rows_by_group():
    times = np.arange(4) * 0.1
    plain = lightcurve_with_groups(times, np.ones(4), np.arange(4), flux=np.full(4, 200.0), err=np.full(4, 2.0))
    injected = lightcurve_with_groups(times[1:] + 1e-4, np.ones(3), np.arange(1, 4), flux=np.array([220.0, 180.0, 200.0]),
                                      flags=np.array([0, 4, 0]))
    rec = V.recovered_factor(plain, injected)
    assert rec["ROW"].tolist() == [1, 2, 3]
    assert rec["RATIO"] == pytest.approx([1.1, 0.9, 1.0])
    assert rec["RATIO_ERR"] == pytest.approx([0.01] * 3)
    assert rec["SSO_INJECTED"].tolist() == [0, 4, 0]


def test_fit_sinusoid_recovers_the_amplitude_and_period():
    rng = np.random.default_rng(3)
    t = 3235.0 + np.arange(4000) * 200 / 86400
    err = np.full(t.size, 0.01)
    ratio = sinusoid(0.05, 0.25, 0.7)(t) + rng.normal(0, 0.01, t.size)
    fit = V.fit_sinusoid(t, ratio, err, 0.25 * 1.004)
    assert fit.amplitude == pytest.approx(0.05, abs=3 * fit.amplitude_err)
    assert fit.amplitude_err == pytest.approx(0.01 * np.sqrt(2 / t.size), rel=0.1)
    assert fit.period == pytest.approx(0.25, abs=max(3 * fit.period_err, 1e-5))
    assert fit.period_err < 1e-4


def test_dip_depth():
    rng = np.random.default_rng(4)
    t = 3235.0 + np.arange(2000) * 200 / 86400
    err = np.full(t.size, 0.01)
    window = (3236.0, 3236.0 + 1 / 24)
    ratio = dips([window], 0.02)(t) + rng.normal(0, 0.01, t.size)
    depth, depth_err = V.dip_depth(t, ratio, err, window)
    assert depth == pytest.approx(0.02, abs=3 * depth_err)
    assert depth_err == pytest.approx(0.01 * np.sqrt(1 / 18 + 1 / 108), rel=0.2)
    assert np.isnan(V.dip_depth(t, ratio, err, (3000.0, 3000.1))[0])


def synthetic_runs(seed=5, absorb=0.0, raise_flags=0, crossing_miss=False):
    """Plain and injected light-curve files of single exposures, the injected flux exact up to ``absorb`` (a
    fraction of each dip lost) and ``raise_flags`` new flags; a bright star at BTJD 3236.0."""
    rng = np.random.default_rng(seed)
    t = 3235.0 + np.arange(4000) * 200 / 86400
    star = np.where(np.abs(t - 3236.0) < 0.02, 500.0, 1.0)
    signal = InjectedSignal(crossing=((3236.0 - 1 / 48, 3236.0 + 1 / 48),), isolated=((3235.5, 3235.5 + 1 / 24),))
    flux = 1000.0 + rng.normal(0, 10, t.size)
    injected = flux * signal(t)
    dipped = V.in_windows(t, signal.windows)
    injected[dipped] += absorb * signal.depth * flux[dipped]
    flags = np.where(np.abs(t - 3236.0) < 0.01, 32, 0)
    flags_inj = flags.copy()
    flags_inj[np.flatnonzero(flags == 0)[:raise_flags]] = 8
    if crossing_miss:
        miss = np.flatnonzero((np.abs(t - 3236.0) < 0.05) & (flags == 0))[0]
        injected[miss] *= 1.2

    def lightcurve(f, q):
        lc = fits.BinTableHDU.from_columns([
            fits.Column(name="TIME", format="D", array=t + 0.001),
            fits.Column(name="GROUP_ID", format="J", array=np.arange(t.size)),
            fits.Column(name="SEGMENT", format="I", array=np.zeros(t.size, int)),
            fits.Column(name="FLUX", format="D", array=f),
            fits.Column(name="FLUX_ERR", format="D", array=np.full(t.size, 10.0)),
            fits.Column(name="SSO_QUALITY", format="J", array=q),
            fits.Column(name="STAR_FLUX", format="D", array=star),
        ], name="LIGHTCURVE_PRF")
        extras = fits.BinTableHDU.from_columns([fits.Column(name="ORIGINAL_TIME", format="D", array=t)], name="EXTRAS")
        groups = fits.BinTableHDU.from_columns([
            fits.Column(name="GROUP_ID", format="J", array=np.arange(t.size)),
            fits.Column(name="ORIGINAL_TIME", format="D", array=t),
            fits.Column(name="WEIGHT", format="D", array=np.ones(t.size)),
        ], name="GROUPS")
        return fits.HDUList([fits.PrimaryHDU(), lc, extras, groups])

    return lightcurve(flux, flags), lightcurve(injected, flags_inj), signal


def test_recovery_of_an_exact_injection_passes():
    plain, injected, signal = synthetic_runs()
    r = V.recovery(plain, injected, signal)
    assert r.rows == 4000 and r.gain == pytest.approx(1.0, abs=1e-9)
    assert r.pull_rms < 1e-6
    assert r.amplitude.amplitude == pytest.approx(0.05, abs=1e-6) and r.amplitude_expected == pytest.approx(0.05, abs=1e-6)
    assert r.dips["CROSSING"].tolist() == [True, False]
    assert r.dips["FLAGGED"].tolist() == [False, False]  # the crossing's flagged points leave unflagged ones
    assert r.dips["DEPTH"] == pytest.approx([0.02, 0.02], abs=1e-6)
    assert r.new_flags == 0 and r.crossing_failures == 0 and r.crossing_flagged > 0
    assert r.criterion3() and r.criterion5()


def test_recovery_catches_an_absorbed_dip_raised_flags_and_a_missed_crossing():
    assert not V.recovery(*synthetic_runs(absorb=0.5)).criterion3()  # half of each dip lost
    r = V.recovery(*synthetic_runs(raise_flags=41))  # just over 1 % of 4000
    assert r.new_flags == 41 and r.flags["NEW"][r.flags["BIT"] == 3][0] == 41 and not r.criterion3()
    assert V.recovery(*synthetic_runs(raise_flags=40)).criterion3()
    r = V.recovery(*synthetic_runs(crossing_miss=True))
    assert r.crossing_failures == 1 and not r.criterion5()
