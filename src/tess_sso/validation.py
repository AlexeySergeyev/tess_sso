"""Validation measures (phase 10): error calibration, light-curve models, crossings, and the reference pipeline.

Each function works on plain arrays or on the files tess_sso and the
reference pipeline write, so the same code runs in tests, in the
validation notebook, and on any other observation. The acceptance criteria
themselves are in the spec, *Initial validation: accepted scenarios*.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from astropy.io import fits
from astropy.table import Table

GAUSSIAN_MAD = 0.6745  # median |x| of a unit Gaussian


def second_difference_pulls(time: np.ndarray, flux: np.ndarray, err: np.ndarray, cadence: float) -> np.ndarray:
    """Pulls of f[i-1] - 2 f[i] + f[i+1] over consecutive points (gaps under 1.5 cadences).

    Insensitive to any signal smooth over two cadences, so their RMS
    measures the point-to-point errors without a light-curve model.
    """
    t, f, e = (np.asarray(v, float) for v in (time, flux, err))
    order = np.argsort(t)
    t, f, e = t[order], f[order], e[order]
    consecutive = (np.diff(t)[:-1] < 1.5 * cadence) & (np.diff(t)[1:] < 1.5 * cadence)
    d2 = (f[:-2] - 2 * f[1:-1] + f[2:])[consecutive]
    var = (e[:-2] ** 2 + 4 * e[1:-1] ** 2 + e[2:] ** 2)[consecutive]
    pulls = d2 / np.sqrt(var)
    return pulls[np.isfinite(pulls)]


def robust_rms(pulls: np.ndarray, clip: float = 5.0) -> float:
    """RMS of the pulls within ``clip`` robust sigma (short events and outliers left out)."""
    pulls = np.asarray(pulls, float)
    scale = max(1.4826 * float(np.median(np.abs(pulls))), 1e-12)
    keep = np.abs(pulls) <= clip * max(scale, 1.0)
    return float(np.sqrt(np.mean(pulls[keep] ** 2)))


def model_period(time: np.ndarray, flux: np.ndarray, err: np.ndarray, min_hours: float = 2.0, max_hours: float = 72.0) -> float:
    """A period (days) the light curve repeats with, for :func:`fourier_model`: twice the peak of a
    two-term Lomb-Scargle periodogram.

    On a stretch of a day or two this is the rotation period or twice it
    (asteroid 95: 17.3 h for 8.7 h; 159: 24.8 h; 4337: 32.7 h), which a
    Fourier model with several harmonics fits equally well. It is a
    modelling aid, not a period determination. The periodogram's peak is
    refined by the chi2 of :func:`fourier_model` within 3 %, since its
    resolution on a short stretch is too coarse for many harmonics.
    """
    from astropy.timeseries import LombScargle

    t, f, e = (np.asarray(v, float) for v in (time, flux, err))
    freq, power = LombScargle(t, f, e, nterms=2).autopower(
        minimum_frequency=2.0 / (max_hours / 24), maximum_frequency=2.0 / (min_hours / 24), samples_per_peak=20
    )
    peak = 2.0 / freq[np.argmax(power)]
    candidates = peak * np.linspace(0.97, 1.03, 121)
    chi2 = [np.sum(((f - fourier_model(t, f, e, p, n_harmonics=8)) / e) ** 2) for p in candidates]
    return float(candidates[int(np.argmin(chi2))])


FALSE_ALARM = 0.01  # a periodogram peak less significant than this gives no Fourier model


def significant_period(time: np.ndarray, flux: np.ndarray, err: np.ndarray, min_hours: float = 2.0,
                       max_hours: float = 72.0) -> float | None:
    """:func:`model_period`, or None when the periodogram's peak is not significant (false-alarm
    probability above FALSE_ALARM): a faint target's noise must not be modelled as rotation."""
    from astropy.timeseries import LombScargle

    t, f, e = (np.asarray(v, float) for v in (time, flux, err))
    periodogram = LombScargle(t, f, e)
    freq, power = periodogram.autopower(minimum_frequency=1.0 / (max_hours / 24), maximum_frequency=2.0 / (min_hours / 24))
    if periodogram.false_alarm_probability(power.max(), minimum_frequency=freq.min(), maximum_frequency=freq.max()) > FALSE_ALARM:
        return None
    return model_period(t, f, e, min_hours, max_hours)


def light_curve_model(time: np.ndarray, flux: np.ndarray, err: np.ndarray, fit: np.ndarray | None = None) -> np.ndarray:
    """The light curve without what ``fit`` leaves out: :func:`local_fourier_model` at a significant
    period, otherwise a quadratic trend per chunk of LOCAL_WINDOW_DAYS."""
    t, f, e = (np.asarray(v, float) for v in (time, flux, err))
    fit = np.ones(len(t), bool) if fit is None else np.asarray(fit, bool)
    period = significant_period(t[fit], f[fit], e[fit])
    if period is not None:
        return local_fourier_model(t, f, e, period, fit=fit)
    n_chunks = max(int(np.floor(np.ptp(t) / LOCAL_WINDOW_DAYS)), 1)
    edges = np.linspace(t.min(), t.max(), n_chunks + 1)
    edges[-1] = np.inf
    model = np.full(len(t), np.nan)
    for lo, hi in zip(edges[:-1], edges[1:]):
        on = (t >= lo) & (t < hi)
        if (on & fit).sum() >= 20:
            model[on] = fourier_model(t[on], f[on], e[on], 1.0, n_harmonics=0, poly=2, fit=fit[on])
    return model


MAX_HARMONICS = 30


def _fourier_design(t: np.ndarray, period: float, n_harmonics: int, poly: int, modulate: bool = False) -> np.ndarray:
    """Polynomial trend and harmonics; with ``modulate`` each harmonic's amplitude also changes linearly in time."""
    tc = (t - t.mean()) / max(np.ptp(t), 1e-9)
    columns = [tc**k for k in range(poly + 1)]
    for h in range(1, n_harmonics + 1):
        pair = [np.sin(2 * np.pi * h * t / period), np.cos(2 * np.pi * h * t / period)]
        columns += pair + ([tc * pair[0], tc * pair[1]] if modulate else [])
    return np.column_stack(columns)


def _fourier_fit(t, f, e, period, n_harmonics, poly, fit, modulate=False) -> tuple[np.ndarray, float]:
    """(model on all points, chi2 on the fitted points)."""
    x = _fourier_design(t, period, n_harmonics, poly, modulate)
    w = 1.0 / e
    beta, *_ = np.linalg.lstsq(x[fit] * w[fit, None], f[fit] * w[fit], rcond=None)
    model = x @ beta
    return model, float(np.sum(((f - model)[fit] * w[fit]) ** 2))


def fourier_harmonics(
    time: np.ndarray, flux: np.ndarray, err: np.ndarray, period: float, poly: int = 2, fit: np.ndarray | None = None,
    max_harmonics: int = MAX_HARMONICS, modulate: bool = False,
) -> int:
    """Number of harmonics (1 to ``max_harmonics``) with the lowest BIC, chi2 + k ln n.

    Sharp minima of an elongated body need many harmonics; noise does not
    earn them. Harmonics that the points cannot constrain (fewer than three
    points per parameter) are not tried.
    """
    t, f, e = (np.asarray(v, float) for v in (time, flux, err))
    fit = np.ones(len(t), bool) if fit is None else np.asarray(fit, bool)
    n = int(fit.sum())
    best, best_bic = 1, np.inf
    for h in range(1, max_harmonics + 1):
        k = poly + 1 + (4 if modulate else 2) * h
        if 3 * k > n:
            break
        _, chi2 = _fourier_fit(t, f, e, period, h, poly, fit, modulate)
        bic = chi2 + k * np.log(n)
        if bic < best_bic:
            best, best_bic = h, bic
    return best


def fourier_model(
    time: np.ndarray, flux: np.ndarray, err: np.ndarray, period: float, n_harmonics: int | None = None, poly: int = 2,
    fit: np.ndarray | None = None, modulate: bool = False,
) -> np.ndarray:
    """A light-curve model: Fourier series in ``period`` plus a polynomial trend, weighted, fitted on ``fit``.

    ``n_harmonics=None`` chooses them by BIC (:func:`fourier_harmonics`);
    ``modulate`` lets each harmonic's amplitude change linearly in time.
    """
    t, f, e = (np.asarray(v, float) for v in (time, flux, err))
    fit = np.ones(len(t), bool) if fit is None else np.asarray(fit, bool)
    if n_harmonics is None:
        n_harmonics = fourier_harmonics(t, f, e, period, poly, fit, modulate=modulate)
    return _fourier_fit(t, f, e, period, n_harmonics, poly, fit, modulate)[0]


LOCAL_WINDOW_DAYS = 2.0  # a light curve's shape changes over days (phase angle, amplitude)
LOCAL_WINDOW_PERIODS = 3.0  # ... but each chunk needs a few rotations to cover every phase


def local_fourier_model(
    time: np.ndarray, flux: np.ndarray, err: np.ndarray, period: float, fit: np.ndarray | None = None,
    window_days: float = LOCAL_WINDOW_DAYS,
) -> np.ndarray:
    """:func:`fourier_model` fitted in consecutive chunks of max(``window_days``, 3 periods), with a linear
    trend and harmonic amplitudes that change linearly within the chunk.

    One model per multi-day segment cannot follow a light curve whose
    amplitude and shape change with the phase angle; chunks of a few
    rotations can, and still predict a left-out window from the other
    rotations of its chunk. A chunk with fewer than 20 fitted points is left
    without a model (NaN).
    """
    t, f, e = (np.asarray(v, float) for v in (time, flux, err))
    fit = np.ones(len(t), bool) if fit is None else np.asarray(fit, bool)
    width = max(window_days, LOCAL_WINDOW_PERIODS * period)
    n_chunks = max(int(np.floor(np.ptp(t) / width)), 1)
    edges = np.linspace(t.min(), t.max(), n_chunks + 1)
    edges[-1] = np.inf
    model = np.full(len(t), np.nan)
    for lo, hi in zip(edges[:-1], edges[1:]):
        on = (t >= lo) & (t < hi)
        if (on & fit).sum() < 20:
            continue
        model[on] = fourier_model(t[on], f[on], e[on], period, poly=1, fit=fit[on], modulate=True)
    return model


@dataclass(frozen=True)
class CrossingCheck:
    n_points: int  # in the window
    n_flagged: int
    n_unflagged_outliers: int  # beyond ``sigma`` of the model without being flagged
    max_deviation: float  # e-/s
    passed: bool  # every point flagged or within ``sigma``; True for a window that cannot be tested
    testable: bool = True  # fitted points on both sides of the window
    model_scale: float = 1.0  # the model's own scatter around fitted points near the window, in total errors


def crossing_windows(time: np.ndarray, star_flux: np.ndarray, factor: float = 5.0, margin: float = 0.05) -> list[tuple[float, float]]:
    """Time windows (BTJD) in which a bright star lies under the asteroid.

    Points whose |STAR_FLUX| exceeds ``factor`` times its median, each
    widened by ``margin`` days on both sides, merged where they overlap.
    """
    t = np.asarray(time, float)
    sf = np.abs(np.asarray(star_flux, float))
    windows: list[list[float]] = []
    for k in np.flatnonzero(sf > factor * np.nanmedian(sf)):
        lo, hi = t[k] - margin, t[k] + margin
        if windows and lo <= windows[-1][1]:
            windows[-1][1] = max(windows[-1][1], hi)
        else:
            windows.append([lo, hi])
    return [(lo, hi) for lo, hi in windows]


def in_windows(time: np.ndarray, windows) -> np.ndarray:
    t = np.asarray(time, float)
    inside = np.zeros(len(t), bool)
    for lo, hi in windows:
        inside |= (t >= lo) & (t <= hi)
    return inside


def crossing_check(
    time: np.ndarray, flux: np.ndarray, total_err: np.ndarray, flags: np.ndarray, window: tuple[float, float],
    model: np.ndarray, sigma: float = 3.0, fit: np.ndarray | None = None, margin: float = 0.25,
    segment: np.ndarray | None = None,
) -> CrossingCheck:
    """Criterion 5: at a crossing, each point is flagged or within ``sigma`` of the light curve without the star.

    ``model`` is the light curve fitted on ``fit`` (without the window's
    points); ``total_err`` combines the random and coherent errors. With
    ``fit``, the window is testable only if fitted points lie within
    ``margin`` days on both sides (at a segment's edge the model is an
    extrapolation), and sigma includes the model's own accuracy: the RMS
    pull of the fitted points within a day of the window, if above one.
    With ``segment`` (per point), only fitted points of the window's own
    segment count, since each segment has its own model.
    """
    t = np.asarray(time, float)
    err = np.asarray(total_err, float)
    inside = (t >= window[0]) & (t <= window[1])
    dev = np.asarray(flux, float) - np.asarray(model, float)
    flagged = np.asarray(flags) != 0
    testable, scale = True, 1.0
    if fit is not None:
        fit = np.asarray(fit, bool) & np.isfinite(dev)
        if segment is not None and inside.any():
            segment = np.asarray(segment)
            fit = fit & (segment == segment[np.flatnonzero(inside)[0]])
        before = fit & (t < window[0]) & (t >= window[0] - margin)
        after = fit & (t > window[1]) & (t <= window[1] + margin)
        testable = bool(before.any() and after.any())
        near = fit & (np.abs(t - 0.5 * (window[0] + window[1])) <= 0.5 * (window[1] - window[0]) + 1.0)
        if near.sum() >= 10:
            scale = max(1.0, float(np.sqrt(np.mean((dev[near] / err[near]) ** 2))))
    outlier = np.abs(dev) > sigma * scale * err
    bad = inside & outlier & ~flagged
    return CrossingCheck(
        n_points=int(inside.sum()),
        n_flagged=int((inside & flagged).sum()),
        n_unflagged_outliers=int(bad.sum()) if testable else 0,
        max_deviation=float(np.nanmax(np.abs(dev[inside]))) if inside.any() else np.nan,
        passed=bool(not testable or (inside.any() and not bad.any())),
        testable=testable,
        model_scale=scale,
    )


def background_decomposition(pixel_product, reference_tpf) -> Table:
    """Reference background minus ours, per common exposure, as a asteroid + b stars + c + plane.

    ``pixel_product`` is a tess_sso single-exposure pixel product and
    ``reference_tpf`` the reference pipeline's target-pixel file (paths or
    HDU lists). Exposures are matched by CADENCENO, and only those whose
    cutouts share the same corner are used. Columns: CADENCENO, ASTEROID
    (fraction of our asteroid model in their background), STARS, LEVEL
    (e-/s per pixel), GRAD_COL, GRAD_ROW (e-/s per pixel per pixel), RMS.
    """
    ours = pixel_product if isinstance(pixel_product, fits.HDUList) else fits.open(pixel_product)
    ref = reference_tpf if isinstance(reference_tpf, fits.HDUList) else fits.open(reference_tpf)
    op, ox = ours["PIXELS"].data, ours["EXTRAS"].data
    rp, rx = ref["PIXELS"].data, ref["EXTRAS"].data
    index = {int(c): i for i, c in enumerate(rx["CADENCENO"])}
    ny, nx = op["FLUX"].shape[1:]
    yy, xx = np.mgrid[0:ny, 0:nx]
    rows = []
    for k in range(len(op)):
        j = index.get(int(ox["CADENCENO"][k]))
        if j is None or (rx["CORNER1"][j], rx["CORNER2"][j]) != (ox["CORNER1"][k], ox["CORNER2"][k]):
            continue
        if rp["FLUX_BKG"][j].shape != (ny, nx):
            continue
        diff = (rp["FLUX_BKG"][j] - op["FLUX_BKG"][k]).ravel()
        x = np.column_stack([op["ASTEROID_MODEL"][k].ravel(), op["STAR_MODEL"][k].ravel(), np.ones(diff.size),
                             xx.ravel() - nx // 2, yy.ravel() - ny // 2])
        ok = np.isfinite(diff) & np.isfinite(x).all(axis=1)
        if ok.sum() < 10:
            continue
        coef, *_ = np.linalg.lstsq(x[ok], diff[ok], rcond=None)
        rows.append((int(ox["CADENCENO"][k]), *coef, float(np.std(diff[ok] - x[ok] @ coef))))
    return Table(rows=rows, names=("CADENCENO", "ASTEROID", "STARS", "LEVEL", "GRAD_COL", "GRAD_ROW", "RMS"))


@dataclass(frozen=True)
class ReferenceComparison:
    n_common: int
    flux_ratio: float  # median of ours / theirs (total fluxes)
    bin_rms: float  # RMS of the difference of median-normalized 30-minute bins
    bin_limit: float  # max(1 %, 2 sigma of that difference)
    agree: bool  # bin_rms <= bin_limit


def compare_with_reference(our_lc, reference_lc, bin_minutes: float = 30.0, hdu: str = "LIGHTCURVE_PRF") -> ReferenceComparison:
    """Our light curve against the reference pipeline's on common exposures (single-exposure files).

    Points flagged in either file (our ``SSO_QUALITY`` or ``QUALITY``, their
    ``QUALITY``) are left out. Both are normalized by their medians per
    observing segment (our ``SEGMENT``; the whole file without it), since an
    additive difference between the two changes their ratio as the asteroid
    brightens over a sector, and averaged in ``bin_minutes`` bins; the
    difference's RMS is compared with max(1 %, 2 sigma), sigma from both
    files' errors.
    """
    ours = Table(fits.getdata(our_lc, hdu))
    ref = Table(fits.getdata(reference_lc, 1))
    index = {int(c): i for i, c in enumerate(ref["CADENCENO"])}
    pairs = [(i, index[int(c)]) for i, c in enumerate(ours["CADENCENO"]) if int(c) in index]
    i_o = np.array([p[0] for p in pairs], int)
    i_r = np.array([p[1] for p in pairs], int)
    t = np.asarray(ours["TIME"], float)[i_o]
    fo, eo = np.asarray(ours["FLUX"], float)[i_o], np.asarray(ours["FLUX_ERR"], float)[i_o]
    fr, er = np.asarray(ref["FLUX"], float)[i_r], np.asarray(ref["FLUX_ERR"], float)[i_r]

    def column(table, name, index):
        return np.asarray(table[name])[index] if name in table.colnames else np.zeros(len(index), int)

    clean = (column(ours, "SSO_QUALITY", i_o) == 0) & (column(ours, "QUALITY", i_o) == 0) & (column(ref, "QUALITY", i_r) == 0)
    segment = column(ours, "SEGMENT", i_o)
    ok = clean & np.isfinite(fo) & np.isfinite(fr) & np.isfinite(eo) & np.isfinite(er) & (fr > 0)
    t, fo, eo, fr, er, segment = t[ok], fo[ok], eo[ok], fr[ok], er[ok], segment[ok]
    if len(t) == 0:
        return ReferenceComparison(0, np.nan, np.nan, np.nan, False)
    ratio = float(np.median(fo / fr))
    diffs, sigmas = [], []
    for seg in np.unique(segment):
        on = segment == seg
        mo, mr = np.median(fo[on]), np.median(fr[on])
        bins = np.floor((t[on] - t[on].min()) / (bin_minutes / 1440)).astype(int)
        for b in np.unique(bins):
            sel = np.flatnonzero(on)[bins == b]
            n = len(sel)
            diffs.append(np.mean(fo[sel]) / mo - np.mean(fr[sel]) / mr)
            sigmas.append(np.hypot(np.sqrt(np.sum(eo[sel] ** 2)) / n / mo, np.sqrt(np.sum(er[sel] ** 2)) / n / mr))
    diffs = np.array(diffs)
    rms = float(np.sqrt(np.mean((diffs - np.median(diffs)) ** 2)))
    limit = max(0.01, 2 * float(np.median(sigmas)))
    return ReferenceComparison(len(t), ratio, rms, limit, rms <= limit)


def reference_flux_budget(pixel_product, reference_tpf, reference_lc) -> Table:
    """Per common exposure: the reference flux predicted from our images and the background decomposition.

    In the reference aperture, their sum is ours minus the background
    difference: A(1 - leak) + S(1 - stars) + R - level N - plane, with A, S,
    R our asteroid model, stellar model, and residual summed in their
    aperture; divided by their FLUX_FRACTION it predicts their FLUX.
    Columns: CADENCENO, ACTUAL, PREDICTED, and the terms as fractions of our
    asteroid light in the aperture: LEAK, STARS, LEVEL, PLANE, RESIDUAL.
    """
    ours = pixel_product if isinstance(pixel_product, fits.HDUList) else fits.open(pixel_product)
    ref = reference_tpf if isinstance(reference_tpf, fits.HDUList) else fits.open(reference_tpf)
    rlc = Table(fits.getdata(reference_lc, 1))
    decomposition = background_decomposition(ours, ref)
    op, ox = ours["PIXELS"].data, ours["EXTRAS"].data
    rx = ref["EXTRAS"].data
    k_of = {int(c): k for k, c in enumerate(ox["CADENCENO"])}
    j_of = {int(c): j for j, c in enumerate(rx["CADENCENO"])}
    lc_of = {int(c): i for i, c in enumerate(rlc["CADENCENO"])}
    ny, nx = op["FLUX"].shape[1:]
    yy, xx = np.mgrid[0:ny, 0:nx]
    rows = []
    for d in decomposition:
        c = int(d["CADENCENO"])
        if c not in lc_of:
            continue
        k, j, i = k_of[c], j_of[c], lc_of[c]
        ap = np.asarray(rx["APERTURE"][j]).reshape(ny, nx) > 0
        a_sum = float(np.nansum(op["ASTEROID_MODEL"][k][ap]))
        s_sum = float(np.nansum(op["STAR_MODEL"][k][ap]))
        r_sum = float(np.nansum(op["RESIDUAL"][k][ap]))
        plane = float(np.sum((d["GRAD_COL"] * (xx - nx // 2) + d["GRAD_ROW"] * (yy - ny // 2))[ap]))
        level = float(d["LEVEL"]) * int(ap.sum())
        predicted_ap = a_sum * (1 - d["ASTEROID"]) + s_sum * (1 - d["STARS"]) + r_sum - level - plane
        fraction = float(rlc["FLUX_FRACTION"][i])
        if not (a_sum > 0 and fraction > 0):
            continue
        rows.append((c, float(rlc["FLUX"][i]), predicted_ap / fraction, float(d["ASTEROID"]),
                     -s_sum * (1 - d["STARS"]) / a_sum, level / a_sum, plane / a_sum, -r_sum / a_sum))
    return Table(rows=rows, names=("CADENCENO", "ACTUAL", "PREDICTED", "LEAK", "STARS", "LEVEL", "PLANE", "RESIDUAL"))


# --- injection and recovery (phase 10c) -----------------------------------------------------------


def expected_factor(lightcurve, factor) -> np.ndarray:
    """Per light-curve row: the injected factor averaged over the group's members with their weights.

    ``lightcurve`` is a light-curve file (path or HDU list); the members'
    SPOC times (``ORIGINAL_TIME`` in ``GROUPS``) are those the injection used.
    """
    hdul = lightcurve if isinstance(lightcurve, fits.HDUList) else fits.open(lightcurve)
    rows = hdul["LIGHTCURVE_PRF"].data
    members = hdul["GROUPS"].data
    values = np.asarray(factor(np.asarray(members["ORIGINAL_TIME"], float)), float) * np.asarray(members["WEIGHT"], float)
    by_group = {}
    for g, v in zip(members["GROUP_ID"], values):
        by_group[int(g)] = by_group.get(int(g), 0.0) + v
    return np.array([by_group.get(int(g), np.nan) for g in rows["GROUP_ID"]])


def recovered_factor(plain_lightcurve, injected_lightcurve, hdu: str = "LIGHTCURVE_PRF") -> Table:
    """Injected over plain flux on common rows (matched by GROUP_ID): TIME, RATIO, RATIO_ERR, SSO_PLAIN,
    SSO_INJECTED, ROW (the plain file's row). RATIO_ERR is the plain random error relative to the flux: the two
    share their noise, so this bounds the ratio's error from above.

    Not matched by TIME: a group's time is weighted by its members' errors, which follow the asteroid's flux.
    """
    def table(lc):
        hdul = lc if isinstance(lc, fits.HDUList) else fits.open(lc)
        return hdul[hdu].data
    plain, injected = table(plain_lightcurve), table(injected_lightcurve)
    index = {int(g): i for i, g in enumerate(injected["GROUP_ID"])}
    pairs = [(i, index.get(int(g))) for i, g in enumerate(plain["GROUP_ID"])]
    pairs = [(i, j) for i, j in pairs if j is not None]
    i_p = np.array([p[0] for p in pairs], int)
    i_i = np.array([p[1] for p in pairs], int)
    fp = np.asarray(plain["FLUX"], float)[i_p]
    return Table({
        "TIME": np.asarray(plain["TIME"], float)[i_p],
        "RATIO": np.asarray(injected["FLUX"], float)[i_i] / fp,
        "RATIO_ERR": np.asarray(plain["FLUX_ERR"], float)[i_p] / np.abs(fp),
        "SSO_PLAIN": np.asarray(plain["SSO_QUALITY"])[i_p],
        "SSO_INJECTED": np.asarray(injected["SSO_QUALITY"])[i_i],
        "ROW": i_p,
    })


@dataclass(frozen=True)
class SinusoidFit:
    amplitude: float
    amplitude_err: float
    period: float  # days
    period_err: float
    phase: float


def fit_sinusoid(time: np.ndarray, ratio: np.ndarray, err: np.ndarray, period_guess: float) -> SinusoidFit:
    """Least squares of 1 + a sin(2 pi t / P + phi) to the recovered factor, P refined around the guess."""
    t, y, e = (np.asarray(v, float) for v in (time, ratio, err))
    ok = np.isfinite(t) & np.isfinite(y) & np.isfinite(e) & (e > 0)
    t, y, e = t[ok], y[ok], e[ok]
    t0 = t.mean()

    def linear(period):
        x = np.column_stack([np.ones_like(t), np.sin(2 * np.pi * (t - t0) / period), np.cos(2 * np.pi * (t - t0) / period)])
        beta, *_ = np.linalg.lstsq(x / e[:, None], y / e, rcond=None)
        return beta, x, float(np.sum(((y - x @ beta) / e) ** 2))

    periods = period_guess * np.linspace(0.98, 1.02, 401)
    chi2 = np.array([linear(p)[2] for p in periods])
    best = int(np.argmin(chi2))
    # the period's error from the chi2 curvature (delta chi2 = 1), scaled by the reduced chi2
    beta, x, c = linear(periods[best])
    reduced = c / max(len(t) - 4, 1)
    lo, hi = max(best - 5, 0), min(best + 6, len(periods))
    curvature = np.polyfit(periods[lo:hi], chi2[lo:hi], 2)[0]
    period_err = float(np.sqrt(reduced / curvature)) if curvature > 0 else np.nan
    cov = np.linalg.inv((x / e[:, None]).T @ (x / e[:, None])) * max(reduced, 1.0)
    amplitude = float(np.hypot(beta[1], beta[2]))
    grad = np.array([0.0, beta[1], beta[2]]) / max(amplitude, 1e-30)
    return SinusoidFit(amplitude, float(np.sqrt(grad @ cov @ grad)), float(periods[best]), period_err,
                       float(np.arctan2(beta[2], beta[1]) - 2 * np.pi * t0 / periods[best]))


def dip_depth(time: np.ndarray, ratio: np.ndarray, err: np.ndarray, window: tuple[float, float]) -> tuple[float, float]:
    """Mean of 1 - ratio inside the window, and its error (weighted), relative to the ratio around it."""
    t, y, e = (np.asarray(v, float) for v in (time, ratio, err))
    inside = (t >= window[0]) & (t <= window[1]) & np.isfinite(y)
    width = window[1] - window[0]
    around = (np.abs(t - 0.5 * (window[0] + window[1])) <= 1.5 * width + 0.05) & ~inside & np.isfinite(y)
    if not inside.any() or not around.any():
        return np.nan, np.nan
    w_in, w_out = 1 / e[inside] ** 2, 1 / e[around] ** 2
    level_in = np.sum(w_in * y[inside]) / w_in.sum()
    level_out = np.sum(w_out * y[around]) / w_out.sum()
    return float(level_out - level_in), float(np.sqrt(1 / w_in.sum() + 1 / w_out.sum()))


FLAG_BITS = {3: "BACKGROUND_GRADIENT", 4: "BACKGROUND_POOR_FIT", 5: "STELLAR_MODEL_UNCERTAIN", 6: "STELLAR_VARIABILITY"}
AMPLITUDE_TOLERANCE = 0.03  # criterion 3
PERIOD_TOLERANCE = 0.001
NEW_FLAG_LIMIT = 0.01  # criterion 3 (amended 2026-09-26): flags raised by the injection on at most 1 % of the points


@dataclass
class Recovery:
    """The injected signal against its recovery, for criteria 3 and 5 (see :func:`recovery`)."""

    rows: int
    gain: float  # recovered over injected flux (least squares over all points)
    pull_rms: float  # (ratio - expected) / RATIO_ERR on points flagged in neither run
    pull_max: float
    amplitude: SinusoidFit  # fitted to the recovered factor outside the dips
    amplitude_expected: float  # the same fit on the expected factor (a group averages the wave down)
    period: float  # injected
    dips: Table  # START, STOP, CROSSING, FLAGGED, DEPTH, DEPTH_ERR, EXPECTED
    flags: Table  # BIT, NAME, NEW, CLEARED, NEW_SIGNAL, CLEARED_SIGNAL (mean A - 1, %)
    changed_pull_max: float  # largest |pull| on points whose flags changed (raised or cleared)
    crossing_points: int
    crossing_flagged: int
    crossing_failures: int  # unflagged crossing points beyond 3 sigma

    @property
    def new_flags(self) -> int:
        """Points with a flag the plain run did not have (a point may count once per bit)."""
        return int(np.sum(self.flags["NEW"]))

    def criterion3(self) -> bool:
        """Amplitude within 3 % and period within 0.1 %; every unflagged dip within 1 sigma in depth; flags
        raised by the injection on at most 1 % of the points, and the flux recovered within 1 sigma wherever a
        flag changed."""
        fit = self.amplitude
        unflagged = self.dips[~self.dips["FLAGGED"] & np.isfinite(self.dips["DEPTH"])]
        return bool(
            abs(fit.amplitude / self.amplitude_expected - 1) <= AMPLITUDE_TOLERANCE
            and abs(fit.period / self.period - 1) <= PERIOD_TOLERANCE
            and np.all(np.abs(unflagged["DEPTH"] - unflagged["EXPECTED"]) <= unflagged["DEPTH_ERR"])
            and self.new_flags <= NEW_FLAG_LIMIT * self.rows
            and not self.changed_pull_max > 1.0
        )

    def criterion5(self) -> bool:
        """Every point at a bright-star crossing recovers the injected signal within 3 sigma or is flagged."""
        return self.crossing_failures == 0


def recovery(plain_lightcurve, injected_lightcurve, signal) -> Recovery:
    """Compare an injected run with the plain one.

    ``signal`` is the injected factor (e.g. :class:`tess_sso.injection.InjectedSignal`): callable on SPOC
    times, with ``amplitude``, ``period``, ``depth``, ``crossing`` and ``isolated`` dip windows.
    """
    hdul_p = plain_lightcurve if isinstance(plain_lightcurve, fits.HDUList) else fits.open(plain_lightcurve)
    hdul_i = injected_lightcurve if isinstance(injected_lightcurve, fits.HDUList) else fits.open(injected_lightcurve)
    rec = recovered_factor(hdul_p, hdul_i)
    row_of_group = {int(g): i for i, g in enumerate(hdul_i["LIGHTCURVE_PRF"].data["GROUP_ID"])}
    expected_all = expected_factor(hdul_i, signal)
    groups = hdul_p["LIGHTCURVE_PRF"].data["GROUP_ID"][rec["ROW"]]
    exp = np.array([expected_all[row_of_group[int(g)]] for g in groups])
    t = np.asarray(hdul_p["EXTRAS"].data["ORIGINAL_TIME"], float)[rec["ROW"]]
    ratio, err = np.asarray(rec["RATIO"], float), np.asarray(rec["RATIO_ERR"], float)
    q_p, q_i = np.asarray(rec["SSO_PLAIN"]), np.asarray(rec["SSO_INJECTED"])
    finite = np.isfinite(ratio) & np.isfinite(err) & (err > 0)
    clean = finite & (q_p == 0) & (q_i == 0)
    pull = (ratio - exp) / err

    flux = np.asarray(hdul_p["LIGHTCURVE_PRF"].data["FLUX"], float)[rec["ROW"]]
    injected, got = (exp - 1) * flux, (ratio - 1) * flux
    gain = float(np.sum(got[finite] * injected[finite]) / np.sum(injected[finite] ** 2))

    windows = list(signal.crossing) + list(signal.isolated)
    out = clean & ~in_windows(t, windows)
    fit = fit_sinusoid(t[out], ratio[out], err[out], signal.period)
    expected_fit = fit_sinusoid(t[out], exp[out], err[out], signal.period)
    wave = 1 + fit.amplitude * np.sin(2 * np.pi * t / fit.period + fit.phase)
    dip_rows = []
    for k, w in enumerate(windows):
        inside = in_windows(t, [w])
        flagged = not (clean & inside).any()
        use = finite if flagged else clean  # a flagged dip is measured anyway, and reported
        if (use & inside).any():
            depth, depth_err = dip_depth(t[use], (ratio / wave)[use], err[use], w)
            expected, _ = dip_depth(t[use], (exp / wave)[use], err[use], w)
        else:
            depth = depth_err = expected = np.nan
        dip_rows.append((w[0], w[1], k < len(signal.crossing), flagged, depth, depth_err, expected))
    dips_table = Table(rows=dip_rows, names=("START", "STOP", "CROSSING", "FLAGGED", "DEPTH", "DEPTH_ERR", "EXPECTED"),
                       dtype=(float, float, bool, bool, float, float, float))

    changed = np.zeros(len(rec), bool)
    flag_rows = []
    for bit, name in FLAG_BITS.items():
        new = ((q_p >> bit) & 1 == 0) & ((q_i >> bit) & 1 == 1)
        cleared = ((q_p >> bit) & 1 == 1) & ((q_i >> bit) & 1 == 0)
        changed |= new | cleared
        flag_rows.append((bit, name, int(new.sum()), int(cleared.sum()),
                          100 * float(np.mean(exp[new] - 1)) if new.any() else np.nan,
                          100 * float(np.mean(exp[cleared] - 1)) if cleared.any() else np.nan))
    flags_table = Table(rows=flag_rows, names=("BIT", "NAME", "NEW", "CLEARED", "NEW_SIGNAL", "CLEARED_SIGNAL"),
                        dtype=(int, str, int, int, float, float))

    d_p = hdul_p["LIGHTCURVE_PRF"].data
    in_cross = in_windows(np.asarray(rec["TIME"], float), crossing_windows(d_p["TIME"], d_p["STAR_FLUX"])) & finite
    return Recovery(
        rows=len(rec), gain=gain,
        pull_rms=float(np.sqrt(np.mean(pull[clean] ** 2))), pull_max=float(np.max(np.abs(pull[clean]))),
        amplitude=fit, amplitude_expected=expected_fit.amplitude, period=float(signal.period),
        dips=dips_table, flags=flags_table,
        changed_pull_max=float(np.max(np.abs(pull[changed & finite]))) if (changed & finite).any() else 0.0,
        crossing_points=int(in_cross.sum()), crossing_flagged=int((in_cross & (q_i != 0)).sum()),
        crossing_failures=int((in_cross & (q_i == 0) & (np.abs(pull) > 3)).sum()),
    )
