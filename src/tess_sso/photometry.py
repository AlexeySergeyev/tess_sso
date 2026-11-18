"""Grouping of exposures and asteroid photometry.

Each exposure contributes a small box of pixels centred on the pixel that
contains the asteroid (a whole-pixel alignment) with the scattered light and
the stellar model already subtracted. The sub-pixel position stays in the
motion-smeared PRF of that exposure.

Primary measurement: one amplitude fitted jointly to all exposures of a
group, each with its own PRF, weights, and masks. Alternative: aperture
photometry on the weighted-mean stack, with an aperture drawn on the stack
PRF. Both estimates are linear in the data, F = sum a_tp D_tp, so the three
uncertainty components follow exactly from the coefficients a: pixel noise,
the stellar-flux covariance, and the scattered-light model covariance (the
last two add coherently across the exposures of a group).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

APERTURE_THRESHOLD = 0.02  # aperture: stack-PRF pixels at or above this fraction
OFFSET_SIGNIFICANCE = 3.0  # a segment's offset shifts the PRF only when it is this many sigma from zero
OFFSET_CLIP = 5.0  # groups this many sigma from the segment's offset are left out of it
PRF_POSITIONS = ("segment", "group", "predicted")
MIN_COVERAGE = 0.5  # groups with fewer accepted exposures than this fraction of the slots are flagged


@dataclass
class Group:
    index: int
    segment: int
    frames: np.ndarray  # accepted member frames (absolute indices)
    expected: int  # exposure slots the group spans
    incomplete: bool  # cut short by the segment edge

    @property
    def coverage(self) -> float:
        return len(self.frames) / max(self.expected, 1)


def make_groups(
    times: np.ndarray,
    segment_ids: np.ndarray,
    accepted: np.ndarray,
    cadence_days: float,
    duration_days: float | None = None,
    count: int | None = None,
) -> list[Group]:
    """Contiguous, non-overlapping groups inside each segment.

    ``duration_days`` (the main parameter) or ``count`` (exposures per group);
    neither means one exposure per group. Slots start at each segment's first
    frame; frames not ``accepted`` are left out but still count as slots.
    """
    if duration_days is not None and count is not None:
        raise ValueError("give duration_days or count, not both")
    groups: list[Group] = []
    for seg in np.unique(segment_ids[segment_ids >= 0]):
        idx = np.flatnonzero(segment_ids == seg)
        t = times[idx]
        if duration_days is not None:
            slot = np.floor((t - t[0]) / duration_days + 1e-9).astype(int)
            per_slot = max(int(round(duration_days / cadence_days)), 1)
            span_end = t[0] + (slot.max() + 1) * duration_days
            last_incomplete = span_end - t[-1] > cadence_days * 1.5
        else:
            n = 1 if count is None else count
            slot = np.arange(len(idx)) // n
            per_slot = n
            last_incomplete = len(idx) % n != 0
        for k in np.unique(slot):
            members = idx[slot == k]
            keep = members[accepted[members]]
            if len(keep) == 0:
                continue
            groups.append(
                Group(
                    index=len(groups),
                    segment=int(seg),
                    frames=keep,
                    expected=per_slot,
                    incomplete=bool(k == slot.max() and last_incomplete),
                )
            )
    return groups


@dataclass
class GroupData:
    """Per-exposure boxes of one group, all shaped (nexp, size, size); masked pixels have weight 0."""

    frames: np.ndarray
    origins: np.ndarray  # (nexp, 2): 1-based (row, col) of each box's [0, 0]
    data: np.ndarray  # flux - scattered light - stars
    weight: np.ndarray  # 1 / sigma^2, 0 where masked
    sigma: np.ndarray  # pixel noise
    prf: np.ndarray  # motion-smeared asteroid PRF, absolute normalization
    prf_dr: np.ndarray | None = None  # d(prf)/d(row) and d(prf)/d(col), for the position offset
    prf_dc: np.ndarray | None = None


def joint_prf_fit(g: GroupData, local_background: bool = False) -> tuple[float, float, np.ndarray, float]:
    """One amplitude for all exposures: (flux, pixel-noise error, coefficients a, reduced chi2).

    With ``local_background``, a constant background for the whole group is
    fitted with it: what the models leave in the box as a level no longer goes
    into the flux (the coefficients then sum to zero).
    """
    wp = g.weight * g.prf
    denom = float((wp * g.prf).sum())
    if denom <= 0:
        return np.nan, np.nan, np.zeros_like(g.data), np.nan
    used = g.weight > 0
    if local_background:
        normal = np.array([[denom, float(wp.sum())], [float(wp.sum()), float(g.weight.sum())]])
        if int(used.sum()) > 2 and np.linalg.det(normal) > 0:
            cov = np.linalg.inv(normal)
            flux, level = cov @ np.array([float((wp * g.data).sum()), float((g.weight * g.data).sum())])
            a = g.weight * (cov[0, 0] * g.prf + cov[0, 1])
            chi2 = float((g.weight * (g.data - flux * g.prf - level) ** 2).sum()) / max(int(used.sum()) - 2, 1)
            return float(flux), float(np.sqrt(cov[0, 0])), a, chi2
    flux = float((wp * g.data).sum()) / denom
    a = wp / denom
    dof = max(int(used.sum()) - 1, 1)
    chi2 = float((g.weight * (g.data - flux * g.prf) ** 2).sum()) / dof
    return flux, float(np.sqrt(1.0 / denom)), a, chi2


def prf_weights(g: GroupData) -> np.ndarray:
    """The PRF-weighted mean under the asteroid, summing to 1: what ``FLUX_BKG`` and ``STAR_FLUX`` project.

    The flux's own coefficients sum to zero with a local background, so a
    background level would project to nothing; these keep the channels a
    measure of what lies under the asteroid.
    """
    wp = g.weight * g.prf
    denom = float((wp * g.prf).sum())
    return wp / denom if denom > 0 else np.zeros_like(g.data)


def exposure_weights(g: GroupData) -> np.ndarray:
    """Per-exposure stacking weight: inverse of the median background variance of its box."""
    var = np.where(g.weight > 0, g.sigma**2, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # a fully masked box has no median
        med = np.nanmedian(var.reshape(len(g.frames), -1), axis=1)
    return np.where(np.isfinite(med) & (med > 0), 1.0 / med, 0.0)


@dataclass
class ApertureResult:
    flux: float
    err_pix: float
    fraction: float  # FLUX_FRACTION: stack-PRF sum over valid aperture pixels
    masked_fraction: float  # stack-PRF sum lost to masked aperture pixels
    a: np.ndarray  # coefficients, same shape as the group data
    stack: np.ndarray  # (size, size) weighted-mean image
    stack_prf: np.ndarray
    coverage: np.ndarray  # (size, size): number of exposures contributing per pixel
    aperture: np.ndarray  # bool (size, size)


def stack_weights(g: GroupData) -> np.ndarray:
    """Per-exposure, per-pixel weights of the stack (nexp, size, size): each pixel's usable exposures,
    weighted by :func:`exposure_weights` and normalized to sum to 1; 0 where no exposure is usable."""
    v = exposure_weights(g)[:, None, None] * (g.weight > 0)
    norm = v.sum(axis=0)
    with np.errstate(all="ignore"):
        return np.where(norm > 0, v / norm, 0.0)


def aperture_photometry(g: GroupData, threshold: float = APERTURE_THRESHOLD) -> ApertureResult:
    """Aperture photometry on the weighted-mean stack; masked pixels are left out of flux and fraction."""
    w_exp = exposure_weights(g)
    valid = g.weight > 0
    v = stack_weights(g)
    norm = v.sum(axis=0)
    stack = (v * g.data).sum(axis=0)
    wn = w_exp / w_exp.sum() if w_exp.sum() > 0 else np.full(len(w_exp), 1 / len(w_exp))
    stack_prf = (wn[:, None, None] * g.prf).sum(axis=0)
    aperture = stack_prf >= threshold
    covered = norm > 0
    use = aperture & covered
    fraction = float(stack_prf[use].sum())
    masked = float(stack_prf[aperture & ~covered].sum())
    if fraction <= 0:
        nan = np.full_like(g.data, 0.0)
        return ApertureResult(np.nan, np.nan, 0.0, masked, nan, stack, stack_prf, valid.sum(axis=0), aperture)
    a = v * use[None] / fraction
    flux = float((a * g.data).sum())
    err = float(np.sqrt((a**2 * g.sigma**2).sum()))
    return ApertureResult(flux, err, fraction, masked, a, stack, stack_prf, valid.sum(axis=0), aperture)


def fit_offset(g: GroupData) -> tuple[float, float, float, float]:
    """(row, col) offset of the asteroid from its predicted position, fitted over the group, and their errors.

    Linearized PRF fit, D = F (P + dr dP/dr + dc dP/dc), weighted like the
    flux fit and solved for (F, F dr, F dc). Valid for offsets well below a
    pixel; NaN without PRF derivatives or with too little signal. The errors
    propagate the fit covariance through the ratios.
    """
    nan = (np.nan, np.nan, np.nan, np.nan)
    if g.prf_dr is None or g.prf_dc is None:
        return nan
    x = np.stack([g.prf.ravel(), g.prf_dr.ravel(), g.prf_dc.ravel()], axis=1)
    w = g.weight.ravel()
    normal = x.T @ (x * w[:, None])
    try:
        beta = np.linalg.solve(normal, x.T @ (w * g.data.ravel()))
        cov = np.linalg.inv(normal)
    except np.linalg.LinAlgError:
        return nan
    if not beta[0] > 0:
        return nan
    dr, dc = beta[1] / beta[0], beta[2] / beta[0]
    var_r = (cov[1, 1] - 2 * dr * cov[0, 1] + dr**2 * cov[0, 0]) / beta[0] ** 2
    var_c = (cov[2, 2] - 2 * dc * cov[0, 2] + dc**2 * cov[0, 0]) / beta[0] ** 2
    return float(dr), float(dc), float(np.sqrt(max(var_r, 0.0))), float(np.sqrt(max(var_c, 0.0)))


@dataclass(frozen=True)
class SegmentOffset:
    """The asteroid's offset from its predicted position over one segment (row, col), pixels."""

    segment: int
    row: float
    col: float
    row_err: float  # inflated by the groups' excess scatter, if any
    col_err: float
    n_groups: int  # groups used, after clipping
    applied: bool  # significant, so the PRF is shifted by it (in "segment" mode)


def segment_offset(segment: int, offsets: np.ndarray, errors: np.ndarray) -> SegmentOffset:
    """Weighted mean of the groups' (row, col) offsets, clipped at OFFSET_CLIP sigma.

    ``offsets`` and ``errors`` are (ngroups, 2). The error of the mean is
    inflated by the square root of the reduced chi2 when the groups scatter
    more than their errors say (a drifting ephemeris, stars). The offset is
    applied when it lies OFFSET_SIGNIFICANCE sigma or more from zero.
    """
    offsets, errors = np.atleast_2d(offsets), np.atleast_2d(errors)
    ok = np.isfinite(offsets).all(axis=1) & np.isfinite(errors).all(axis=1) & (errors > 0).all(axis=1)
    o, e = offsets[ok], errors[ok]
    if len(o) == 0:
        return SegmentOffset(segment, 0.0, 0.0, np.nan, np.nan, 0, False)
    def fit(keep: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Weighted mean, its error, and the per-axis scale of the scatter over the errors (>= 1)."""
        w = 1.0 / e[keep] ** 2
        mean = (w * o[keep]).sum(axis=0) / w.sum(axis=0)
        n = int(keep.sum())
        reduced = (((o[keep] - mean) / e[keep]) ** 2).sum(axis=0) / (n - 1) if n > 1 else np.ones(2)
        scale = np.sqrt(np.maximum(reduced, 1.0))
        return mean, scale / np.sqrt(w.sum(axis=0)), scale

    keep = np.ones(len(o), bool)
    for _ in range(10):
        mean, err, scale = fit(keep)
        # clip against the errors inflated by the scatter, so tiny formal errors do not reject good groups
        new_keep = (np.abs(o - mean) <= OFFSET_CLIP * e * scale).all(axis=1)
        if not new_keep.any() or np.array_equal(new_keep, keep):
            break
        keep = new_keep
    mean, err, _ = fit(keep)
    n = int(keep.sum())
    applied = bool(np.hypot(*(mean / err)) >= OFFSET_SIGNIFICANCE)
    return SegmentOffset(segment, float(mean[0]), float(mean[1]), float(err[0]), float(err[1]), n, applied)


def prf_shifts(
    mode: str, segments: np.ndarray, offsets: np.ndarray, errors: np.ndarray
) -> tuple[np.ndarray, list[SegmentOffset]]:
    """(row, col) shift of the PRF for each group, and each segment's offset.

    ``segments``, ``offsets``, and ``errors`` are per group. ``mode`` is
    "segment" (the segment's offset where it is applied), "group" (each
    group's own offset; zero where it is not finite), or "predicted" (zero).
    """
    if mode not in PRF_POSITIONS:
        raise ValueError(f"prf_position must be one of {PRF_POSITIONS}, got {mode!r}")
    segments = np.asarray(segments)
    offsets, errors = np.asarray(offsets, float).reshape(-1, 2), np.asarray(errors, float).reshape(-1, 2)
    per_segment = [segment_offset(int(s), offsets[segments == s], errors[segments == s]) for s in np.unique(segments)]
    shifts = np.zeros((len(segments), 2))
    if mode == "segment":
        for so in per_segment:
            if so.applied:
                shifts[segments == so.segment] = (so.row, so.col)
    elif mode == "group":
        shifts = np.where(np.isfinite(offsets), offsets, 0.0)
    return shifts, per_segment


def star_variance(a: np.ndarray, star_design: np.ndarray, covariance: np.ndarray) -> float:
    """Stellar-flux covariance projected on the estimate: u^T C u with u_s = sum a * T_s.

    ``star_design`` is (nexp, ncomponents, size, size): each exposure's
    stellar templates (with its pointing shift) on its box.
    """
    u = np.einsum("tij,tsij->s", a, star_design)
    return float(u @ covariance @ u)


def variability_impact(a: np.ndarray, star_design: np.ndarray, step: np.ndarray) -> np.ndarray:
    """|change of the estimate| a step in each star's flux would cause (ncomponents,), e-/s.

    ``star_design`` as in :func:`star_variance`; ``step`` per component (NaN: none).
    """
    u = np.einsum("tij,tsij->s", a, star_design)
    return np.abs(np.nan_to_num(np.asarray(step, float)) * u)


def background_variance(
    a: np.ndarray, pixels: np.ndarray, frames_in_segment: np.ndarray, segment_model
) -> float:
    """Scattered-light covariance projected on the estimate: the sum of :func:`background_variance_parts`."""
    return sum(background_variance_parts(a, pixels, frames_in_segment, segment_model))


def background_variance_parts(
    a: np.ndarray, pixels: np.ndarray, frames_in_segment: np.ndarray, segment_model
) -> tuple[float, float]:
    """Scattered-light covariance projected on the estimate.

    (spatial, temporal). Spatial term: sum_k g_k^T C_k g_k with
    g_k = sum_t coef_k(t) sum_p a_tp x(p): the spline weights are shared by
    the whole segment, so this error is coherent from point to point (a
    systematic). Temporal term: sum_t rms(t)^2 sum_k (sum_p a_tp fitted_k(p))^2,
    independent between exposures (random). ``pixels`` is (nexp, npix, 2),
    matching ``a`` flattened per exposure.
    """
    k = segment_model.weights.shape[1]
    g = np.zeros((k, segment_model.weights.shape[0]))
    temporal = 0.0
    for t in range(len(a)):
        at = a[t].ravel()
        if not np.any(at):
            continue
        x = segment_model.basis.design(pixels[t])
        xa = at @ x  # (nbasis,)
        coef = segment_model.temporal[frames_in_segment[t]]
        g += coef[:, None] * xa[None, :]
        fitted = xa @ segment_model.weights  # (k,)
        temporal += segment_model.frame_rms[frames_in_segment[t]] ** 2 * float((fitted**2).sum())
    spatial = float(sum(g[j] @ segment_model.covariance[j] @ g[j] for j in range(k)))
    return spatial, float(temporal)


def white_excess(times: np.ndarray, fluxes: np.ndarray, errors: np.ndarray, cadence: float) -> float:
    """Extra white noise per exposure (e-/s) beyond ``errors``, from second differences of single exposures.

    Uses triples of consecutive exposures (gaps under 1.5 cadences). A smooth
    signal's curvature over one exposure is negligible. Second differences
    beyond 5 robust sigma (the edges of short events, cosmic rays) are left
    out; the excess makes the mean squared pull of the rest equal to one.
    Returns 0 when the errors already explain the scatter, NaN with fewer
    than 10 triples.
    """
    t, f, e = (np.asarray(v, float) for v in (times, fluxes, errors))
    ok = np.isfinite(t) & np.isfinite(f) & np.isfinite(e)
    t, f, e = t[ok], f[ok], e[ok]
    if len(t) < 3:
        return np.nan
    consecutive = (np.diff(t)[:-1] < 1.5 * cadence) & (np.diff(t)[1:] < 1.5 * cadence)
    d2 = (f[:-2] - 2 * f[1:-1] + f[2:])[consecutive]
    var = (e[:-2] ** 2 + 4 * e[1:-1] ** 2 + e[2:] ** 2)[consecutive]
    if len(d2) < 10:
        return np.nan
    pull = d2 / np.sqrt(var)
    scale = max(1.4826 * float(np.median(np.abs(pull))), 1.0)
    keep = np.abs(pull) <= 5 * scale
    d2, var = d2[keep], var[keep]

    def mean_square_pull(excess: float) -> float:
        return float(np.mean(d2**2 / (var + 6 * excess**2)))

    if mean_square_pull(0.0) <= 1.0:
        return 0.0
    lo, hi = 0.0, float(np.sqrt(np.mean(d2**2) / 6))
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if mean_square_pull(mid) > 1.0 else (lo, mid)
    return 0.5 * (lo + hi)


@dataclass
class GroupResult:
    group: Group
    time: float  # weighted mean of member mid-times (BTJD)
    weights: np.ndarray  # per member, used for every averaged quantity
    flux: float
    err_pix: float
    err_star: float  # coherent: the stellar fluxes are constant over a segment
    err_bkg: float  # random: the temporal scattered-light term
    red_chi2: float
    offset: tuple[float, float]
    aperture: ApertureResult
    ap_err_star: float
    ap_err_bkg: float  # random (temporal) part of the scattered-light error, like err_bkg
    # Static stellar-model residual (its mean on clean frames) projected on the flux: the correction
    # applied when Settings.static_correction is on, otherwise the bias left in the flux.
    star_residual: float = np.nan
    sso_quality: int = 0
    spoc_quality: int = 0
    offset_err: tuple[float, float] = (np.nan, np.nan)
    origins: np.ndarray | None = None  # (nexp, 2): 1-based (row, col) of each member box's [0, 0]
    member_flags: np.ndarray | None = None  # SSO_QUALITY of each member exposure
    # Instrumental channels, each projected with the estimate's own coefficients (independent of the
    # asteroid's brightness): the scattered light and the stellar model the flux had removed.
    bkg_flux: float = np.nan
    star_flux: float = np.nan
    ap_bkg_flux: float = np.nan
    ap_star_flux: float = np.nan
    ap_star_residual: float = np.nan
    bkg_std: float = np.nan  # weighted mean per-exposure RMS of the scattered-light fit residual
    bkg_gradient: float = np.nan  # weighted mean scattered-light gradient at the asteroid, e-/s/pixel
    bkg_mad: float = np.nan  # weighted mean per-exposure MAD of the scattered-light fit residual
    bkg_chi2: float = np.nan  # weighted mean per-exposure chi2 per pixel of that residual
    bkg_usable: float = np.nan  # weighted mean share of strip pixels near the asteroid used by the fit
    prf_fraction: float = np.nan  # weighted PRF sum over the usable pixels of the boxes
    images: dict | None = None  # stacked images for the pixel product, see group_images()
    prf_shift: tuple[float, float] = (0.0, 0.0)  # (row, col) shift of the PRF from the predicted position
    err_bkg_sys: float = 0.0  # coherent: the spatial scattered-light term
    ap_err_bkg_sys: float = 0.0
    err_excess: float = 0.0  # random: the segment's white excess, with this group's member weights
    ap_err_excess: float = 0.0
    ap_npix: int = 0  # usable aperture pixels on the stack

    @property
    def flux_err(self) -> float:
        """Random error, point to point: pixel noise, temporal scattered light, white excess."""
        return float(np.sqrt(self.err_pix**2 + self.err_bkg**2 + self.err_excess**2))

    @property
    def flux_err_sys(self) -> float:
        """Coherent error, shared by the points of a segment: stellar model and spatial scattered light."""
        return float(np.sqrt(self.err_star**2 + self.err_bkg_sys**2))

    @property
    def ap_flux_err(self) -> float:
        return float(np.sqrt(self.aperture.err_pix**2 + self.ap_err_bkg**2 + self.ap_err_excess**2))

    @property
    def ap_flux_err_sys(self) -> float:
        return float(np.sqrt(self.ap_err_star**2 + self.ap_err_bkg_sys**2))


def group_images(
    g: GroupData, flux: float, background: np.ndarray, background_sigma: np.ndarray, stars: np.ndarray
) -> dict[str, np.ndarray]:
    """Stacked images of one group for the pixel product, each (size, size), NaN where no exposure is usable.

    ``background``, ``background_sigma``, and ``stars`` are per exposure, shaped like
    ``g.data``; ``stars`` is everything the stellar model removed (with the static
    correction when it is applied). FLUX follows SPOC: the data minus scattered
    light, stars left in. The background-model error is coherent across the
    exposures, so its stack is the weighted mean of the per-exposure sigmas.
    """
    v = stack_weights(g)
    used = v.sum(axis=0) > 0

    def stack(values: np.ndarray) -> np.ndarray:
        return np.where(used, (v * np.where(v > 0, values, 0.0)).sum(axis=0), np.nan).astype(np.float32)

    model = flux * stack(g.prf) if np.isfinite(flux) else np.full(used.shape, np.nan, np.float32)
    return {
        "FLUX": stack(g.data + stars),
        "FLUX_ERR": np.where(used, np.sqrt((v**2 * g.sigma**2).sum(axis=0)), np.nan).astype(np.float32),
        "FLUX_BKG": stack(background),
        "FLUX_BKG_ERR": stack(background_sigma),
        "STAR_MODEL": stack(stars),
        "ASTEROID_MODEL": model,
        "RESIDUAL": stack(g.data) - model,
        "COVERAGE": (g.weight > 0).sum(axis=0).astype(np.int16),
    }


def member_weights(g: GroupData) -> np.ndarray:
    """Per-exposure weight of the joint PRF fit (its share of the information), summing to 1."""
    info = (g.weight * g.prf**2).reshape(len(g.frames), -1).sum(axis=1)
    total = info.sum()
    return info / total if total > 0 else np.full(len(info), 1.0 / len(info))


def prf_fraction(g: GroupData, weights: np.ndarray) -> float:
    """Share of the asteroid PRF on usable pixels, averaged over the members with ``weights``."""
    inside = (g.prf * (g.weight > 0)).reshape(len(g.frames), -1).sum(axis=1)
    return float(np.sum(weights * inside))


def weighted_mean(values, weights: np.ndarray) -> float:
    """Mean of the finite ``values`` with their ``weights``; NaN if none is finite or all weights are 0."""
    values, weights = np.asarray(values, float), np.asarray(weights, float)
    ok = np.isfinite(values) & (weights > 0)
    return float(np.sum(weights[ok] * values[ok]) / np.sum(weights[ok])) if ok.any() else np.nan


@dataclass
class Photometry:
    """All groups of one observation, with the member exposures and their weights."""

    groups: list[GroupResult] = field(default_factory=list)
    segment_offsets: list[SegmentOffset] = field(default_factory=list)
    excess: dict = field(default_factory=dict)  # segment -> white excess per exposure, e-/s
    model_error: dict = field(default_factory=dict)  # segment -> stellar-model error fraction (epsilon)
    tstart: np.ndarray | None = None  # per frame (BTJD), to report group start and stop times
    tstop: np.ndarray | None = None

    def to_table(self):
        from astropy.table import Table

        rows = []
        for r in self.groups:
            g = r.group
            rows.append({
                "group": g.index,
                "segment": g.segment,
                "time": r.time,
                "time_start": float(self.tstart[g.frames[0]]) if self.tstart is not None else np.nan,
                "time_stop": float(self.tstop[g.frames[-1]]) if self.tstop is not None else np.nan,
                "n_exp": len(g.frames),
                "n_expected": g.expected,
                "coverage": g.coverage,
                "exposure": float((self.tstop[g.frames] - self.tstart[g.frames]).sum()) if self.tstart is not None else np.nan,
                "flux": r.flux,
                "flux_err": r.flux_err,
                "flux_err_pix": r.err_pix,
                "flux_err_star": r.err_star,
                "flux_err_bkg": r.err_bkg,
                "flux_err_bkg_sys": r.err_bkg_sys,
                "flux_err_excess": r.err_excess,
                "flux_err_sys": r.flux_err_sys,
                "red_chi2": r.red_chi2,
                "star_residual": r.star_residual,
                "ap_flux": r.aperture.flux,
                "ap_flux_err": r.ap_flux_err,
                "ap_flux_err_sys": r.ap_flux_err_sys,
                "flux_fraction": r.aperture.fraction,
                "offset_row": r.offset[0],
                "offset_col": r.offset[1],
                "offset_row_err": r.offset_err[0],
                "offset_col_err": r.offset_err[1],
                "prf_shift_row": r.prf_shift[0],
                "prf_shift_col": r.prf_shift[1],
                "bkg_flux": r.bkg_flux,
                "bkg_std": r.bkg_std,
                "bkg_gradient": r.bkg_gradient,
                "bkg_mad": r.bkg_mad,
                "bkg_chi2": r.bkg_chi2,
                "bkg_usable": r.bkg_usable,
                "star_flux": r.star_flux,
                "prf_fraction": r.prf_fraction,
                "ap_bkg_flux": r.ap_bkg_flux,
                "ap_star_flux": r.ap_star_flux,
                "ap_star_residual": r.ap_star_residual,
                "ap_npix": r.ap_npix,
                "sso_quality": r.sso_quality,
                "quality": r.spoc_quality,
            })
        return Table(rows=rows) if rows else Table()

    def members(self):
        from astropy.table import Table

        return Table({
            "group": np.concatenate([np.full(len(r.group.frames), r.group.index) for r in self.groups] or [np.empty(0, int)]),
            "frame": np.concatenate([r.group.frames for r in self.groups] or [np.empty(0, int)]),
            "weight": np.concatenate([r.weights for r in self.groups] or [np.empty(0)]),
        })


def groups_path(path):
    """The member exposures of each group, next to the photometry: ``photometry_groups.ecsv``."""
    from pathlib import Path

    path = Path(path)
    return path.with_name(f"{path.stem}_groups{path.suffix}")


def write_photometry(path, phot: Photometry, header: dict) -> None:
    """One row per group in ``path`` (ECSV, ``header`` in its metadata); members in ``groups_path(path)``."""
    from pathlib import Path

    from tess_sso.storage.ecsv_io import write_table

    write_table(groups_path(path), phot.members(), header)
    write_table(Path(path), phot.to_table(), header)
