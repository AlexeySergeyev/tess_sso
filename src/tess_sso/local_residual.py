"""Time-local residual per pixel (Settings.residual_model = "local").

What the scattered-light and stellar models leave at a pixel -- faint stars,
errors of the bright stars' model, structure the smooth scattered-light
surface misses, pixel sensitivity -- is estimated from that pixel's own
residual (data - scattered light - stars) around the asteroid's passage: a
polynomial in time over a day on each side, fitted with the asteroid left
out (its footprint, and every frame with the asteroid within a few pixels),
then evaluated under the asteroid. It replaces the static residual (one
mean per pixel over a segment), which cannot follow what changes within a
day. The reference pipeline's background is of this kind (a spline in time
per pixel, the target masked); here it corrects tess_sso's models rather
than replacing them.

With ``residual_model = "local_pointing"`` the pixel's residual may also follow
the stellar model's pointing shift per frame, where that lowers the BIC: on a
bright star the model's error pulses with the pointing (3486, sector 45: a
4.3 h sawtooth, ±10 e-/s at a 1300 e-/s star), which a polynomial in time
cannot follow and the shift measured in every frame can.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

WINDOW_DAYS = 1.0  # fitted this long before and after the pixel's passage, within its segment
DEGREE = 2  # polynomial in time, when the fit has data on both sides of the passage; else a level
NEAR_PX = 3.0  # frames with the asteroid this close to the pixel are left out of the fit
MIN_POINTS = 30  # fewer usable frames: the static residual is kept
CLIP_SIGMA = 3.0
# The pointing terms are used only where the shifts' own measurement noise they carry into the
# residual is under this fraction of what they explain (35534, sector 55: unchecked, a drift's
# shifts added ±3-6 e-/s frame to frame; 3486, sector 45: 1.5 e-/s carried for 7.3 explained).
MAX_POINTING_NOISE = 0.5


def _pointing_terms(shift: np.ndarray, centre: np.ndarray) -> np.ndarray:
    """Row and column shifts about ``centre``: linear terms take nearly all of it (3486, sector 45:
    7.68 -> 2.50 e-/s, 2.40 with second-order terms, whose coefficients then fit the noise)."""
    return shift - centre


def fit_pixel(t: np.ndarray, resid: np.ndarray, err: np.ndarray, usable: np.ndarray, t_pred: np.ndarray,
              shift: np.ndarray | None = None, shift_pred: np.ndarray | None = None):
    """Residual predicted at ``t_pred`` from the ``usable`` samples, or ``None`` if there are too few.

    A polynomial of :data:`DEGREE` when at least a third of :data:`MIN_POINTS`
    lie on each side of ``t_pred``, otherwise (a segment's edge) a weighted
    level: a one-sided polynomial would be extrapolated.

    With ``shift`` (nframes, 2) and ``shift_pred``, the pointing shifts of the
    stellar model at the samples and at ``t_pred``, the residual may also follow
    the pointing: the time terms plus the shifts, linearly.
    Only the fast part of the shifts is used (each shift less its own
    polynomial in time): a slow drift is the time terms' (35534, sector 55:
    the row shift and time correlate at 0.99 over a window). The pointing
    terms are kept where they lower the BIC and where the shifts' frame-to-frame
    noise they carry stays under MAX_POINTING_NOISE of what they explain, so a
    pixel without a bright star keeps the time polynomial; ``shift_pred`` is
    clipped to the range the fit saw, so the pointing is never extrapolated.
    """
    fit = usable & np.isfinite(resid) & np.isfinite(err) & (err > 0)
    if shift is not None:
        fit &= np.isfinite(shift).all(axis=1)
    if fit.sum() < MIN_POINTS:
        return None
    both = ((fit & (t < t_pred.min())).sum() >= MIN_POINTS // 3
            and (fit & (t > t_pred.max())).sum() >= MIN_POINTS // 3)
    degree = DEGREE if both else 0
    t0 = float(t_pred.mean())
    x = np.vander(t - t0, degree + 1)
    x_pred = np.vander(t_pred - t0, degree + 1)
    jitter_noise = None
    if shift is not None and shift_pred is not None and np.isfinite(shift_pred).all():
        # the fast part of the shifts: each less its own polynomial in time over the fit's samples
        slow = np.linalg.lstsq(np.vander(t[fit] - t0, DEGREE + 1), shift[fit], rcond=None)[0]
        jitter = shift - np.vander(t - t0, DEGREE + 1) @ slow
        jitter_pred = shift_pred - np.vander(t_pred - t0, DEGREE + 1) @ slow
        centre = np.median(jitter[fit], axis=0)
        lo, hi = jitter[fit].min(axis=0), jitter[fit].max(axis=0)
        x_full = np.column_stack([x, _pointing_terms(jitter, centre)])
        x_full_pred = np.column_stack([x_pred, _pointing_terms(np.clip(jitter_pred, lo, hi), centre)])
        d2 = np.diff(np.where(fit[:, None], jitter, np.nan), 2, axis=0)
        jitter_noise = 1.4826 * np.nanmedian(np.abs(d2 - np.nanmedian(d2, axis=0)), axis=0) / np.sqrt(6)
    else:
        x_full = x_full_pred = None

    def clipped(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Coefficients and kept samples after iterative clipping (the time-only fit as it always was)."""
        keep, beta = fit.copy(), None
        for _ in range(3):
            beta, *_ = np.linalg.lstsq(a[keep] / err[keep, None], resid[keep] / err[keep], rcond=None)
            pull = (resid - a @ beta) / err
            scale = 1.4826 * np.median(np.abs(pull[keep]))
            new = fit & (np.abs(pull) < CLIP_SIGMA * max(scale, 1e-9))
            if new.sum() < MIN_POINTS or (new == keep).all():
                break
            keep = new
        return beta, keep

    def bic(a: np.ndarray, keep: np.ndarray) -> float:
        coef, *_ = np.linalg.lstsq(a[keep] / err[keep, None], resid[keep] / err[keep], rcond=None)
        return float((((resid[keep] - a[keep] @ coef) / err[keep]) ** 2).sum()) + a.shape[1] * np.log(keep.sum())

    beta, _ = clipped(x)
    if x_full is None:
        return x_pred @ beta
    beta_full, keep_full = clipped(x_full)
    # compared on the same samples: those the pointing model keeps (the time polynomial would clip the pulses)
    if bic(x_full, keep_full) >= bic(x, keep_full):
        return x_pred @ beta
    # what the pointing terms explain, against the shifts' measurement noise they carry into the prediction
    time_only, *_ = np.linalg.lstsq(x[keep_full] / err[keep_full, None], resid[keep_full] / err[keep_full], rcond=None)
    rms_time = np.std(resid[keep_full] - x[keep_full] @ time_only)
    rms_full = np.std(resid[keep_full] - x_full[keep_full] @ beta_full)
    explained = np.sqrt(max(rms_time**2 - rms_full**2, 0.0))
    carried = float(np.hypot(beta_full[degree + 1] * jitter_noise[0], beta_full[degree + 2] * jitter_noise[1]))
    if not np.isfinite(carried) or carried > MAX_POINTING_NOISE * explained:
        return x_pred @ beta
    return x_full_pred @ beta_full

def local_residuals(box, frames, first, last, flux_all, err_all, column, region, stellar, sl_by_seg, st_by_seg,
                    seg_of, masks, track, good_background, pointing: bool = False) -> tuple[int, int]:
    """Replace each box pixel's ``static`` residual (and its ``data``) with the time-local one.

    ``box`` maps a frame to its photometric box (``pix``, ``ok``, ``weight``,
    ``data`` with the static residual subtracted, ``static``);
    ``good_background`` marks frames whose scattered-light fit is usable.
    Returns (pixel-segments modelled, pixel-segments that kept the static residual).
    """
    t_all = frames.time
    covered = defaultdict(set)  # grid index -> frames with the asteroid's footprint on it
    for f in range(first, last + 1):
        for q in masks.footprints[f]:
            covered[int(q)].add(f)
    need = defaultdict(list)
    for f, b in box.items():
        for k, (r, c) in enumerate(b["pix"]):
            if b["ok"][k] and b["weight"][k] > 0:
                need[(int(r), int(c))].append((f, k))
    grid = masks.grid
    stars_by_seg = {}  # the stars' image and its shift terms over the region, per segment
    done = kept = 0
    for (r, c), items in need.items():
        ridx, col = region.get((r, c)), column.get((r, c))
        if ridx is None or col is None:
            continue
        on_pixel = covered.get((r - grid.origin[0]) * grid.shape[1] + (c - grid.origin[1]), set())
        box_frames = np.array([f for f, _ in items])
        for sid in np.unique(seg_of[box_frames]):
            sel = [(f, k) for f, k in items if seg_of[f] == sid]
            fs = np.array([f for f, _ in sel])
            seg_sl, seg_st = sl_by_seg[sid], st_by_seg[sid]
            lo = max(first, int(np.searchsorted(t_all, t_all[fs.min()] - WINDOW_DAYS)))
            hi = min(last, int(np.searchsorted(t_all, t_all[fs.max()] + WINDOW_DAYS)))
            w = np.arange(lo, hi + 1)
            w = w[(seg_of[w] == sid) & good_background[w]]
            if len(w) < MIN_POINTS:
                kept += 1
                continue
            sl = seg_sl.evaluate(np.array([[r, c]]), w - seg_sl.frames[0])[:, 0]
            if sid not in stars_by_seg:
                stars_by_seg[sid] = stellar.templates.project(seg_st.fluxes)
            amp = stars_by_seg[sid][:, ridx]  # the stars at this pixel, and their shift terms
            sh = seg_st.shifts[w - seg_st.frames[0]]
            resid = flux_all[w - first, col] - sl - (amp[0] + sh[:, 0] * amp[1] + sh[:, 1] * amp[2])
            away = (np.hypot(track.row[w] - r, track.col[w] - c) >= NEAR_PX) & ~np.isin(w, list(on_pixel))
            pred = fit_pixel(t_all[w], resid, err_all[w - first, col], away, t_all[fs],
                             *((sh, seg_st.shifts[fs - seg_st.frames[0]]) if pointing else ()))
            if pred is None:
                kept += 1
                continue
            for (f, k), value in zip(sel, pred):
                b = box[f]
                b["data"][k] = b["data"][k] + b["static"][k] - value
                b["static"][k] = value
            done += 1
    return done, kept
