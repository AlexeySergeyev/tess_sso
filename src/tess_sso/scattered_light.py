"""Scattered-light model: PCA of background pixels, with spatial profiles fitted by splines.

For each continuous observing segment:

1. Take the usable pixels (outside the star and asteroid masks) over the
   segment's good frames: a matrix F (frames x pixels).
2. Find its first ``n_components`` principal components (no mean
   subtraction, so the first one carries the overall level).
3. Fit each component's spatial profile with tensor-product cubic
   B-splines in (row, column) -- a Bayesian linear fit with the profile
   noise estimated from the residuals -- with iterative sigma clipping of
   outlying pixels. The splines predict the profile at masked pixels too.
4. Get every frame's temporal coefficients by projecting its usable pixels
   on the (orthonormal) PCA profiles, so frames excluded from the PCA (bad
   SPOC flags, missing pixels) still have a model. Projecting on the
   spline-fitted profiles instead is unstable: the noise components' fitted
   profiles are nearly zero and collinear.

The model is stored compactly (coefficients and spline weights) and
evaluated on demand for any pixels and frames. Spline-weight covariances
give the model variance and its covariance between pixels, which are
correlated and must not be summed in quadrature.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.interpolate import BSpline
from scipy.linalg import eigh

N_COMPONENTS = 8
KNOT_SPACING_PX = 20.0
# Gaussian prior on spline weights: mean = the profile's mean, sigma = PRIOR_SCALE x the profile's
# largest value. Together with the profile noise estimated from the fit residuals, it leaves
# well-constrained weights unbiased and keeps weights with little data near the mean level.
PRIOR_SCALE = 10.0
CLIP_SIGMA = 5.0
CLIP_ITERATIONS = 3
MIN_PIXELS_PER_BASIS = 5  # below this a spline basis function is poorly constrained
MIN_USABLE_PIXELS = 50  # below this a segment gets no model
DEGREE = 3


class FewBackgroundPixelsWarning(UserWarning):
    """Too few usable pixels to constrain the scattered-light model everywhere."""


@dataclass(frozen=True)
class SplineBasis:
    """Tensor-product cubic B-splines over a pixel bounding box, restricted to the columns in use."""

    knots_row: np.ndarray
    knots_col: np.ndarray
    columns: np.ndarray  # indices of the tensor-product functions kept

    @staticmethod
    def _knots(lo: float, hi: float, spacing: float) -> np.ndarray:
        n_inner = max(int(np.round((hi - lo) / spacing)) - 1, 0)
        inner = np.linspace(lo, hi, n_inner + 2)[1:-1]
        return np.concatenate([[lo] * (DEGREE + 1), inner, [hi] * (DEGREE + 1)])

    @classmethod
    def for_pixels(cls, pixels: np.ndarray, spacing: float = KNOT_SPACING_PX) -> "SplineBasis":
        """Knots over the pixels' bounding box (half a pixel beyond the edge pixels)."""
        rows, cols = pixels[:, 0].astype(float), pixels[:, 1].astype(float)
        kr = cls._knots(rows.min() - 0.5, rows.max() + 0.5, spacing)
        kc = cls._knots(cols.min() - 0.5, cols.max() + 0.5, spacing)
        full = cls(kr, kc, np.arange((len(kr) - DEGREE - 1) * (len(kc) - DEGREE - 1)))
        used = np.flatnonzero(np.abs(full.design(pixels)).sum(axis=0) > 0)
        return cls(kr, kc, used)

    def _axis(self, x: np.ndarray, knots: np.ndarray) -> np.ndarray:
        x = np.clip(np.asarray(x, dtype=float), knots[0], np.nextafter(knots[-1], -np.inf))
        return BSpline.design_matrix(x, knots, DEGREE).toarray()

    def design(self, pixels: np.ndarray) -> np.ndarray:
        """Design matrix (npixels, ncolumns) at pixel centres."""
        br = self._axis(pixels[:, 0], self.knots_row)
        bc = self._axis(pixels[:, 1], self.knots_col)
        full = (br[:, :, None] * bc[:, None, :]).reshape(len(pixels), -1)
        return full[:, self.columns]

    def gradient(self, row: float, col: float) -> tuple[np.ndarray, np.ndarray]:
        """d(design)/d(row) and d(design)/d(col) at one point, by central differences."""
        h = 0.05
        p = np.array([[row + h, col], [row - h, col], [row, col + h], [row, col - h]])
        d = self.design(p)
        return (d[0] - d[1]) / (2 * h), (d[2] - d[3]) / (2 * h)


@dataclass
class SegmentModel:
    segment: int
    frames: np.ndarray  # frame indices of this segment
    temporal: np.ndarray  # (nframes, ncomponents)
    weights: np.ndarray  # (nbasis, ncomponents): spatial profile k = design @ weights[:, k]
    covariance: np.ndarray  # (ncomponents, nbasis, nbasis): covariance of each weight vector
    basis: SplineBasis
    n_usable: int
    n_clipped: int
    frame_rms: np.ndarray  # per frame: RMS over usable pixels of the time-varying residual
    static_rms: float  # RMS of each usable pixel's median residual (e.g. faint unmasked stars)
    poorly_constrained: np.ndarray  # (nbasis,) True where a basis function has too few usable pixels
    frame_mad: np.ndarray | None = None  # per frame: median absolute deviation of the time-varying residual
    frame_chi2: np.ndarray | None = None  # per frame: mean (residual / pixel error)^2; None without errors
    fit_pixels: np.ndarray | None = None  # (n_usable, 2): the pixels the fit kept (1-based row, col)

    def evaluate(self, pixels: np.ndarray, frames: np.ndarray | None = None) -> np.ndarray:
        """Model flux (nframes, npixels); ``frames`` are indices into this segment."""
        coef = self.temporal if frames is None else self.temporal[frames]
        return coef @ (self.basis.design(pixels) @ self.weights).T

    def variance(self, pixels: np.ndarray, frames: np.ndarray | None = None) -> np.ndarray:
        """Model variance (nframes, npixels).

        Two terms: the spline-weight covariances (spatial profile error) and
        the temporal-coefficient error, whose variance is the frame's
        residual variance for each orthonormal PCA profile. On synthetic
        data the pulls of the model error have a standard deviation of
        about 1.2-1.4 with both terms (about 2-4 with the first alone).
        """
        coef = self.temporal if frames is None else self.temporal[frames]
        rms = self.frame_rms if frames is None else self.frame_rms[frames]
        per_component, fitted_sq = self.variance_terms(pixels)
        return coef**2 @ per_component + (rms**2)[:, None] * fitted_sq[None, :]

    def variance_terms(self, pixels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The frame-independent parts of :meth:`variance`: per-component spatial variance (k, npix)
        and the squared norm of the fitted profiles (npix,). The variance of frame t is
        ``temporal[t]**2 @ per_component + frame_rms[t]**2 * fitted_sq``."""
        x = self.basis.design(pixels)
        per_component = np.einsum("pi,kij,pj->kp", x, self.covariance, x)
        fitted = x @ self.weights  # (npix, k)
        return per_component, (fitted**2).sum(axis=1)

    def pixel_covariance(self, pixels: np.ndarray, frame: int) -> np.ndarray:
        """Model covariance between ``pixels`` in one frame (npixels, npixels), both terms."""
        x = self.basis.design(pixels)
        coef = self.temporal[frame]
        fitted = x @ self.weights
        spatial = sum(coef[k] ** 2 * x @ self.covariance[k] @ x.T for k in range(len(coef)))
        return spatial + self.frame_rms[frame] ** 2 * fitted @ fitted.T

    def constrained(self, pixels: np.ndarray) -> np.ndarray:
        """False where a poorly constrained basis function contributes to the pixel."""
        x = self.basis.design(pixels)
        return ~(np.abs(x[:, self.poorly_constrained]).sum(axis=1) > 1e-6)

    def usable_fraction(self, pixels: np.ndarray) -> float:
        """Share of ``pixels`` (npix, 2) that the fit used; NaN if unknown or ``pixels`` is empty."""
        if self.fit_pixels is None or len(pixels) == 0:
            return np.nan
        return float(np.isin(_pixel_keys(pixels), _pixel_keys(self.fit_pixels)).mean())

    def gradient(self, row: float, col: float, frames: np.ndarray | None = None) -> np.ndarray:
        """|grad model| (e-/s per pixel) at one point, per frame."""
        gr, gc = self.basis.gradient(row, col)
        coef = self.temporal if frames is None else self.temporal[frames]
        return np.hypot(coef @ (gr @ self.weights), coef @ (gc @ self.weights))


def _top_components(matrix: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """First ``n`` right singular vectors (n, npix) and singular values, via the frame Gram matrix."""
    gram = matrix @ matrix.T
    n = min(n, gram.shape[0])
    vals, vecs = eigh(gram, subset_by_index=[gram.shape[0] - n, gram.shape[0] - 1])
    order = np.argsort(vals)[::-1]
    s = np.sqrt(np.clip(vals[order], 0, None))
    u = vecs[:, order]
    v = (u.T @ matrix) / np.where(s > 0, s, 1)[:, None]
    return v, s


def _pixel_keys(pixels: np.ndarray) -> np.ndarray:
    """One integer per (row, col) pixel, for set operations."""
    pixels = np.asarray(pixels, np.int64)
    return pixels[:, 0] * 100_000 + pixels[:, 1]


def _mad_sigma(x: np.ndarray) -> float:
    return 1.4826 * float(np.median(np.abs(x - np.median(x))))


def fit_segment(
    segment: int,
    frames: np.ndarray,
    good: np.ndarray,
    flux: np.ndarray,
    usable_pixels: np.ndarray,
    strip_pixels: np.ndarray,
    n_components: int = N_COMPONENTS,
    knot_spacing: float = KNOT_SPACING_PX,
    prior_scale: float = PRIOR_SCALE,
    clip_sigma: float = CLIP_SIGMA,
    clip_iterations: int = CLIP_ITERATIONS,
    flux_err: np.ndarray | None = None,
) -> SegmentModel:
    """Fit one segment.

    ``flux`` (and ``flux_err``, for the per-frame chi2) is (len(frames), len(usable_pixels)); ``good`` marks the frames
    used for the PCA. ``strip_pixels`` sets the spline domain, so the model
    covers masked pixels too.
    """
    basis = SplineBasis.for_pixels(strip_pixels, knot_spacing)
    finite_cols = np.isfinite(flux[good]).mean(axis=0) > 0.99
    pix = usable_pixels[finite_cols]
    data = flux[:, finite_cols]
    pca_input = data[good]
    col_median = np.nanmedian(pca_input, axis=0)
    pca_input = np.where(np.isfinite(pca_input), pca_input, col_median)
    profiles, _ = _top_components(pca_input, n_components)  # (k, npix)

    x = basis.design(pix)
    k = profiles.shape[0]
    prior_mean = profiles.mean(axis=1)  # (k,): every weight starts at the profile's mean level
    prior_var = (prior_scale * np.maximum(np.abs(profiles).max(axis=1), 1e-12)) ** 2  # (k,)
    eye = np.eye(x.shape[1])

    def solve(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Posterior weights (nbasis, k), covariances (k, nbasis, nbasis), and profile noise (k,)."""
        xr = x[rows]
        gram = xr.T @ xr
        weights = np.empty((x.shape[1], k))
        covs = np.empty((k, x.shape[1], x.shape[1]))
        noise = np.empty(k)
        dof = max(int(rows.sum()) - x.shape[1], 1)
        for j in range(k):
            v = profiles[j, rows]
            w0 = np.linalg.solve(gram + 1e-9 * np.trace(gram) / len(gram) * eye, xr.T @ v)
            noise_var = max(float(((v - xr @ w0) ** 2).sum()) / dof, 1e-30)
            noise[j] = noise_var
            precision = gram / noise_var + eye / prior_var[j]
            covs[j] = np.linalg.inv(precision)
            weights[:, j] = covs[j] @ (xr.T @ v / noise_var + prior_mean[j] / prior_var[j])
        return weights, covs, noise

    keep = np.ones(len(pix), dtype=bool)
    weights, covariance, noise = solve(keep)
    for _ in range(clip_iterations):
        # residual of each component in units of its own noise, so a spike in a smooth
        # component is not drowned by the noise-like components
        chi = np.sqrt((((profiles.T - x @ weights) ** 2) / noise).sum(axis=1))
        new_keep = np.abs(chi - np.median(chi[keep])) <= clip_sigma * _mad_sigma(chi[keep])
        if np.array_equal(new_keep, keep):
            break
        keep = new_keep
        weights, covariance, noise = solve(keep)
    xk = x[keep]

    # temporal coefficients for every frame: projection of its usable pixels on the PCA profiles
    fitted = x @ weights  # (npix, k): smooth profiles, used for the model and its residuals
    temporal = np.full((len(frames), k), np.nan)
    for i in range(len(frames)):
        ok = keep & np.isfinite(data[i])
        if ok.sum() <= k:
            continue
        coef, *_ = np.linalg.lstsq(profiles[:, ok].T, data[i, ok], rcond=None)
        temporal[i] = coef
    # Residuals split into a static part (each pixel's median over the segment: faint
    # unmasked stars, left for the stellar model) and a time-varying part (noise and misfit).
    residual = (data[:, keep] - temporal @ fitted[keep].T).astype(np.float32)
    static = np.nanmedian(residual, axis=0)
    varying = residual - static
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)  # frames without finite pixels
        frame_rms = np.sqrt(np.nanmean(varying**2, axis=1)).astype(np.float64)
        frame_mad = np.nanmedian(np.abs(varying - np.nanmedian(varying, axis=1)[:, None]), axis=1).astype(np.float64)
        frame_chi2 = None
        if flux_err is not None:
            err = flux_err[:, finite_cols][:, keep]
            pull2 = np.where(err > 0, (varying / err) ** 2, np.nan)
            frame_chi2 = np.nanmean(pull2, axis=1).astype(np.float64)
    static_rms = float(np.sqrt(np.nanmean(static**2)))

    support = (np.abs(xk) > 1e-6).sum(axis=0)
    return SegmentModel(
        segment=segment,
        frames=np.asarray(frames),
        temporal=temporal,
        weights=weights,
        covariance=covariance,
        basis=basis,
        n_usable=int(keep.sum()),
        n_clipped=int((~keep).sum()),
        frame_rms=frame_rms,
        static_rms=static_rms,
        poorly_constrained=support < MIN_PIXELS_PER_BASIS,
        frame_mad=frame_mad,
        frame_chi2=frame_chi2,
        fit_pixels=pix[keep],
    )


@dataclass
class ScatteredLightModel:
    segments: list[SegmentModel] = field(default_factory=list)

    def segment_of(self, frame: int) -> SegmentModel:
        for seg in self.segments:
            if seg.frames[0] <= frame <= seg.frames[-1]:
                return seg
        raise IndexError(f"frame {frame} is in no fitted segment")

    def evaluate(self, pixels: np.ndarray, frames: Sequence[int]) -> np.ndarray:
        """Model (len(frames), npixels) for absolute frame indices."""
        frames = np.asarray(frames)
        out = np.full((len(frames), len(pixels)), np.nan)
        for seg in self.segments:
            inside = (frames >= seg.frames[0]) & (frames <= seg.frames[-1])
            if inside.any():
                out[inside] = seg.evaluate(pixels, frames[inside] - seg.frames[0])
        return out

    def describe(self) -> str:
        lines = []
        for seg in self.segments:
            lines.append(
                f"segment {seg.segment}: {seg.n_usable} usable px ({seg.n_clipped} clipped), "
                f"{len(seg.basis.columns)} spline functions ({int(seg.poorly_constrained.sum())} poorly constrained), "
                f"residual RMS {np.nanmedian(seg.frame_rms):.2f} e-/s time-varying, {seg.static_rms:.2f} static"
            )
        return "\n".join(lines)


def warn_if_poorly_constrained(model: ScatteredLightModel, pixels: np.ndarray, label: str) -> None:
    """Warn when ``pixels`` (e.g. the photometric cutout) depend on poorly constrained splines."""
    for seg in model.segments:
        bad = ~seg.constrained(pixels)
        if bad.any():
            warnings.warn(
                f"{label}, segment {seg.segment}: {int(bad.sum())} of {len(pixels)} pixels rely on "
                f"spline functions with fewer than {MIN_PIXELS_PER_BASIS} usable background pixels; "
                "widen the strip",
                FewBackgroundPixelsWarning,
                stacklevel=2,
            )


def write_model(path, model: ScatteredLightModel, header: dict) -> None:
    """Save the compact model: per segment, coefficients, spline weights, and covariances."""
    from astropy.io import fits

    from tess_sso.storage.fits_io import write_atomic

    primary = fits.PrimaryHDU()
    primary.header["NSEG"] = len(model.segments)
    for key, value in header.items():
        primary.header[key] = value
    hdus = [primary]
    for seg in model.segments:
        s = seg.segment
        info = fits.ImageHDU(seg.temporal.astype(np.float64), name=f"SL{s}_TEMPORAL")
        info.header["SEGMENT"] = s
        info.header["FRAME0"] = int(seg.frames[0])
        info.header["FRAME1"] = int(seg.frames[-1])
        info.header["NUSABLE"] = seg.n_usable
        info.header["NCLIP"] = seg.n_clipped
        info.header["STATRMS"] = seg.static_rms
        hdus += [
            info,
            fits.ImageHDU(seg.weights, name=f"SL{s}_WEIGHTS"),
            fits.ImageHDU(seg.covariance, name=f"SL{s}_COV"),
            fits.ImageHDU(seg.basis.knots_row, name=f"SL{s}_KNOTSR"),
            fits.ImageHDU(seg.basis.knots_col, name=f"SL{s}_KNOTSC"),
            fits.ImageHDU(seg.basis.columns.astype(np.int32), name=f"SL{s}_COLUMNS"),
            fits.ImageHDU(seg.poorly_constrained.astype(np.uint8), name=f"SL{s}_POOR"),
            fits.ImageHDU(seg.frame_rms, name=f"SL{s}_RMS"),
        ]
        if seg.frame_mad is not None:
            hdus.append(fits.ImageHDU(seg.frame_mad, name=f"SL{s}_MAD"))
        if seg.frame_chi2 is not None:
            hdus.append(fits.ImageHDU(seg.frame_chi2, name=f"SL{s}_CHI2"))
        if seg.fit_pixels is not None:
            hdus.append(fits.ImageHDU(seg.fit_pixels.astype(np.int32), name=f"SL{s}_PIXELS"))
    write_atomic(path, fits.HDUList(hdus))


def read_model(path) -> ScatteredLightModel:
    from astropy.io import fits

    with fits.open(path) as hdul:

        def optional(name, dtype=None):  # absent in files written before phase 9c
            return np.array(hdul[name].data, dtype=dtype) if name in hdul else None

        segments = []
        for s in range(hdul[0].header["NSEG"]):
            info = hdul[f"SL{s}_TEMPORAL"]
            segments.append(
                SegmentModel(
                    segment=info.header["SEGMENT"],
                    frames=np.arange(info.header["FRAME0"], info.header["FRAME1"] + 1),
                    temporal=np.array(info.data),
                    weights=np.array(hdul[f"SL{s}_WEIGHTS"].data),
                    covariance=np.array(hdul[f"SL{s}_COV"].data),
                    basis=SplineBasis(
                        np.array(hdul[f"SL{s}_KNOTSR"].data),
                        np.array(hdul[f"SL{s}_KNOTSC"].data),
                        np.array(hdul[f"SL{s}_COLUMNS"].data, dtype=np.int64),
                    ),
                    n_usable=info.header["NUSABLE"],
                    n_clipped=info.header["NCLIP"],
                    frame_rms=np.array(hdul[f"SL{s}_RMS"].data),
                    static_rms=info.header["STATRMS"],
                    poorly_constrained=np.array(hdul[f"SL{s}_POOR"].data, dtype=bool),
                    frame_mad=optional(f"SL{s}_MAD"),
                    frame_chi2=optional(f"SL{s}_CHI2"),
                    fit_pixels=optional(f"SL{s}_PIXELS", np.int64),
                )
            )
    return ScatteredLightModel(segments)
