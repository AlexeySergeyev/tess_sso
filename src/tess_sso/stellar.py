"""Stellar model: Gaia stars rendered with the PRF, fluxes fitted per segment.

After the scattered light is subtracted, the pixels near the asteroid track
are modelled as

    D(p, t) = sum_s F_s * [P_s(p) + dr_t * dP_s/dr(p) + dc_t * dP_s/dc(p)]

with one constant flux F_s per star and continuous observing segment, and one
pointing shift (dr_t, dc_t) per exposure shared by all stars. The shift is
small, so the PRF is linearized around the catalogue position instead of
being re-evaluated for every star in every frame.

The asteroid is protected by construction: pixels inside its footprint in a
frame get zero weight, so the fluxes are fixed by the frames in which the
star is clean, and the model carries no free parameter during a crossing.

Only components whose core (pixels with at least CORE_FRACTION of the flux)
lies in the fit region are fitted. Stars whose wings merely reach the region
are held at their Gaia flux: fitted from their wings alone they absorb
whatever else is at the region's edge (on asteroid 95 one went from 449 to
7420 e-/s).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy import ndimage

from tess_sso.prf import window_origin

WINDOW = 15  # PRF stamp size for stars (pixels)
REGION_MARGIN_PX = 3  # fit region = photometric cutout dilated by this much
SATURATION_TMAG = 6.8
SATURATION_RADIUS_PX = 7
PAIR_PX = 0.5  # stars closer than this are fitted as one component
SQUARE = np.ones((3, 3), dtype=bool)  # dilation element: grow by whole pixels in rows and columns
PRIOR_FRACTION = 0.5  # Gaia-based prior: sigma = this x prior flux
PRIOR_FLOOR = 1.0  # e-/s, minimum prior sigma
N_ITERATIONS = 2  # flux -> shift rounds
BRIGHT_FOR_SHIFT = 20  # brightest components used to fit the pointing shift
CORE_FRACTION = 0.05  # a star's core: pixels holding at least this fraction of its flux
MIN_CLEAN_FRAMES = 20
PRIOR_DOMINATED = 0.5  # flux error > this x prior sigma: the data barely constrain the star
FIXED_SIGMA = 1e-3  # e-/s: prior sigma that holds a component at its Gaia flux
CHI2_LIMIT = 10.0  # reduced chi-square of the core above this: the model does not describe the star
VARIABILITY_WINDOW_DAYS = 0.5
VARIABILITY_Z = 3.0
MIN_ENSEMBLE = 10  # tested stars needed to calibrate the variability z across the segment's stars
MODEL_ERROR_BIN_DAYS = 1.0 / 24  # the residual's hourly means measure the model error, not photon noise
MODEL_ERROR_MIN_FLUX = 200.0  # e-/s: stars used for the model-error fraction
# In a dense field (the Gaia mask threshold raised, see masks.py) only stars whose fitted flux is
# within this factor of Gaia's enter the model-error fraction: where the background fails, fits
# 10-1000 x Gaia made it 1.3-2.8 (35534, sector 39), every point flagged; x3 gives 0.07.
DENSE_GAIA_FACTOR = 3.0
# Above this many components the covariance is computed per tile: square cells of components
# (by centre) fitted with their neighbours within TILE_MARGIN_PX, keeping each cell's block;
# the fluxes still solve the whole segment (conjugate gradients). Below it one tile holds
# everything: the whole inverse, as before tiles. In the Galactic plane (35534, sector 39:
# 46 000 components on 20 000 pixels) the whole inverse would take ~20 GB.
TILE_COMPONENTS = 2000
TILE_MARGIN_PX = 7  # half a PRF window: farther stars share no pixels with the cell's stars
MAX_TILE_PX, MIN_TILE_PX = 256, 4  # cell sizes tried, halving from the largest
CG_RTOL = 1e-10  # conjugate gradients, relative residual of the scaled system
CG_MAXITER = 50_000  # 35534 sector 39: ~11 000 from the prior, 2 min


@dataclass
class Components:
    """Stars to fit; a close pair is one component with fixed Gaia flux ratios."""

    rows: list[np.ndarray]  # per component: member rows (1-based pixel coordinates)
    cols: list[np.ndarray]
    ratios: list[np.ndarray]  # per component: member flux fractions (sum to 1)
    source_ids: list[np.ndarray]
    prior_flux: np.ndarray  # e-/s, from Gaia

    def __len__(self) -> int:
        return len(self.prior_flux)

    def centre(self, i: int) -> tuple[float, float]:
        return float(np.sum(self.rows[i] * self.ratios[i])), float(np.sum(self.cols[i] * self.ratios[i]))


def fit_region(cutout_pixels: np.ndarray, strip: np.ndarray, grid_origin: tuple[int, int]) -> np.ndarray:
    """Boolean image on the mask grid: the cutout dilated by REGION_MARGIN_PX, within the strip."""
    image = np.zeros(strip.shape, dtype=bool)
    idx = cutout_pixels - np.array(grid_origin)
    inside = (idx[:, 0] >= 0) & (idx[:, 0] < strip.shape[0]) & (idx[:, 1] >= 0) & (idx[:, 1] < strip.shape[1])
    image[idx[inside, 0], idx[inside, 1]] = True
    return ndimage.binary_dilation(image, SQUARE, iterations=REGION_MARGIN_PX) & strip


def select_components(
    source_id: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    tmag: np.ndarray,
    flux: np.ndarray,
    region: np.ndarray,
    grid_origin: tuple[int, int],
) -> tuple[Components, np.ndarray]:
    """Unsaturated stars whose PRF window touches the region, close pairs merged.

    Returns the components and a boolean image of pixels near saturated stars,
    which the fit excludes.
    """
    reach = ndimage.binary_dilation(region, SQUARE, iterations=WINDOW // 2)
    near_saturated = np.zeros(region.shape, dtype=bool)
    candidates = []
    for i in np.argsort(-flux):
        if not (np.isfinite(rows[i]) and np.isfinite(cols[i]) and np.isfinite(flux[i])):
            continue
        r, c = int(np.floor(rows[i] + 0.5)) - grid_origin[0], int(np.floor(cols[i] + 0.5)) - grid_origin[1]
        if tmag[i] <= SATURATION_TMAG:
            rr, cc = np.ogrid[: region.shape[0], : region.shape[1]]
            near_saturated |= (np.abs(rr - r) <= SATURATION_RADIUS_PX) & (np.abs(cc - c) <= SATURATION_RADIUS_PX)
            continue
        if 0 <= r < region.shape[0] and 0 <= c < region.shape[1] and reach[r, c]:
            candidates.append(i)
    groups: list[list[int]] = []
    # brightest first: a fainter star joins the earliest group whose leader is within PAIR_PX. The
    # leaders are found on a grid of PAIR_PX cells (the 3 x 3 around a star), not by scanning every
    # group: in the Galactic plane (35534, sector 39) that scan was 10^5 x 10^4 pairs.
    leaders: dict[tuple[int, int], list[int]] = {}
    for i in candidates:
        cell = (int(np.floor(rows[i] / PAIR_PX)), int(np.floor(cols[i] / PAIR_PX)))
        near = [
            g for dr in (-1, 0, 1) for dc in (-1, 0, 1) for g in leaders.get((cell[0] + dr, cell[1] + dc), ())
            if np.hypot(rows[i] - rows[groups[g][0]], cols[i] - cols[groups[g][0]]) < PAIR_PX
        ]
        if near:
            groups[min(near)].append(i)
        else:
            leaders.setdefault(cell, []).append(len(groups))
            groups.append([i])
    total = np.array([flux[g].sum() for g in groups])
    comps = Components(
        rows=[rows[g] for g in groups],
        cols=[cols[g] for g in groups],
        ratios=[flux[g] / flux[g].sum() for g in groups],
        source_ids=[source_id[g] for g in groups],
        prior_flux=total if len(groups) else np.empty(0),
    )
    return comps, near_saturated


@dataclass
class Templates:
    """Each component's PRF image and its row and column derivatives on the fit region.

    Three sparse (ncomponents, npixels) matrices, ``h[0]``, ``h[1]``, ``h[2]``:
    a star covers at most WINDOW x WINDOW of the region's pixels, so a dense
    array grows as components x pixels (~20 GB in the Galactic plane).
    """

    slots: tuple

    def __post_init__(self):
        from scipy import sparse

        self.slots = tuple(sparse.csr_matrix(s) for s in self.slots)
        for s in self.slots:
            s.eliminate_zeros()
            s.sort_indices()

    @classmethod
    def from_dense(cls, dense) -> "Templates":
        return cls(tuple(np.asarray(d, dtype=np.float64) for d in dense))

    @property
    def shape(self) -> tuple[int, int]:
        return self.slots[0].shape

    def __getitem__(self, k: int):
        return self.slots[k]

    def toarray(self) -> np.ndarray:
        return np.stack([s.toarray() for s in self.slots])

    def project(self, fluxes: np.ndarray) -> np.ndarray:
        """``fluxes`` through each slot: the stars' image and its shift terms (3, npixels)."""
        return np.stack([s.T @ fluxes for s in self.slots])

    def block(self, stars: np.ndarray, pixels: np.ndarray, shift: np.ndarray) -> np.ndarray:
        """Templates of ``stars`` on ``pixels`` with a pointing ``shift``, dense (nstars, npixels)."""
        sub = [s[stars][:, pixels].toarray() for s in self.slots]
        return sub[0] + shift[0] * sub[1] + shift[1] * sub[2]

    def touching(self, pixels: np.ndarray | None = None, threshold: float | None = None) -> np.ndarray:
        """Components with a nonzero PRF value (or at least ``threshold``) on ``pixels`` (default: all)."""
        sub = self.slots[0] if pixels is None else self.slots[0][:, pixels].tocsr()
        hit = sub.data != 0 if threshold is None else sub.data >= threshold
        rows = np.repeat(np.arange(sub.shape[0]), np.diff(sub.indptr))
        return np.bincount(rows[hit], minlength=sub.shape[0]) > 0


def templates(prf, comps: Components, pixels: np.ndarray, step: float = 0.05) -> Templates:
    """PRF images and their row/column derivatives on ``pixels``, sparse (see :class:`Templates`)."""
    pixels = np.asarray(pixels, dtype=np.int64).reshape(-1, 2)
    n, npix = len(comps), len(pixels)
    if npix == 0:
        return Templates.from_dense(np.zeros((3, n, 0)))
    lo = pixels.min(axis=0) - WINDOW
    shape = pixels.max(axis=0) - lo + WINDOW + 1
    lookup = np.full(tuple(shape), -1, dtype=np.int64)  # region index of each pixel of the bounding box
    lookup[pixels[:, 0] - lo[0], pixels[:, 1] - lo[1]] = np.arange(npix)
    member, column, values = [], [], [[], [], []]
    for i in range(n):
        r0, c0 = comps.centre(i)
        wr, wc = window_origin(r0, c0, WINDOW)
        a, b = wr - lo[0], wc - lo[1]
        ra, rb, ca, cb = max(a, 0), min(a + WINDOW, shape[0]), max(b, 0), min(b + WINDOW, shape[1])
        if ra >= rb or ca >= cb:
            continue
        idx = lookup[ra:rb, ca:cb]
        inside = idx >= 0
        if not inside.any():
            continue
        stamps = np.zeros((3, WINDOW, WINDOW))
        rows, cols, ratio = comps.rows[i], comps.cols[i], comps.ratios[i]
        # slot 0: the PRF; slots 1 and 2: central differences along rows and columns
        for dr, dc, sign, slot in ((0, 0, 1, 0), (step, 0, 1, 1), (-step, 0, -1, 1), (0, step, 1, 2), (0, -step, -1, 2)):
            imgs = prf.evaluate(rows + dr, cols + dc, (wr, wc), (WINDOW, WINDOW))
            img = (imgs * ratio[:, None, None]).sum(axis=0)
            stamps[slot] += img if slot == 0 else sign * img / (2 * step)
        window = stamps[:, ra - a : rb - a, ca - b : cb - b]
        member.append(np.full(int(inside.sum()), i))
        column.append(idx[inside])
        for slot in range(3):
            values[slot].append(window[slot][inside])
    from scipy import sparse

    member = np.concatenate(member) if member else np.empty(0, np.int64)
    column = np.concatenate(column) if column else np.empty(0, np.int64)
    return Templates(tuple(
        sparse.csr_matrix((np.concatenate(v) if v else np.empty(0), (member, column)), shape=(n, npix)) for v in values
    ))


def _moments(weight: np.ndarray, data: np.ndarray, shifts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel sums over frames, all the normal equations need of the data.

    ``m[i, j] = sum_t w e_i e_j`` (3, 3, npixels) and ``r[i] = sum_t w d e_i``
    (3, npixels), with ``e = (1, row shift, column shift)`` per frame.
    """
    e = np.column_stack([np.ones(len(shifts)), shifts[:, 0], shifts[:, 1]])
    m = np.empty((3, 3, weight.shape[1]))
    for i in range(3):
        for j in range(i, 3):
            m[i, j] = m[j, i] = (e[:, i] * e[:, j]) @ weight
    wd = weight * data
    return m, np.stack([e[:, i] @ wd for i in range(3)])


def _apply(h: Templates, m: np.ndarray, x: np.ndarray) -> np.ndarray:
    """The data part of the normal matrix times ``x``, without forming the matrix."""
    u = [h[j].T @ x for j in range(3)]
    return sum(h[i] @ (m[i, 0] * u[0] + m[i, 1] * u[1] + m[i, 2] * u[2]) for i in range(3))


def _block(h: Templates, m: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """The data part of the normal matrix on ``rows`` x ``rows``, dense."""
    hs = [h[i][rows] for i in range(3)]
    out = np.zeros((len(rows), len(rows)))
    for i in range(3):
        for j in range(3):
            out += (hs[i].multiply(m[i, j][None, :]).tocsr() @ hs[j].T).toarray()
    return out


def _rhs(h: Templates, r: np.ndarray) -> np.ndarray:
    return sum(h[i] @ r[i] for i in range(3))


def _diagonal(h: Templates, m: np.ndarray) -> np.ndarray:
    """The data part of the normal matrix's diagonal."""
    return sum(np.asarray(h[i].multiply(h[j]) @ m[i, j]).ravel() for i in range(3) for j in range(3))


@dataclass
class BlockCovariance:
    """Flux covariance of the components: one dense block per tile, zero between tiles."""

    n: int
    blocks: list  # (component indices, their covariance) per tile

    def __post_init__(self):
        self._block = np.full(self.n, -1)
        self._pos = np.zeros(self.n, dtype=np.int64)
        for b, (idx, _) in enumerate(self.blocks):
            self._block[idx] = b
            self._pos[idx] = np.arange(len(idx))

    def diag(self) -> np.ndarray:
        out = np.full(self.n, np.nan)
        for idx, cov in self.blocks:
            out[idx] = np.diag(cov)
        return out

    def sub(self, idx: np.ndarray) -> np.ndarray:
        """Covariance of the components ``idx``, dense (zero between tiles)."""
        idx = np.asarray(idx, dtype=np.int64)
        out = np.zeros((len(idx), len(idx)))
        which = self._block[idx]
        for b in np.unique(which):
            sel = np.flatnonzero(which == b)
            at = self._pos[idx[sel]]
            out[np.ix_(sel, sel)] = self.blocks[b][1][np.ix_(at, at)]
        return out

    def apply(self, x: np.ndarray) -> np.ndarray:
        out = np.zeros(self.n)
        for idx, cov in self.blocks:
            out[idx] = cov @ x[idx]
        return out


def make_tiles(
    centres: np.ndarray, tile_components: int = TILE_COMPONENTS, margin: float = TILE_MARGIN_PX
) -> list[tuple[np.ndarray, np.ndarray]]:
    """(core, fit) component indices per tile, sorted.

    The core is a square cell of components (by centre), the fit set every
    component within ``margin`` px of the cell. One tile holds everything when
    there are at most ``tile_components``; otherwise the largest cell (halving
    from MAX_TILE_PX, down to MIN_TILE_PX) whose fit sets stay within it.
    """
    n = len(centres)
    if n <= tile_components:
        everything = np.arange(n)
        return [(everything, everything)]
    centres = np.asarray(centres, dtype=np.float64)
    cell = MAX_TILE_PX
    while True:
        key = np.floor(centres / cell).astype(np.int64)
        tiles = []
        for k in np.unique(key, axis=0):
            lo, hi = k * cell - margin, (k + 1) * cell + margin
            core = np.flatnonzero((key == k).all(axis=1))
            fit = np.flatnonzero(((centres >= lo) & (centres < hi)).all(axis=1))
            tiles.append((core, fit))
        if cell <= MIN_TILE_PX or max(len(fit) for _, fit in tiles) <= tile_components:
            return tiles
        cell //= 2


def fit_fluxes(h: Templates, weight, data, shifts, prior_mean, prior_sigma, tiles=None, start=None):
    """Fluxes of every component and their covariance per tile.

    The fluxes solve the whole segment's normal equations: with one tile (the
    default) by its inverse, otherwise by conjugate gradients preconditioned
    with the tiles' blocks. A tile's block is the inverse of the normal matrix
    on its fit set, kept on its core. ``start``: fluxes to start the
    conjugate gradients from (default: the prior).
    """
    n = h.shape[0]
    tiles = tiles if tiles is not None else [(np.arange(n), np.arange(n))]
    m, r = _moments(weight, data, shifts)
    precision = 1.0 / prior_sigma**2
    b = _rhs(h, r) + prior_mean * precision
    blocks = []
    for core, fit in tiles:
        cov = np.linalg.inv(_block(h, m, fit) + np.diag(precision[fit]))
        at = np.searchsorted(fit, core)
        blocks.append((core, cov[np.ix_(at, at)]))
    covariance = BlockCovariance(n, blocks)
    if len(tiles) == 1:
        return covariance.apply(b), covariance
    # the departure from the prior: held components (precision 1e6) would otherwise dominate the
    # right-hand side, and with it the convergence test
    start = prior_mean if start is None else start
    delta = _conjugate_gradients(h, m, precision, _rhs(h, r) - _apply(h, m, prior_mean), covariance, start - prior_mean)
    return prior_mean + delta, covariance


def _conjugate_gradients(h: Templates, m, precision, b, preconditioner: BlockCovariance, x0: np.ndarray) -> np.ndarray:
    """Solve the whole normal equations, scaled to a unit diagonal so held and fitted components converge alike."""
    from scipy.sparse.linalg import LinearOperator, cg

    n = len(b)
    scale = 1.0 / np.sqrt(_diagonal(h, m) + precision)
    op = LinearOperator((n, n), matvec=lambda y: scale * (_apply(h, m, scale * y) + precision * scale * y), dtype=np.float64)
    pre = LinearOperator((n, n), matvec=lambda y: preconditioner.apply(y / scale) / scale, dtype=np.float64)
    y, info = cg(op, scale * b, x0=x0 / scale, M=pre, rtol=CG_RTOL, atol=0.0, maxiter=CG_MAXITER)
    if info != 0:
        raise RuntimeError(f"stellar fluxes: conjugate gradients did not converge in {CG_MAXITER} iterations")
    return scale * y


def model_image(h: Templates, fluxes, shifts):
    """Stellar model (nframes, npixels)."""
    amp = h.project(fluxes)
    return amp[0][None, :] + shifts[:, 0:1] * amp[1][None, :] + shifts[:, 1:2] * amp[2][None, :]


def fit_shifts(h, fluxes, weight, data, bright: np.ndarray) -> np.ndarray:
    """Per-frame pointing shift from the residual around the brightest components."""
    gr = h[1][bright].T @ fluxes[bright]
    gc = h[2][bright].T @ fluxes[bright]
    resid = data - (h[0].T @ fluxes)[None, :]
    a11 = (weight * gr**2).sum(axis=1)
    a12 = (weight * gr * gc).sum(axis=1)
    a22 = (weight * gc**2).sum(axis=1)
    b1 = (weight * gr * resid).sum(axis=1)
    b2 = (weight * gc * resid).sum(axis=1)
    det = a11 * a22 - a12**2
    ok = det > 1e-12 * np.maximum(a11 * a22, 1e-300)
    shifts = np.zeros((len(data), 2))
    shifts[ok, 0] = (a22[ok] * b1[ok] - a12[ok] * b2[ok]) / det[ok]
    shifts[ok, 1] = (a11[ok] * b2[ok] - a12[ok] * b1[ok]) / det[ok]
    return shifts


@dataclass
class StarDiagnostics:
    n_clean: np.ndarray  # frames in which the star's core is free of the asteroid
    n_crossing: np.ndarray  # frames in which the asteroid footprint overlaps the core
    chi2: np.ndarray  # reduced chi-square of the core pixels over clean frames
    variability_z: np.ndarray  # (after - before) residual around the crossing, in sigma
    prior_dominated: np.ndarray  # flux error > PRIOR_DOMINATED x prior sigma
    poorly_constrained: np.ndarray  # fitted, but few clean frames or prior-dominated
    variable: np.ndarray  # |variability_z| > VARIABILITY_Z
    fitted: np.ndarray  # core in the fit region; False: held at the Gaia flux
    poor_fit: np.ndarray  # fitted, with core chi2 > CHI2_LIMIT or a flux <= 0
    variability_step: np.ndarray | None = None  # (after - before) change of the star's flux, e-/s
    model_error: np.ndarray | None = None  # |residual| of the star's flux on clean frames: mean and hourly spread, e-/s


@dataclass
class SegmentStars:
    segment: int
    frames: np.ndarray  # absolute frame indices
    fluxes: np.ndarray
    covariance: BlockCovariance
    shifts: np.ndarray  # (nframes, 2): pointing shift (rows, cols) in pixels
    diagnostics: StarDiagnostics
    mean_residual: np.ndarray  # (npixels,) over clean good frames
    model_error_fraction: float = np.nan  # epsilon: the stellar-model error as a fraction of a star's flux
    dense: bool = False  # a dense field: epsilon from the stars that agree with Gaia (DENSE_GAIA_FACTOR)


@dataclass
class StellarModel:
    components: Components
    pixels: np.ndarray  # (npixels, 2): the fit region
    templates: Templates
    segments: list[SegmentStars] = field(default_factory=list)
    near_saturated: np.ndarray | None = None  # (npixels,) excluded from the fit: close to a saturated star

    def evaluate(self, frames: Sequence[int]) -> np.ndarray:
        """Stellar model on the fit region for absolute frame indices (nframes, npixels)."""
        frames = np.asarray(frames)
        out = np.full((len(frames), len(self.pixels)), np.nan)
        for seg in self.segments:
            inside = (frames >= seg.frames[0]) & (frames <= seg.frames[-1])
            if inside.any():
                out[inside] = model_image(self.templates, seg.fluxes, seg.shifts[frames[inside] - seg.frames[0]])
        return out

    def variance(self, frames: Sequence[int]) -> np.ndarray:
        """Model variance on the fit region from the flux covariance (shifts held fixed)."""
        frames = np.asarray(frames)
        out = np.full((len(frames), len(self.pixels)), np.nan)
        for seg in self.segments:
            inside = (frames >= seg.frames[0]) & (frames <= seg.frames[-1])
            for k in np.flatnonzero(inside):
                s = seg.shifts[frames[k] - seg.frames[0]]
                out[k] = 0.0
                for idx, cov in seg.covariance.blocks:  # zero between tiles
                    cols = np.unique(np.concatenate([self.templates[j][idx].indices for j in range(3)]))
                    design = self.templates.block(idx, cols, s)
                    out[k, cols] += np.einsum("ip,ij,jp->p", design, cov, design)
        return out

    def describe(self) -> str:
        lines = [f"{len(self.components)} component(s) on {len(self.pixels)} pixels"]
        for seg in self.segments:
            d = seg.diagnostics
            lines.append(
                f"  segment {seg.segment}: shift rms {np.sqrt(np.mean(seg.shifts**2)):.3f} px; "
                f"fitted {int(d.fitted.sum())}, poorly constrained {int(d.poorly_constrained.sum())}, "
                f"variable {int(d.variable.sum())}, poor fit {int(d.poor_fit.sum())}, "
                f"median chi2 {np.nanmedian(d.chi2[d.fitted]) if d.fitted.any() else float('nan'):.2f}"
            )
        return "\n".join(lines)


def _step_z(series: np.ndarray, clean: np.ndarray, times: np.ndarray, t0: float, t1: float) -> tuple[float, float]:
    """Step in the residual flux across a crossing (e-/s), and its size in units of the same statistic elsewhere.

    The step is mean(after) - mean(before) over VARIABILITY_WINDOW_DAYS on each
    side of the crossing [t0, t1]. Its scale is the MAD of the same step
    measured at other epochs of the clean data (every 0.1 d), so correlated
    residuals such as background trends set the noise level instead of
    inflating the significance.
    """
    w = VARIABILITY_WINDOW_DAYS

    def step(a: float, b: float) -> float:
        before = clean & (times < a) & (times >= a - w)
        after = clean & (times > b) & (times <= b + w)
        if before.sum() < 3 or after.sum() < 3:
            return np.nan
        return float(np.mean(series[after]) - np.mean(series[before]))

    at_crossing = step(t0, t1)
    width = t1 - t0
    null = [step(e, e + width) for e in np.arange(times.min() + w, times.max() - w - width, 0.1)
            if e + width < t0 - w or e > t1 + w]
    null = np.array([v for v in null if np.isfinite(v)])
    if not np.isfinite(at_crossing) or len(null) < 5:
        return np.nan, np.nan
    step = float(at_crossing - np.median(null))
    scale = 1.4826 * np.median(np.abs(null - np.median(null)))
    return step, (step / scale if scale > 0 else np.nan)


def _ensemble_z(z: np.ndarray, tested: np.ndarray) -> np.ndarray:
    """z recalibrated against all tested stars (median and MAD across them), so a drift common to
    the stars is not a detection; unchanged when fewer than MIN_ENSEMBLE stars are tested."""
    if tested.sum() < MIN_ENSEMBLE:
        return z
    zt = z[tested]
    scale = 1.4826 * np.median(np.abs(zt - np.median(zt)))
    if not scale > 0:
        return z
    return np.where(tested, (z - np.median(zt)) / max(scale, 1.0), z)


def _diagnostics(h, fluxes, cov, prior_sigma, weight, data, shifts, asteroid, times, fitted, quiet=None) -> StarDiagnostics:
    """Per-component crossing, chi-square, and variability diagnostics.

    ``quiet`` marks the frames with a normal scattered-light fit; the
    variability step uses only those (the start of an orbit can leave large
    residuals that look like a step).
    """
    quiet = np.ones(len(times), bool) if quiet is None else np.asarray(quiet, bool)
    n = len(fluxes)
    resid = data - model_image(h, fluxes, shifts)
    h0 = h[0]
    n_clean = np.zeros(n, int)
    n_crossing = np.zeros(n, int)
    chi2 = np.full(n, np.nan)
    z = np.full(n, np.nan)
    step = np.full(n, np.nan)
    model_error = np.full(n, np.nan)
    for i in range(n):
        values = h0.data[h0.indptr[i] : h0.indptr[i + 1]]
        in_core = values >= CORE_FRACTION
        if not in_core.any():
            continue
        pix = h0.indices[h0.indptr[i] : h0.indptr[i + 1]][in_core]
        tmpl = values[in_core]
        hit = asteroid[:, pix].any(axis=1)  # frames in which the asteroid touches the core
        good = weight[:, pix].sum(axis=1) > 0
        clean = good & ~hit
        n_clean[i], n_crossing[i] = int(clean.sum()), int(hit.sum())
        w = weight[:, pix][clean]  # the core's pixels first: copying whole frames per star is slow
        r = resid[:, pix][clean]
        if w.size:
            chi2[i] = float((w * r**2).sum() / max(w.size - 1, 1))
        flux_series = (resid[:, pix] * tmpl).sum(axis=1) / (tmpl**2).sum()
        calm = clean & quiet
        if calm.sum() >= MIN_CLEAN_FRAMES:
            bins = np.floor((times[calm] - times[calm].min()) / MODEL_ERROR_BIN_DAYS).astype(int)
            counts = np.bincount(bins)
            means = np.bincount(bins, flux_series[calm])[counts > 0] / counts[counts > 0]
            model_error[i] = float(np.hypot(np.mean(flux_series[calm]), np.std(means) if len(means) > 1 else 0.0))
        if hit.any() and clean.any():
            step[i], z[i] = _step_z(flux_series, clean & quiet, times, times[hit].min(), times[hit].max())
    z = _ensemble_z(z, fitted & np.isfinite(z))
    flux_err = np.sqrt(cov.diag())
    prior_dominated = flux_err > PRIOR_DOMINATED * prior_sigma
    return StarDiagnostics(
        n_clean=n_clean,
        n_crossing=n_crossing,
        chi2=chi2,
        variability_z=z,
        prior_dominated=prior_dominated & fitted,
        poorly_constrained=fitted & ((n_clean < MIN_CLEAN_FRAMES) | prior_dominated),
        variable=fitted & (np.abs(np.nan_to_num(z)) > VARIABILITY_Z),
        fitted=fitted,
        poor_fit=fitted & ((np.nan_to_num(chi2) > CHI2_LIMIT) | (fluxes <= 0)),
        variability_step=step,
        model_error=model_error,
    )


def model_error_fraction(
    diagnostics: StarDiagnostics, fluxes: np.ndarray, prior: np.ndarray | None = None, gaia_factor: float | None = None
) -> float:
    """The segment's stellar-model error as a fraction of a star's flux (epsilon).

    Summed model_error over summed flux of the fitted stars brighter than
    MODEL_ERROR_MIN_FLUX, so the bright stars, whose error is a fraction of
    their flux, dominate over the faint ones, whose error is the background's;
    NaN without such stars. With ``gaia_factor``, only the stars whose flux is
    within that factor of ``prior`` (their Gaia flux) count.
    """
    if diagnostics.model_error is None:
        return np.nan
    use = diagnostics.fitted & (fluxes > MODEL_ERROR_MIN_FLUX) & np.isfinite(diagnostics.model_error)
    if gaia_factor is not None:
        use &= (fluxes < gaia_factor * prior) & (fluxes > prior / gaia_factor)
    return float(diagnostics.model_error[use].sum() / fluxes[use].sum()) if use.any() else np.nan


def fit_segment(
    segment: int,
    frames: np.ndarray,
    times: np.ndarray,
    h: np.ndarray,
    prior_flux: np.ndarray,
    data: np.ndarray,
    weight: np.ndarray,
    asteroid: np.ndarray,
    quiet: np.ndarray | None = None,
    centres: np.ndarray | None = None,
    tile_components: int = TILE_COMPONENTS,
    dense: bool = False,
) -> SegmentStars:
    """Fit one segment. ``data`` and ``weight`` are (nframes, npixels); weight is 0 where excluded.

    ``asteroid`` marks, per frame, the fit-region pixels inside the asteroid
    footprint (their weight must already be 0). ``quiet`` marks the frames
    the variability test may use (default: all). ``centres`` (ncomponents, 2)
    split the covariance into tiles (see :func:`make_tiles`); without them it
    is one tile. ``dense`` marks a dense field (see DENSE_GAIA_FACTOR).
    """
    tiles = make_tiles(centres, tile_components) if centres is not None else None
    fitted = h.touching(threshold=CORE_FRACTION)
    prior_sigma = np.where(fitted, np.maximum(PRIOR_FRACTION * prior_flux, PRIOR_FLOOR), FIXED_SIGMA)
    shifts = np.zeros((len(frames), 2))
    bright = np.argsort(-np.where(fitted, prior_flux, -np.inf))[: min(BRIGHT_FOR_SHIFT, int(fitted.sum()))]
    fluxes, cov = fit_fluxes(h, weight, data, shifts, prior_flux, prior_sigma, tiles)
    for _ in range(N_ITERATIONS):
        shifts = fit_shifts(h, fluxes, weight, data, bright)
        fluxes, cov = fit_fluxes(h, weight, data, shifts, prior_flux, prior_sigma, tiles, start=fluxes)
    diag = _diagnostics(h, fluxes, cov, prior_sigma, weight, data, shifts, asteroid, times, fitted, quiet)
    resid = data - model_image(h, fluxes, shifts)
    wsum = weight.sum(axis=0)
    mean_residual = np.where(wsum > 0, (weight * resid).sum(axis=0) / np.where(wsum > 0, wsum, 1), np.nan)
    eps = model_error_fraction(diag, fluxes, prior_flux, DENSE_GAIA_FACTOR) if dense else model_error_fraction(diag, fluxes)
    return SegmentStars(segment, np.asarray(frames), fluxes, cov, shifts, diag, mean_residual, eps, dense)


def write_stellar(path, model: StellarModel, header: dict) -> None:
    """Save components, the fit region, and per segment fluxes, covariance, shifts, diagnostics."""
    from astropy.io import fits

    from tess_sso.storage.fits_io import write_atomic

    comps = model.components
    member = np.concatenate([np.full(len(r), i) for i, r in enumerate(comps.rows)] or [np.empty(0, int)])
    primary = fits.PrimaryHDU()
    primary.header["NSEG"] = len(model.segments)
    primary.header["NCOMP"] = len(comps)
    for key, value in header.items():
        primary.header[key] = value
    hdus = [
        primary,
        fits.BinTableHDU.from_columns(
            [
                fits.Column(name="COMPONENT", format="J", array=member),
                fits.Column(name="SOURCE_ID", format="K", array=np.concatenate(comps.source_ids or [np.empty(0, int)])),
                fits.Column(name="ROW", format="D", array=np.concatenate(comps.rows or [np.empty(0)])),
                fits.Column(name="COLUMN", format="D", array=np.concatenate(comps.cols or [np.empty(0)])),
                fits.Column(name="RATIO", format="D", array=np.concatenate(comps.ratios or [np.empty(0)])),
                fits.Column(name="PRIOR", format="D", array=comps.prior_flux[member] if len(member) else np.empty(0)),
            ],
            name="STARS",
        ),
        fits.BinTableHDU.from_columns(
            [
                fits.Column(name="ROW", format="J", array=model.pixels[:, 0]),
                fits.Column(name="COLUMN", format="J", array=model.pixels[:, 1]),
                fits.Column(
                    name="NEAR_SATURATED",
                    format="L",
                    array=np.zeros(len(model.pixels), bool) if model.near_saturated is None else model.near_saturated,
                ),
            ],
            name="REGION",
        ),
    ]
    for seg in model.segments:
        s = seg.segment
        flux = fits.ImageHDU(seg.fluxes, name=f"ST{s}_FLUX")
        flux.header["SEGMENT"] = s
        flux.header["FRAME0"] = int(seg.frames[0])
        flux.header["FRAME1"] = int(seg.frames[-1])
        eps = seg.model_error_fraction
        flux.header["EPSILON"] = (eps if np.isfinite(eps) else None, "stellar-model error, fraction of a star's flux")
        flux.header["DENSE"] = (bool(seg.dense), "dense field: EPSILON from stars near Gaia flux")
        d = seg.diagnostics
        hdus += [
            flux,
            # the tiles' blocks, flattened in order; ST{s}_COVIDX names their components
            fits.ImageHDU(np.concatenate([c.ravel() for _, c in seg.covariance.blocks] or [np.empty(0)]), name=f"ST{s}_COV"),
            fits.BinTableHDU.from_columns(
                [
                    fits.Column(name="COMPONENT", format="J",
                                array=np.concatenate([idx for idx, _ in seg.covariance.blocks] or [np.empty(0, int)])),
                    fits.Column(name="BLOCK", format="J",
                                array=np.concatenate([np.full(len(idx), b) for b, (idx, _) in enumerate(seg.covariance.blocks)]
                                                     or [np.empty(0, int)])),
                ],
                name=f"ST{s}_COVIDX",
            ),
            fits.ImageHDU(seg.shifts, name=f"ST{s}_SHIFT"),
            fits.ImageHDU(seg.mean_residual, name=f"ST{s}_RESID"),
            fits.BinTableHDU.from_columns(
                [
                    fits.Column(name="N_CLEAN", format="J", array=d.n_clean),
                    fits.Column(name="N_CROSSING", format="J", array=d.n_crossing),
                    fits.Column(name="CHI2", format="D", array=d.chi2),
                    fits.Column(name="VARIABILITY_Z", format="D", array=d.variability_z),
                    fits.Column(name="VARIABILITY_STEP", format="D",
                                array=d.variability_step if d.variability_step is not None else np.full(len(d.chi2), np.nan)),
                    fits.Column(name="MODEL_ERROR", format="D",
                                array=d.model_error if d.model_error is not None else np.full(len(d.chi2), np.nan)),
                    fits.Column(name="PRIOR_DOMINATED", format="L", array=d.prior_dominated),
                    fits.Column(name="POORLY_CONSTRAINED", format="L", array=d.poorly_constrained),
                    fits.Column(name="VARIABLE", format="L", array=d.variable),
                    fits.Column(name="FITTED", format="L", array=d.fitted),
                    fits.Column(name="POOR_FIT", format="L", array=d.poor_fit),
                ],
                name=f"ST{s}_DIAG",
            ),
        ]
    write_atomic(path, fits.HDUList(hdus))


def _float_or_nan(value) -> float:
    """A header value that may be empty (undefined) or absent (older files)."""
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else np.nan


def _read_covariance(hdul, s: int, n: int) -> BlockCovariance:
    """ST{s}_COV as blocks; a file written before the tiles holds one dense matrix and no ST{s}_COVIDX."""
    values = np.array(hdul[f"ST{s}_COV"].data, dtype=np.float64)
    if f"ST{s}_COVIDX" not in hdul:
        return BlockCovariance(n, [(np.arange(n), values.reshape(n, n))])
    index = hdul[f"ST{s}_COVIDX"].data
    component, block = np.array(index["COMPONENT"]), np.array(index["BLOCK"])
    blocks, offset = [], 0
    for b in range(int(block.max()) + 1 if len(block) else 0):
        idx = component[block == b]
        blocks.append((idx, values[offset : offset + len(idx) ** 2].reshape(len(idx), len(idx))))
        offset += len(idx) ** 2
    return BlockCovariance(n, blocks)


def read_stellar(path, prf) -> StellarModel:
    """Load a saved model; the templates are recomputed with ``prf``."""
    from astropy.io import fits

    with fits.open(path) as hdul:
        stars = hdul["STARS"].data
        n = hdul[0].header["NCOMP"]
        groups = [np.flatnonzero(stars["COMPONENT"] == i) for i in range(n)]
        comps = Components(
            rows=[np.array(stars["ROW"][g]) for g in groups],
            cols=[np.array(stars["COLUMN"][g]) for g in groups],
            ratios=[np.array(stars["RATIO"][g]) for g in groups],
            source_ids=[np.array(stars["SOURCE_ID"][g]) for g in groups],
            prior_flux=np.array([stars["PRIOR"][g][0] for g in groups]),
        )
        region = hdul["REGION"].data
        pixels = np.column_stack([region["ROW"], region["COLUMN"]]).astype(np.int64)
        near_saturated = np.array(region["NEAR_SATURATED"], dtype=bool)
        segments = []
        for s in range(hdul[0].header["NSEG"]):
            flux = hdul[f"ST{s}_FLUX"]
            d = hdul[f"ST{s}_DIAG"].data
            segments.append(
                SegmentStars(
                    segment=flux.header["SEGMENT"],
                    frames=np.arange(flux.header["FRAME0"], flux.header["FRAME1"] + 1),
                    fluxes=np.array(flux.data),
                    covariance=_read_covariance(hdul, s, len(flux.data)),
                    shifts=np.array(hdul[f"ST{s}_SHIFT"].data),
                    diagnostics=StarDiagnostics(
                        n_clean=np.array(d["N_CLEAN"]),
                        n_crossing=np.array(d["N_CROSSING"]),
                        chi2=np.array(d["CHI2"]),
                        variability_z=np.array(d["VARIABILITY_Z"]),
                        prior_dominated=np.array(d["PRIOR_DOMINATED"], dtype=bool),
                        poorly_constrained=np.array(d["POORLY_CONSTRAINED"], dtype=bool),
                        variable=np.array(d["VARIABLE"], dtype=bool),
                        fitted=np.array(d["FITTED"], dtype=bool),
                        poor_fit=np.array(d["POOR_FIT"], dtype=bool),
                        # absent in files written before phase 10
                        variability_step=np.array(d["VARIABILITY_STEP"]) if "VARIABILITY_STEP" in d.columns.names else None,
                        model_error=np.array(d["MODEL_ERROR"]) if "MODEL_ERROR" in d.columns.names else None,
                    ),
                    mean_residual=np.array(hdul[f"ST{s}_RESID"].data),
                    model_error_fraction=_float_or_nan(flux.header.get("EPSILON")),
                    dense=bool(flux.header.get("DENSE", False)),  # absent before dense fields
                )
            )
    return StellarModel(comps, pixels, templates(prf, comps, pixels), segments, near_saturated)
