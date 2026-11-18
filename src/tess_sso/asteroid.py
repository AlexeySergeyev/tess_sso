"""User-facing entry point: one asteroid, its observations, and its pixels."""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
from astropy.table import Table
from astropy.time import Time

from tess_sso.designation import Target, load_target, resolve_designation
from tess_sso.errors import ServiceError
from tess_sso.ephemeris import HorizonsFetch, ephemeris_path, horizons_fetch, load_ephemeris
from tess_sso.gaia import RELEASE as GAIA_RELEASE
from tess_sso.gaia import GaiaFetch, StarCatalog, cones_along_track, gaia_tap_fetch, load_stars
from tess_sso.geometry import (
    BTJD0,
    Track,
    interpolate_geometry,
    make_track,
    sky_to_pixels,
    wcs_offsets,
    with_average_wcs,
)
from tess_sso.local_residual import local_residuals
from tess_sso.masks import (
    exposure_ends,
    ASTEROID_K_SIGMA,
    ASTEROID_THRESHOLD,
    ASTEROID_V_TO_T,
    FLUX_FACTOR,
    GRADIENT_FACTOR,
    K_SIGMA,
    MIN_USABLE_FRACTION,
    DenseFieldWarning,
    Grid,
    ObservationMasks,
    asteroid_footprints,
    asteroid_thresholds,
    build_segment_masks,
    ephemeris_margin_px,
    segment_medians,
    write_masks,
)
from tess_sso.prf import Prf, prf_cache_dir, render_stars
from tess_sso.photometry import (
    APERTURE_THRESHOLD,
    MIN_COVERAGE,
    GroupData,
    GroupResult,
    Photometry,
    aperture_photometry,
    background_variance_parts,
    exposure_weights,
    fit_offset,
    group_images,
    joint_prf_fit,
    prf_weights,
    make_groups,
    member_weights,
    prf_fraction,
    prf_shifts,
    star_variance,
    variability_impact,
    weighted_mean,
    white_excess,
    write_photometry,
)
from tess_sso.output import exposure_table, lightcurve_hdulist
from tess_sso.pixel_product import pixel_hdulist
from tess_sso.quality import SPOC_BAD_BITMASK, PixelMask, SsoQuality, spoc_bad
from tess_sso.scattered_light import (
    CLIP_SIGMA,
    KNOT_SPACING_PX,
    MIN_USABLE_PIXELS,
    N_COMPONENTS,
    PRIOR_SCALE,
    FewBackgroundPixelsWarning,
    ScatteredLightModel,
    fit_segment,
    warn_if_poorly_constrained,
    write_model,
)
from tess_sso.paths import asteroid_dir, data_root, observation_dir, observation_key
from tess_sso.search import SectorsFetch, find_observations, sector_jd_range, tesscut_sectors
from tess_sso.segments import (
    DownlinkMismatchWarning,
    Segment,
    check_segments,
    fetch_orbit_times,
    find_segments,
    load_orbit_times,
    segment_ids,
    spacecraft_time,
    user_segments,
)
from tess_sso.settings import Settings
from tess_sso.timing import BarycentricCorrection, asteroid_timecorr, lkspacecraft_correction
from tess_sso.stellar import (
    PRIOR_FRACTION,
    CORE_FRACTION,
    TILE_COMPONENTS,
    StellarModel,
    Templates,
    fit_region,
    select_components,
    templates,
    write_stellar,
)
from tess_sso.stellar import fit_segment as fit_stellar_segment
from tess_sso.storage.download import DownloadReport, PixelSource, TessCubeSource, download_pixels, ensure_frames
from tess_sso.storage.ecsv_io import read_table
from tess_sso.storage.fits_io import write_atomic
from tess_sso.storage.frames import FrameTable
from tess_sso.storage.plan_cache import plan_pixels
from tess_sso.storage.store import MissingPixelsError, PixelStore

# Per-frame WCS and tesswcs positions further apart than this are reported...
WCS_WARN_PIXELS = 1.0
# ... and this far apart a frame's own WCS is wrong: the photometry would measure the wrong pixels.
# Such frames use the average WCS instead (flagged WCS_FALLBACK), as frames without a WCS do...
WCS_FAIL_PIXELS = 5.0
# ... unless they are more than this fraction of the checked frames: a systematic error, which stops.
WCS_FAIL_FRACTION = 0.05
# Stars are queried this far (pixels) beyond the strip edge, so their light is modelled.
STAR_MARGIN_PX = 7
# Gaia query depth: this many magnitudes fainter than the faintest predicted V, at most G_LIMIT.
G_DEPTH = 5.0
G_LIMIT = 20.0
# Stars brighter than this TESS magnitude saturate; they are masked, not modelled.
SATURATION_TMAG = 6.8
# Background flags: scattered-light gradient or fit residual above this multiple of the
# observation's median (over the frames being measured).
BACKGROUND_FLAG_FACTOR = 3.0


def stellar_design(templates: Templates, stars: np.ndarray, pixels: np.ndarray, shift: np.ndarray) -> np.ndarray:
    """Stellar templates of ``stars`` on region ``pixels`` with a pointing ``shift`` (nstars, npixels)."""
    return templates.block(stars, pixels, shift)


class MissingModelWarning(UserWarning):
    """A segment has no scattered-light model, so its stars cannot be fitted."""


class NoBackgroundModelError(ValueError):
    """No segment of the observation has a scattered-light model: nothing can be measured."""


def require_background(scattered: ScatteredLightModel, key: str) -> None:
    """Stop before the stellar model and the photometry, which would come out empty."""
    if not scattered.segments:
        raise NoBackgroundModelError(
            f"{key}: no segment has a scattered-light model (each has fewer than {MIN_USABLE_PIXELS} usable "
            "background pixels, see FewBackgroundPixelsWarning and masks(...).describe()); nothing to measure"
        )


class WcsDisagreementWarning(UserWarning):
    """Per-frame WCS and the tesswcs model disagree on the asteroid position."""


class WcsDisagreementError(ValueError):
    """Per-frame WCS and the tesswcs model put the asteroid more than WCS_FAIL_PIXELS apart."""


@dataclass
class DownloadPlan:
    """What a download would fetch. Nothing is downloaded to build it."""

    sector: int
    camera: int
    ccd: int
    cutout_pixels: list[tuple[int, int]]
    strip_pixels: list[tuple[int, int]]  # everything that must be stored
    missing: list[tuple[int, int]] = field(default_factory=list)
    nframes: int = 0

    @property
    def missing_bytes(self) -> int:
        """Download size: flux and error, 4 bytes each, all frames."""
        return len(self.missing) * self.nframes * 8

    def describe(self) -> str:
        key = observation_key(self.sector, self.camera, self.ccd)
        if not self.missing:
            return f"{key}: all {len(self.strip_pixels)} pixel(s) are stored; nothing to download"
        return (
            f"{key}: additional download required: {len(self.missing)} of {len(self.strip_pixels)} "
            f"pixel(s), about {self.missing_bytes / 1e9:.2f} GB"
        )


class Asteroid:
    """An asteroid resolved once; everything it downloads or computes goes to its own folder."""

    def __init__(
        self,
        name: str | int,
        root: str | os.PathLike | None = None,
        settings: Settings = Settings(),
        resolve: Callable[[str | int], Target] = resolve_designation,
        sectors: SectorsFetch = tesscut_sectors,
        horizons: HorizonsFetch = horizons_fetch,
        source: Callable[[int, int, int], PixelSource] = TessCubeSource,
        orbit_times: Callable[[], str] = fetch_orbit_times,
        gaia: GaiaFetch = gaia_tap_fetch,
        prf: Callable[[int, int, int], Prf] | None = None,
        barycentric: BarycentricCorrection = lkspacecraft_correction,
    ):
        self.root = data_root(root)
        self.target = load_target(name, self.root, resolve)
        self.settings = settings
        self._sectors, self._horizons, self._source = sectors, horizons, source
        self._sources: dict[tuple[int, int, int], PixelSource] = {}
        self._orbit_times = orbit_times
        self._gaia = gaia
        self._barycentric = barycentric
        self.injection = None  # a tess_sso.injection.Injection added to the pixels as they are read (phase 10c)
        self._prf = prf or (lambda s, c, d: Prf(s, c, d, cache_dir=prf_cache_dir(self.root)))

    def __repr__(self) -> str:
        return f"Asteroid({self.target.fullname!r})"

    def observations(self, refresh: bool = False) -> Table:
        return find_observations(
            self.target.designation,
            root=self.root,
            refresh=refresh,
            step=self.settings.ephemeris_step,
            resolve=lambda _: self.target,
            sectors=self._sectors,
            horizons=self._horizons,
        )

    def source(self, sector: int, camera: int, ccd: int) -> PixelSource:
        key = (sector, camera, ccd)
        if key not in self._sources:
            self._sources[key] = self._source(sector, camera, ccd)
        return self._sources[key]

    @property
    def directory(self) -> Path:
        """This asteroid's folder: delete it to free everything downloaded for it."""
        return asteroid_dir(self.target.dir_name, self.root)

    def observation_dir(self, sector: int, camera: int, ccd: int) -> Path:
        return observation_dir(self.target.dir_name, sector, camera, ccd, self.root)

    def store(self, sector: int, camera: int, ccd: int) -> PixelStore:
        store = PixelStore(self.observation_dir(sector, camera, ccd), sector, camera, ccd)
        if self.injection is not None:
            from tess_sso.injection import InjectedStore

            return InjectedStore(store, self.injection)
        return store

    def _frames(self, sector: int, camera: int, ccd: int) -> FrameTable:
        """The stored frame table; the cube is opened only to write it the first time."""
        store = self.store(sector, camera, ccd)
        if store.has_frames():
            return store.frames()
        return ensure_frames(store, self.source(sector, camera, ccd))

    def ephemeris(self, sector: int):
        start, stop = sector_jd_range(sector)
        return load_ephemeris(self.target, start, stop, self.settings.ephemeris_step, self.root, fetch=self._horizons)

    def track(
        self, sector: int, camera: int, ccd: int, time_range: tuple[float, float] | None = None
    ) -> Track:
        """Position in every frame; fetches the frame table (not the pixels) on first use.

        ``time_range`` (BTJD) keeps all frames but uses only those inside it.

        Frames whose own WCS puts the asteroid more than ``WCS_FAIL_PIXELS`` from
        the tesswcs model use the average WCS and are flagged ``wcs_fallback``;
        more than ``WCS_FAIL_FRACTION`` of the checked frames that far off stops.
        """
        key = observation_key(sector, camera, ccd)
        frames = self._frames(sector, camera, ccd)
        track = make_track(self.ephemeris(sector), frames)
        offsets = wcs_offsets(track, sector, camera, ccd)
        bad = offsets > WCS_FAIL_PIXELS  # NaN (unchecked) compares False
        checked = int(np.isfinite(offsets).sum())
        if bad.sum() > WCS_FAIL_FRACTION * checked:
            raise WcsDisagreementError(
                f"{key}: per-frame WCS and tesswcs positions differ by up to {np.nanmax(offsets):.2f} px; "
                f"{bad.sum()} of {checked} frames more than {WCS_FAIL_PIXELS:g} px apart"
            )
        if bad.any():
            track = with_average_wcs(track, frames.table, bad)
            times = ", ".join(f"{t:.4f}" for t in track.time[bad][:5]) + (", ..." if bad.sum() > 5 else "")
            warnings.warn(
                f"{key}: {bad.sum()} of {checked} frames have a WCS up to {np.nanmax(offsets[bad]):.2f} px "
                f"from tesswcs (BTJD {times}); they use the average WCS and are flagged WCS_FALLBACK",
                WcsDisagreementWarning,
                stacklevel=2,
            )
        worst = np.nanmax(offsets[~bad]) if np.isfinite(offsets[~bad]).any() else np.nan
        if worst > WCS_WARN_PIXELS:
            warnings.warn(
                f"{key}: per-frame WCS and tesswcs positions differ by up to {worst:.2f} px",
                WcsDisagreementWarning,
                stacklevel=2,
            )
        return track.within(*time_range) if time_range else track

    def segments(
        self,
        sector: int,
        camera: int,
        ccd: int,
        bounds: list[tuple[float, float]] | None = None,
    ) -> list[Segment]:
        """Continuous observing segments; ``bounds`` (spacecraft BTJD pairs) overrides them.

        Segments found in the data are cross-checked against the TESS
        orbit-times table; a disagreement warns but never moves a boundary.
        """
        frames = self._frames(sector, camera, ccd)
        t = spacecraft_time(frames)
        if bounds is not None:
            return user_segments(t, bounds)
        segments = find_segments(t)
        key = observation_key(sector, camera, ccd)
        try:
            orbits = load_orbit_times(sector, self.root, self._orbit_times)
        except ServiceError as exc:
            warnings.warn(f"{key}: segments not cross-checked: {exc}", DownlinkMismatchWarning, stacklevel=2)
            return segments
        check = check_segments(segments, orbits)
        if not check.ok:
            warnings.warn(f"{key}: {check.message}", DownlinkMismatchWarning, stacklevel=2)
        return segments

    def plan(
        self, sector: int, camera: int, ccd: int, time_range: tuple[float, float] | None = None
    ) -> DownloadPlan:
        """Return required and missing pixels for this visit.

        The most recent plan is saved beside frames.fits. Unchanged track pixels,
        box sizes and FITS block metadata allow reuse across Python sessions.
        """
        track = self.track(sector, camera, ccd, time_range)
        cutout, strip, missing = plan_pixels(
            track, self.settings.cutout_size, self.settings.strip_width, self.store(sector, camera, ccd)
        )
        return DownloadPlan(
            sector=sector,
            camera=camera,
            ccd=ccd,
            cutout_pixels=cutout,
            strip_pixels=strip,
            missing=missing,
            nframes=len(track),
        )

    def download(
        self, sector: int, camera: int, ccd: int, time_range: tuple[float, float] | None = None
    ) -> DownloadReport:
        """Download the pixels of the current plan that are not stored yet."""
        plan = self.plan(sector, camera, ccd, time_range)
        if not plan.missing:  # everything is on disk: the cube is not opened
            return DownloadReport(already_stored=len(plan.strip_pixels))
        return download_pixels(self.store(sector, camera, ccd), plan.missing, self.source(sector, camera, ccd).fetch)

    def stars(
        self, sector: int, camera: int, ccd: int, time_range: tuple[float, float] | None = None
    ) -> tuple[StarCatalog, np.ndarray, np.ndarray]:
        """Gaia stars around the strip, propagated to the observation epoch, and their pixels."""
        track = self.track(sector, camera, ccd, time_range)
        frames = self.store(sector, camera, ccd).frames()
        on = track.on_science
        cones = cones_along_track(track.ra[on], track.dec[on], self.settings.strip_width / 2 + STAR_MARGIN_PX)
        v = self.ephemeris(sector).v
        g_max = min(G_LIMIT, float(np.nanmax(v)) + G_DEPTH) if np.isfinite(v).any() else G_LIMIT
        epoch = Time(float(np.nanmedian(track.time[on])) + BTJD0, format="jd", scale="tdb").jyear
        catalog = load_stars(self.observation_dir(sector, camera, ccd), cones, g_max, epoch, self._gaia)
        rows, cols = sky_to_pixels(frames.table, catalog.ra, catalog.dec)
        return catalog, rows, cols

    def masks(
        self,
        sector: int,
        camera: int,
        ccd: int,
        time_range: tuple[float, float] | None = None,
        k: float = K_SIGMA,
        save: bool = True,
    ) -> ObservationMasks:
        """Star and asteroid masks per segment. Needs the strip downloaded (see ``plan``).

        ``time_range`` limits the strip (which pixels), not the asteroid mask:
        the models use every frame of a segment, so the footprint covers every
        frame in which the asteroid is on the grid.
        """
        plan = self.plan(sector, camera, ccd, time_range)
        if plan.missing:
            raise MissingPixelsError(plan.missing)
        store = self.store(sector, camera, ccd)
        frames = store.frames()
        track = self.track(sector, camera, ccd)  # full track: see the docstring
        segments = self.segments(sector, camera, ccd)
        grid = Grid.around(plan.strip_pixels)
        strip = grid.image(plan.strip_pixels)
        median, noise = segment_medians(store, grid, strip, segments, spoc_bad(frames.quality))
        prf = self._prf(sector, camera, ccd)
        catalog, rows, cols = self.stars(sector, camera, ccd, time_range)
        modelled = catalog.tmag > SATURATION_TMAG
        star_image = render_stars(prf, rows[modelled], cols[modelled], catalog.flux[modelled], grid.origin, grid.shape)
        eph = self.ephemeris(sector)
        margin = ephemeris_margin_px(float(np.nanmax(np.concatenate([eph.ra_3sigma, eph.dec_3sigma, [0.0]]))))
        t_sc = frames.time + BTJD0 - frames.barycorr
        v_frames = np.interp(t_sc, eph.time_tdb, eph.v, left=np.nan, right=np.nan)
        thresholds = asteroid_thresholds(v_frames, float(np.nanmedian(noise[:, strip])))
        footprints = asteroid_footprints(prf, track, frames.tstop - frames.tstart, grid, thresholds, margin_px=margin)
        per_segment = [
            build_segment_masks(seg.index, strip, median[i], noise[i], star_image, footprints[seg.first : seg.last + 1], k)
            for i, seg in enumerate(segments)
        ]
        key = observation_key(sector, camera, ccd)
        for seg, m in zip(segments, per_segment):
            if m.k > k:
                warnings.warn(
                    f"{key}, segment {seg.index}: Gaia mask threshold raised from k = {k:g} to {m.k:.3g} to leave "
                    f"{MIN_USABLE_FRACTION:.0%} of the strip to the background (dense field); the background model "
                    "takes up the fainter stars' wings",
                    DenseFieldWarning,
                    stacklevel=2,
                )
        header = {
            "KSIGMA": k, **{f"KSIG{seg.index}": m.k for seg, m in zip(segments, per_segment)}, "FLUXFAC": FLUX_FACTOR, "GRADFAC": GRADIENT_FACTOR, "ASTTHR": ASTEROID_THRESHOLD, "ASTKSIG": ASTEROID_K_SIGMA, "ASTV2T": ASTEROID_V_TO_T,
            "ASTMARG": margin, "SATTMAG": SATURATION_TMAG, "NSTARS": len(catalog), "NSATUR": int((~modelled).sum()),
            "GAIAREL": catalog.release, "PRFVER": prf.lkprf_version, "PRFSET": prf.model_set,
        }
        result = ObservationMasks(grid, segments, per_segment, footprints, median, noise, star_image, header)
        if save:
            write_masks(self.results_dir(sector, camera, ccd, create=True) / "masks.fits", result)
        return result

    def scattered_light(
        self,
        sector: int,
        camera: int,
        ccd: int,
        time_range: tuple[float, float] | None = None,
        masks: ObservationMasks | None = None,
        save: bool = True,
    ) -> ScatteredLightModel:
        """Scattered-light model per segment, fitted on the usable pixels of the source masks.

        Pass ``masks`` to reuse ones already computed. Segments with fewer than
        ``MIN_USABLE_PIXELS`` usable pixels get no model and warn; so does a
        photometric cutout that relies on poorly constrained splines.
        """
        masks = masks if masks is not None else self.masks(sector, camera, ccd, time_range, save=save)
        store = self.store(sector, camera, ccd)
        bad = spoc_bad(store.frames().quality)
        origin = np.array(masks.grid.origin)
        strip_pixels = np.argwhere(masks.per_segment[0].strip) + origin
        key = observation_key(sector, camera, ccd)
        fitted = []
        for seg, seg_masks in zip(masks.segments, masks.per_segment):
            usable = np.argwhere(seg_masks.usable) + origin
            if len(usable) < MIN_USABLE_PIXELS:
                warnings.warn(
                    f"{key}, segment {seg.index}: only {len(usable)} usable background pixels; "
                    "no scattered-light model. Widen the strip.",
                    FewBackgroundPixelsWarning,
                    stacklevel=2,
                )
                continue
            flux, flux_err = store.read_pixels([tuple(p) for p in usable], frames=slice(seg.first, seg.last + 1))
            frames = np.arange(seg.first, seg.last + 1)
            fitted.append(fit_segment(seg.index, frames, ~bad[frames], flux, usable, strip_pixels, flux_err=flux_err))
        model = ScatteredLightModel(fitted)
        cutout = np.array(self.plan(sector, camera, ccd, time_range).cutout_pixels)
        warn_if_poorly_constrained(model, cutout, f"{key} photometric cutout")
        if save:
            header = {"NCOMP": N_COMPONENTS, "KNOTPX": KNOT_SPACING_PX, "PRIORSC": PRIOR_SCALE, "CLIPSIG": CLIP_SIGMA}
            write_model(self.results_dir(sector, camera, ccd, create=True) / "scattered_light.fits", model, header)
        return model

    def stellar_model(
        self,
        sector: int,
        camera: int,
        ccd: int,
        time_range: tuple[float, float] | None = None,
        masks: ObservationMasks | None = None,
        scattered: ScatteredLightModel | None = None,
        save: bool = True,
    ) -> StellarModel:
        """Fluxes of the Gaia stars near the photometric cutout, per segment, with pointing shifts.

        The data are the stored pixels minus the scattered-light model;
        pixels inside the asteroid footprint of a frame, near saturated
        stars, or in frames with bad SPOC flags get zero weight.
        """
        masks = masks if masks is not None else self.masks(sector, camera, ccd, time_range, save=save)
        scattered = scattered if scattered is not None else self.scattered_light(sector, camera, ccd, time_range, masks, save)
        require_background(scattered, observation_key(sector, camera, ccd))
        store = self.store(sector, camera, ccd)
        frames_table = store.frames()
        bad = spoc_bad(frames_table.quality)
        grid = masks.grid
        cutout = np.array(self.plan(sector, camera, ccd, time_range).cutout_pixels)
        region = fit_region(cutout, masks.per_segment[0].strip, grid.origin)
        catalog, rows, cols = self.stars(sector, camera, ccd, time_range)
        comps, near_saturated = select_components(
            catalog.source_id, rows, cols, catalog.tmag, catalog.flux, region, grid.origin
        )
        cells = np.argwhere(region)
        pixels = cells + np.array(grid.origin)
        prf = self._prf(sector, camera, ccd)
        h = templates(prf, comps, pixels)
        excluded_pixels = near_saturated[cells[:, 0], cells[:, 1]]
        lookup = np.full(grid.shape[0] * grid.shape[1], -1)
        lookup[np.ravel_multi_index((cells[:, 0], cells[:, 1]), grid.shape)] = np.arange(len(cells))
        model = StellarModel(comps, pixels, h, near_saturated=excluded_pixels)
        centres = np.array([comps.centre(i) for i in range(len(comps))]).reshape(-1, 2)
        key = observation_key(sector, camera, ccd)
        for seg, seg_masks in zip(masks.segments, masks.per_segment):
            try:
                background = scattered.segment_of(seg.first)
            except IndexError:
                warnings.warn(f"{key}, segment {seg.index}: no scattered-light model; stars not fitted",
                              MissingModelWarning, stacklevel=2)
                continue
            frames = np.arange(seg.first, seg.last + 1)
            flux, err = store.read_pixels([tuple(p) for p in pixels], frames=slice(seg.first, seg.last + 1))
            data = flux - background.evaluate(pixels, frames - background.frames[0])
            asteroid = np.zeros(data.shape, dtype=bool)
            for i, frame in enumerate(frames):
                idx = lookup[masks.footprints[frame]]
                asteroid[i, idx[idx >= 0]] = True
            with np.errstate(divide="ignore", invalid="ignore"):
                weight = np.where(np.isfinite(data) & np.isfinite(err) & (err > 0), 1.0 / err**2, 0.0)
            weight[asteroid] = 0.0
            weight[:, excluded_pixels] = 0.0
            # where the background splines have no data the model can be anything (-8e4 e-/s in the
            # Galactic plane, 35534 sector 39), and stars there would be fitted to it
            weight[:, ~background.constrained(pixels)] = 0.0
            weight[bad[frames]] = 0.0
            data = np.where(weight > 0, data, 0.0)
            rms = background.frame_rms[frames - background.frames[0]]
            quiet = rms <= BACKGROUND_FLAG_FACTOR * np.nanmedian(rms)  # not the start of an orbit
            model.segments.append(
                fit_stellar_segment(seg.index, frames, frames_table.time[frames], h, comps.prior_flux, data, weight,
                                    asteroid, quiet, centres=centres, tile_components=TILE_COMPONENTS,
                                    dense=seg_masks.k > masks.header.get("KSIGMA", K_SIGMA))
            )
        if save:
            header = {"NPIX": len(pixels), "TILECOMP": (TILE_COMPONENTS, "components above which the covariance is tiled"),
                      "PRIORFR": PRIOR_FRACTION, "PRFVER": prf.lkprf_version, "PRFSET": prf.model_set,
                      "GAIAREL": catalog.release}
            write_stellar(self.results_dir(sector, camera, ccd, create=True) / "stellar.fits", model, header)
        return model

    def photometry(
        self,
        sector: int,
        camera: int,
        ccd: int,
        time_range: tuple[float, float] | None = None,
        masks: ObservationMasks | None = None,
        scattered: ScatteredLightModel | None = None,
        stellar: StellarModel | None = None,
        save: bool = True,
    ) -> Photometry:
        """Asteroid photometry per group of exposures (see Settings.group_minutes / group_count).

        Each exposure's box (``cutout_size``) is centred on the pixel holding
        the asteroid; scattered light and stars are subtracted. The primary
        flux is a joint PRF fit per group, the alternative an aperture on the
        group's stack. Exposures with bad SPOC flags, off the science area, or
        in a segment without models are left out.
        """
        masks = masks if masks is not None else self.masks(sector, camera, ccd, time_range, save=save)
        scattered = scattered if scattered is not None else self.scattered_light(sector, camera, ccd, time_range, masks, save)
        require_background(scattered, observation_key(sector, camera, ccd))
        stellar = stellar if stellar is not None else self.stellar_model(sector, camera, ccd, time_range, masks, scattered, save)
        store = self.store(sector, camera, ccd)
        frames = store.frames()
        track = self.track(sector, camera, ccd, time_range)
        prf = self._prf(sector, camera, ccd)
        size, half = self.settings.cutout_size, self.settings.cutout_size // 2
        seg_of = segment_ids(masks.segments, len(frames))
        sl_by_seg = {s.segment: s for s in scattered.segments}
        st_by_seg = {s.segment: s for s in stellar.segments}
        has_models = np.array([sid in sl_by_seg and sid in st_by_seg for sid in seg_of])
        accepted = track.on_science & ~spoc_bad(frames.quality) & np.isfinite(track.row) & has_models
        cadence = float(np.median(np.diff(frames.time)))
        groups = make_groups(
            frames.time, seg_of, accepted, cadence,
            duration_days=None if self.settings.group_minutes is None else self.settings.group_minutes / 1440.0,
            count=self.settings.group_count,
        )
        phot = Photometry(tstart=frames.tstart, tstop=frames.tstop)
        if not groups:
            return phot
        used = np.concatenate([g.frames for g in groups])
        first, last = int(used.min()), int(used.max())
        cutout = [tuple(p) for p in self.plan(sector, camera, ccd, time_range).cutout_pixels]
        column = {p: i for i, p in enumerate(cutout)}
        flux_all, err_all = store.read_pixels(cutout, frames=slice(first, last + 1))
        region = {(int(r), int(c)): i for i, (r, c) in enumerate(stellar.pixels)}
        grid = masks.grid
        grid_to_region = np.full(grid.shape[0] * grid.shape[1], -1)
        cells = stellar.pixels - np.array(grid.origin)
        grid_to_region[np.ravel_multi_index((cells[:, 0], cells[:, 1]), grid.shape)] = np.arange(len(cells))
        r0s, c0s, r1s, c1s = exposure_ends(track, frames.tstop - frames.tstart)
        reach = self.settings.strip_width // 2  # window for the usable background-pixel fraction
        # scattered-light model variance on the cutout, frame-independent parts per segment
        variance_terms = {s: m.variance_terms(np.array(cutout)) for s, m in sl_by_seg.items()}

        # per-exposure boxes and flags
        box = {}
        for f in used:
            seg_sl, seg_st = sl_by_seg[seg_of[f]], st_by_seg[seg_of[f]]
            wr, wc = int(np.floor(track.row[f] + 0.5)) - half, int(np.floor(track.col[f] + 0.5)) - half
            pix = np.array([(wr + i, wc + j) for i in range(size) for j in range(size)])
            cols = np.array([column.get((int(r), int(c)), -1) for r, c in pix])
            ridx = np.array([region.get((int(r), int(c)), -1) for r, c in pix])
            ok = (cols >= 0) & (ridx >= 0)
            flux = np.where(ok, flux_all[f - first, np.maximum(cols, 0)], np.nan)
            err = np.where(ok, err_all[f - first, np.maximum(cols, 0)], np.nan)
            background = seg_sl.evaluate(pix, np.array([f - seg_sl.frames[0]]))[0]
            per_component, fitted_sq = variance_terms[seg_of[f]]
            rel_f = f - seg_sl.frames[0]
            safe = np.maximum(cols, 0)
            bkg_var = seg_sl.temporal[rel_f] ** 2 @ per_component[:, safe] + seg_sl.frame_rms[rel_f] ** 2 * fitted_sq[safe]
            bkg_sigma = np.where(cols >= 0, np.sqrt(np.maximum(bkg_var, 0.0)), np.nan)
            shift = seg_st.shifts[f - seg_st.frames[0]]
            region_idx = np.maximum(ridx, 0)
            # only the pixels of this box, and only the stars that reach them (the region holds thousands)
            active = np.flatnonzero(stellar.templates.touching(region_idx[ok]))
            design = np.where(ok, stellar_design(stellar.templates, active, region_idx, shift), 0.0)
            stars = seg_st.fluxes[active] @ design
            static = np.nan_to_num(np.where(ok, seg_st.mean_residual[np.maximum(ridx, 0)], 0.0))
            data = flux - background - stars
            if self.settings.static_correction:
                data = data - static
            with np.errstate(divide="ignore", invalid="ignore"):
                weight = np.where(ok & np.isfinite(data) & np.isfinite(err) & (err > 0), 1.0 / err**2, 0.0)
            ends = (r0s[f], c0s[f], r1s[f], c1s[f])
            if not np.isfinite(ends).all():
                ends = (track.row[f], track.col[f], track.row[f], track.col[f])
            model_prf = prf.smeared(*ends, (wr, wc), (size, size))
            h = 0.05  # px, for the PRF derivatives used by the offset fit
            dr_prf = (prf.smeared(ends[0] + h, ends[1], ends[2] + h, ends[3], (wr, wc), (size, size))
                      - prf.smeared(ends[0] - h, ends[1], ends[2] - h, ends[3], (wr, wc), (size, size))) / (2 * h)
            dc_prf = (prf.smeared(ends[0], ends[1] + h, ends[2], ends[3] + h, (wr, wc), (size, size))
                      - prf.smeared(ends[0], ends[1] - h, ends[2], ends[3] - h, (wr, wc), (size, size))) / (2 * h)
            flags = 0
            if track.wcs_fallback[f]:
                flags |= SsoQuality.WCS_FALLBACK
            if not seg_sl.constrained(pix[ok]).all():
                flags |= SsoQuality.FEW_BACKGROUND_PIXELS
            if stellar.near_saturated is not None and stellar.near_saturated[ridx[ok]].any():
                flags |= SsoQuality.SATURATED_STAR
            cells = pix - np.array(grid.origin)
            on_grid = np.all((cells >= 0) & (cells < np.array(grid.shape)), axis=1)
            flat = np.where(on_grid, np.ravel_multi_index(tuple(np.clip(cells, 0, np.array(grid.shape) - 1).T), grid.shape), -1)
            footprint = grid_to_region[masks.footprints[f]]
            footprint = footprint[footprint >= 0]
            seg_masks = masks.per_segment[[s.index for s in masks.segments].index(seg_of[f])]
            # strip pixels within strip_width / 2 of the asteroid, and the share the background fit used
            lo = np.array([wr + half - reach, wc + half - reach]) - np.array(grid.origin)
            start = np.maximum(lo, 0)
            window = np.argwhere(seg_masks.strip[start[0]:lo[0] + 2 * reach + 1, start[1]:lo[1] + 2 * reach + 1])
            usable_fraction = seg_sl.usable_fraction(window + start + np.array(grid.origin))
            missing_from_gaia = seg_masks.star_data & ~seg_masks.star_gaia
            if missing_from_gaia.flat[masks.footprints[f]].any():
                flags |= SsoQuality.ONE_STAR_MASK_ONLY
            variable_under = np.zeros(len(stellar.components), bool)
            if len(footprint):
                # stars whose core lies under the asteroid in this exposure
                under = stellar.templates.touching(footprint, CORE_FRACTION)
                variable_under = under & seg_st.diagnostics.variable  # flagged per group, by their effect
                if (under & seg_st.diagnostics.poorly_constrained).any():
                    flags |= SsoQuality.STELLAR_MODEL_UNCERTAIN
            box[f] = dict(
                origin=(wr, wc), ends=ends, pix=pix, data=np.where(weight > 0, data, 0.0), weight=weight, static=static,
                sigma=np.where(weight > 0, err, 0.0), prf=model_prf, prf_dr=dr_prf, prf_dc=dc_prf, flags=int(flags),
                active=active, region_idx=region_idx, ok=ok, shift=shift,
                background=np.where(ok, background, 0.0), stars=np.where(ok, stars, 0.0), bkg_sigma=bkg_sigma,
                star_mask=on_grid & seg_masks.star.ravel()[np.maximum(flat, 0)], variable_under=variable_under,
                asteroid_mask=on_grid & np.isin(flat, masks.footprints[f]),
                gradient=float(seg_sl.gradient(track.row[f], track.col[f], np.array([f - seg_sl.frames[0]]))[0]),
                rms=float(seg_sl.frame_rms[f - seg_sl.frames[0]]),
                mad=float(seg_sl.frame_mad[f - seg_sl.frames[0]]) if seg_sl.frame_mad is not None else np.nan,
                chi2=float(seg_sl.frame_chi2[f - seg_sl.frames[0]]) if seg_sl.frame_chi2 is not None else np.nan,
                usable=usable_fraction,
            )
        if self.settings.static_correction and self.settings.residual_model in ("local", "local_pointing"):
            rms = np.full(len(frames), np.inf)
            for s in sl_by_seg.values():
                rms[s.frames] = s.frame_rms / np.nanmedian(s.frame_rms)
            local_residuals(box, frames, first, last, flux_all, err_all, column, region, stellar, sl_by_seg,
                            st_by_seg, seg_of, masks, track, ~spoc_bad(frames.quality) & (rms <= BACKGROUND_FLAG_FACTOR),
                            pointing=self.settings.residual_model == "local_pointing")
        grad_limit = BACKGROUND_FLAG_FACTOR * np.nanmedian([b["gradient"] for b in box.values()])
        rms_limit = BACKGROUND_FLAG_FACTOR * np.nanmedian([b["rms"] for b in box.values()])
        for b in box.values():
            if b["gradient"] > grad_limit:
                b["flags"] |= SsoQuality.BACKGROUND_GRADIENT
            if b["rms"] > rms_limit:
                b["flags"] |= SsoQuality.BACKGROUND_POOR_FIT

        # first pass: each group's offset from the predicted position, with the PRF there
        group_data, measured = {}, {}
        for g in groups:
            members = [box[f] for f in g.frames]
            shape = (len(members), size, size)
            group_data[g.index] = GroupData(
                frames=g.frames,
                origins=np.array([m["origin"] for m in members]),
                data=np.stack([m["data"] for m in members]).reshape(shape),
                weight=np.stack([m["weight"] for m in members]).reshape(shape),
                sigma=np.stack([m["sigma"] for m in members]).reshape(shape),
                prf=np.stack([m["prf"] for m in members]),
                prf_dr=np.stack([m["prf_dr"] for m in members]),
                prf_dc=np.stack([m["prf_dc"] for m in members]),
            )
            measured[g.index] = fit_offset(group_data[g.index])
        shifts, phot.segment_offsets = prf_shifts(
            self.settings.prf_position,
            np.array([g.segment for g in groups]),
            np.array([measured[g.index][:2] for g in groups]),
            np.array([measured[g.index][2:] for g in groups]),
        )

        singles = []  # (segment, time, flux, random error) of each exposure, for the white excess
        excess_weights = []  # per group: how a per-exposure excess enters the PRF and aperture fluxes
        for g, shift in zip(groups, shifts):
            members = [box[f] for f in g.frames]
            shape = (len(members), size, size)
            gd = group_data[g.index]
            if shift.any():
                # the smeared PRF recomputed with the track ends moved by the chosen offset
                gd.prf = np.stack([
                    prf.smeared(m["ends"][0] + shift[0], m["ends"][1] + shift[1], m["ends"][2] + shift[0],
                                m["ends"][3] + shift[1], m["origin"], (size, size))
                    for m in members
                ])
            flux, err_pix, a, chi2 = joint_prf_fit(gd, self.settings.local_background)
            under = prf_weights(gd)
            ap = aperture_photometry(gd)
            seg_sl, seg_st = sl_by_seg[g.segment], st_by_seg[g.segment]
            # the stellar templates of the stars that reach any member box, per member (nexp, nstars, size, size)
            union = np.unique(np.concatenate([m["active"] for m in members]))
            design = np.stack([
                np.where(m["ok"], stellar_design(stellar.templates, union, m["region_idx"], m["shift"]), 0.0)
                for m in members
            ]).reshape(len(members), len(union), size, size)
            covariance = seg_st.covariance.sub(union)
            pixels = np.stack([m["pix"] for m in members])
            rel = g.frames - seg_sl.frames[0]
            err_star = np.sqrt(star_variance(a, design, covariance))
            bkg_spatial, bkg_temporal = background_variance_parts(a, pixels, rel, seg_sl)
            ap_err_star = np.sqrt(star_variance(ap.a, design, covariance)) if ap.fraction > 0 else np.nan
            ap_spatial, ap_temporal = background_variance_parts(ap.a, pixels, rel, seg_sl) if ap.fraction > 0 else (np.nan, np.nan)
            weights = member_weights(gd)
            for t, f in enumerate(g.frames):  # each exposure alone: its flux and random error
                wp = gd.weight[t] * gd.prf[t]
                denom = float((wp * gd.prf[t]).sum())
                if denom > 0:
                    _, temporal_t = background_variance_parts((wp / denom)[None], pixels[t : t + 1], rel[t : t + 1], seg_sl)
                    singles.append((g.segment, frames.time[f], float((wp * gd.data[t]).sum()) / denom,
                                    np.sqrt(1.0 / denom + temporal_t)))
            ap_share = exposure_weights(gd)
            ap_share = ap_share / ap_share.sum() if ap_share.sum() > 0 else np.full(len(ap_share), 1.0 / len(ap_share))
            excess_weights.append((float(np.sqrt(np.sum(weights**2))), float(np.sqrt(np.sum(ap_share**2)))))
            static = np.stack([m["static"] for m in members]).reshape(shape)
            star_residual = float((a * static).sum())
            background = np.stack([m["background"] for m in members]).reshape(shape)
            stars = np.stack([m["stars"] for m in members]).reshape(shape)
            dr, dc, dr_err, dc_err = measured[g.index]  # from the predicted position, whatever the PRF shift
            removed_stars = stars + static if self.settings.static_correction else stars
            images = group_images(gd, flux, background, np.stack([m["bkg_sigma"] for m in members]).reshape(shape),
                                  removed_stars)
            pixel_mask = np.zeros((size, size), np.int16)
            pixel_mask[np.stack([m["star_mask"] for m in members]).reshape(shape).any(axis=0)] |= PixelMask.STAR
            pixel_mask[np.stack([m["asteroid_mask"] for m in members]).reshape(shape).any(axis=0)] |= PixelMask.ASTEROID
            pixel_mask[images["COVERAGE"] > 0] |= PixelMask.USED
            pixel_mask[ap.aperture] |= PixelMask.APERTURE
            images["PIXEL_MASK"] = pixel_mask
            flags = 0
            for m in members:
                flags |= m["flags"]
            # without the static correction, an uncorrected residual larger than the noise is flagged
            residual_left = 0.0 if self.settings.static_correction else abs(star_residual)
            if err_star > err_pix or residual_left > err_pix:
                flags |= SsoQuality.STELLAR_MODEL_UNCERTAIN
            member_flags = np.array([m["flags"] for m in members])
            # a variable star under the asteroid, if its step could move the flux by more than the pixel error
            variable_under = np.stack([m["variable_under"] for m in members])[:, union]  # (nexp, nstars)
            if variable_under.any() and seg_st.diagnostics.variability_step is not None:
                impact = variability_impact(a, design, seg_st.diagnostics.variability_step[union])
                if (impact[variable_under.any(axis=0)] > err_pix).any():
                    flags |= SsoQuality.STELLAR_VARIABILITY
                    member_flags[variable_under.any(axis=1)] |= SsoQuality.STELLAR_VARIABILITY
            if g.coverage < MIN_COVERAGE:
                flags |= SsoQuality.LOW_GROUP_COVERAGE
            if g.incomplete:
                flags |= SsoQuality.INCOMPLETE_GROUP
            if ap.masked_fraction > 0:
                flags |= SsoQuality.MASKED_APERTURE_PIXELS
            spoc = 0
            for f in g.frames:
                spoc |= int(frames.quality[f])
            phot.groups.append(GroupResult(
                group=g, time=float(np.sum(weights * frames.time[g.frames])), weights=weights,
                flux=flux, err_pix=err_pix, err_star=float(err_star), err_bkg=float(np.sqrt(bkg_temporal)), red_chi2=chi2,
                offset=(dr, dc), aperture=ap, ap_err_star=float(ap_err_star), ap_err_bkg=float(np.sqrt(ap_temporal)),
                err_bkg_sys=float(np.sqrt(bkg_spatial)), ap_err_bkg_sys=float(np.sqrt(ap_spatial)),
                star_residual=star_residual, sso_quality=int(flags), spoc_quality=spoc,
                offset_err=(dr_err, dc_err), origins=gd.origins, member_flags=member_flags,
                bkg_flux=float((under * background).sum()), star_flux=float((under * stars).sum()),
                ap_bkg_flux=float((ap.a * background).sum()) if ap.fraction > 0 else np.nan,
                ap_star_flux=float((ap.a * stars).sum()) if ap.fraction > 0 else np.nan,
                ap_star_residual=float((ap.a * static).sum()) if ap.fraction > 0 else np.nan,
                bkg_std=float(np.sum(weights * [m["rms"] for m in members])),
                bkg_gradient=float(np.sum(weights * [m["gradient"] for m in members])),
                bkg_mad=weighted_mean([m["mad"] for m in members], weights),
                bkg_chi2=weighted_mean([m["chi2"] for m in members], weights),
                bkg_usable=weighted_mean([m["usable"] for m in members], weights),
                prf_fraction=prf_fraction(gd, weights), images=images, prf_shift=(float(shift[0]), float(shift[1])),
                ap_npix=int((ap.aperture & (ap.coverage > 0)).sum()),
            ))
        # the white excess of each segment's exposures, carried into the groups with their weights
        cadence = float(np.median(np.diff(frames.time)))
        for seg_index in sorted({s for s, *_ in singles}):
            rows = np.array([row[1:] for row in singles if row[0] == seg_index])
            order = np.argsort(rows[:, 0])
            excess = white_excess(rows[order, 0], rows[order, 1], rows[order, 2], cadence)
            phot.excess[seg_index] = float(excess) if np.isfinite(excess) else 0.0
        for r, (prf_share, ap_share) in zip(phot.groups, excess_weights):
            x = phot.excess.get(r.group.segment, 0.0)
            r.err_excess, r.ap_err_excess = x * prf_share, x * ap_share
            # a bright star under the asteroid: its (slow, nearly coherent) model error exceeds the point's total error
            eps = st_by_seg[r.group.segment].model_error_fraction
            phot.model_error[r.group.segment] = eps
            if np.isfinite(eps) and eps * abs(r.star_flux) > np.hypot(r.flux_err, r.flux_err_sys):
                r.sso_quality |= int(SsoQuality.STELLAR_MODEL_UNCERTAIN)
                r.member_flags = r.member_flags | int(SsoQuality.STELLAR_MODEL_UNCERTAIN)
        if save:
            header = {"APTHR": APERTURE_THRESHOLD, "MINCOV": MIN_COVERAGE, "BKGFLAG": BACKGROUND_FLAG_FACTOR,
                      "STATCORR": self.settings.static_correction, "PRFPOS": self.settings.prf_position,
                      "RESMODEL": self.settings.residual_model, "LOCALBKG": self.settings.local_background,
                      "PRFVER": prf.lkprf_version, "PRFSET": prf.model_set}
            write_photometry(self.results_dir(sector, camera, ccd, create=True) / "photometry.ecsv", phot, header)
        return phot

    def lightcurve(
        self,
        sector: int,
        camera: int,
        ccd: int,
        time_range: tuple[float, float] | None = None,
        masks: ObservationMasks | None = None,
        scattered: ScatteredLightModel | None = None,
        stellar: StellarModel | None = None,
        photometry: Photometry | None = None,
        save: bool = True,
    ):
        """The light-curve file (an ``HDUList``), written as ``tess-<name>-s<sector>-<camera>-<ccd>-lc.fits``.

        Layout compatible with the reference pipeline, see :mod:`tess_sso.output`.
        The barycentric correction toward the asteroid needs the TESS SPICE
        kernels (lkspacecraft); without them the SPOC correction is used,
        with a warning and ``TIMECORR = 'SPOC'`` in the header.
        """
        photometry, frames, exposures, segments, provenance = self._output_inputs(
            sector, camera, ccd, time_range, masks, scattered, stellar, photometry, save
        )
        hdul = lightcurve_hdulist(
            photometry, frames, exposures, segments, self.target.designation, self.target.fullname, provenance,
            aperture_header={"AP_TYPE": ("prf", "aperture method"), "AP_PAR": (APERTURE_THRESHOLD, "threshold")},
        )
        if save:
            write_atomic(self.lightcurve_path(sector, camera, ccd, create=True), hdul)
        return hdul

    def pixel_product(
        self,
        sector: int,
        camera: int,
        ccd: int,
        time_range: tuple[float, float] | None = None,
        masks: ObservationMasks | None = None,
        scattered: ScatteredLightModel | None = None,
        stellar: StellarModel | None = None,
        photometry: Photometry | None = None,
        save: bool = True,
    ):
        """The group stacks in the SPOC target-pixel layout (an ``HDUList``), for inspection.

        Written as ``tess-<name>-s<sector>-<camera>-<ccd>-tpf.fits``, one row per
        group like the light-curve file; see :mod:`tess_sso.pixel_product`.
        """
        photometry, frames, exposures, segments, provenance = self._output_inputs(
            sector, camera, ccd, time_range, masks, scattered, stellar, photometry, save
        )
        hdul = pixel_hdulist(photometry, frames, exposures, self.target.designation, self.target.fullname, provenance)
        if save:
            write_atomic(self.pixel_product_path(sector, camera, ccd, create=True), hdul)
        return hdul

    def _output_inputs(self, sector, camera, ccd, time_range, masks, scattered, stellar, photometry, save):
        """Photometry (computed if not given), frames, the per-exposure table, segments, and provenance."""
        if photometry is None:
            masks = masks if masks is not None else self.masks(sector, camera, ccd, time_range, save=save)
            photometry = self.photometry(sector, camera, ccd, time_range, masks, scattered, stellar, save)
        segments = masks.segments if masks is not None else self.segments(sector, camera, ccd)
        frames = self.store(sector, camera, ccd).frames()
        track = self.track(sector, camera, ccd, time_range)
        eph = self.ephemeris(sector)
        timecorr, source = asteroid_timecorr(
            frames.time - frames.barycorr, track.ra, track.dec, frames.barycorr, self._barycentric
        )
        exposures = exposure_table(frames, track, interpolate_geometry(eph, frames), timecorr)
        prf = self._prf(sector, camera, ccd)
        start, stop = sector_jd_range(sector)
        eph_file = ephemeris_path(self.target, start, stop, self.settings.ephemeris_step, self.root)
        eph_date = str(read_table(eph_file).meta.get("QUERYDAT", "")) if eph_file.exists() else ""
        s = self.settings
        provenance = {
            "TCORRSRC": source,
            "SETHASH": (s.hash(), "settings hash, names the results folder"),
            "CUTSIZE": (s.cutout_size, "[pix] photometric cutout side"),
            "STRIPW": (s.strip_width, "[pix] background strip width"),
            "EPHSTEP": (s.ephemeris_step, "Horizons step"),
            "GRPMIN": (s.group_minutes, "[min] group duration"),
            "GRPCNT": (s.group_count, "exposures per group"),
            "STATCORR": (s.static_correction, "static stellar residual subtracted"),
            "RESMODEL": (s.residual_model, "residual: local, local_pointing, static"),
            "LOCALBKG": (s.local_background, "free local background in the PRF fit"),
            "PRFPOS": (s.prf_position, "PRF position: segment, group, or predicted"),
            "EPHSRC": ("JPL Horizons @-95", "ephemeris source"),
            "EPHDATE": (eph_date[:10], "ephemeris query date"),
            "GAIAREL": (GAIA_RELEASE, "Gaia release of the stellar model"),
            "PRFVER": (prf.lkprf_version, "lkprf version"),
            "PRFSET": (prf.model_set, "PRF model set"),
            "BAD_SPOC": (SPOC_BAD_BITMASK, "SPOC bits excluded from all models"),
            "APTHR": (APERTURE_THRESHOLD, "aperture: stack-PRF threshold"),
            "MINCOV": (MIN_COVERAGE, "group coverage below this is flagged"),
            "BKGFLAG": (BACKGROUND_FLAG_FACTOR, "background flags: factor over median"),
        }
        if self.injection is not None:
            provenance["INJECTED"] = (self.injection.label or "yes", "test signal injected into the pixels")
        return photometry, frames, exposures, segments, provenance

    def lightcurve_path(self, sector: int, camera: int, ccd: int, create: bool = False) -> Path:
        name = f"tess-{self.target.dir_name}-s{sector:04d}-{camera}-{ccd}-lc.fits"
        return self.results_dir(sector, camera, ccd, create=create) / name

    def pixel_product_path(self, sector: int, camera: int, ccd: int, create: bool = False) -> Path:
        name = f"tess-{self.target.dir_name}-s{sector:04d}-{camera}-{ccd}-tpf.fits"
        return self.results_dir(sector, camera, ccd, create=create) / name

    def results_dir(self, sector: int, camera: int, ccd: int, create: bool = False) -> Path:
        """``<asteroid>/<observation>/<settings hash>/`` (``.../injection/`` with an injection); with ``create``,
        also writes settings.json."""
        path = self.observation_dir(sector, camera, ccd) / self.settings.hash()
        if self.injection is not None:  # never mixed with the real results
            path = path / "injection"
        if create:
            path.mkdir(parents=True, exist_ok=True)
            (path / "settings.json").write_text(json.dumps(self.settings.to_dict(), indent=2, sort_keys=True) + "\n")
        return path
