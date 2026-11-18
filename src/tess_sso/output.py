"""Light-curve FITS compatible with the reference pipeline (1.5.1), and lightkurve access.

HDUs, in order:

- ``PRIMARY``: provenance, settings and their hash, averaged geometry.
- ``LIGHTCURVE_PRF``: the joint PRF fit, one row per group (HDU 1, what
  lightkurve and binaries read).
- ``LIGHTCURVE_AP``: aperture photometry on the stack, the same rows.
- ``EXTRAS``: time, position, and geometry, the same rows again, so every
  table is row-aligned and also matchable by ``CADENCENO``.
- ``GROUPS``: one row per member exposure with its weight; the group rows
  of the other tables are these weighted means.
- ``SEGMENTS``: the observing segments.

A group's ``CADENCENO`` is its member exposure closest to the group time.
``TIME`` is barycentric toward the asteroid; ``TIME_EMIT`` is the emission
time at the asteroid. Every weighted group quantity uses the same member
weights (the joint PRF fit's), so ``TIME_EMIT`` is the exact weighted mean
of the members' emission times.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Sequence

import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time

from tess_sso.geometry import BTJD0, FrameGeometry, Track, pixel_offset_to_sky
from tess_sso.photometry import Photometry
from tess_sso.segments import Segment
from tess_sso.storage.frames import FrameTable
from tess_sso.timing import emission_time

TESSMAG_ZERO_POINT = 20.44  # reference-pipeline zero point, mag
TESSMAG_ZERO_POINT_ERR = 0.05
FILE_VERSION = "1.0"

# Written into every table HDU: lightkurve reads the time scale from the HDU it opens.
TIME_KEYWORDS = (
    ("TIMEREF", "SOLARSYSTEM", "barycentric correction applied to times"),
    ("TASSIGN", "SPACECRAFT", "where time is assigned"),
    ("TIMESYS", "TDB", "time system is Barycentric Dynamical Time (TDB)"),
    ("BJDREFI", 2457000, "integer part of BTJD reference date"),
    ("BJDREFF", 0.0, "fraction of the day in BTJD reference date"),
    ("TIMEUNIT", "d", "time unit for TIME, TSTART and TSTOP"),
)

BTJD_UNIT = "BJD - 2457000, days"


@dataclass(frozen=True)
class Column:
    name: str
    format: str
    unit: str | None = None
    comment: str = ""


def tess_magnitude(flux: np.ndarray, flux_err: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(mag, symmetric error, upper error, lower error) with the zero point and its error; NaN where flux <= 0."""
    flux = np.where(np.asarray(flux, float) > 0, flux, np.nan)
    flux_err = np.asarray(flux_err, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        mag = -2.5 * np.log10(flux) + TESSMAG_ZERO_POINT
        err = np.hypot(TESSMAG_ZERO_POINT_ERR, 2.5 / np.log(10) * flux_err / flux)
        upper = np.hypot(-2.5 * np.log10(flux - flux_err) + TESSMAG_ZERO_POINT - mag, TESSMAG_ZERO_POINT_ERR)
        lower = np.hypot(mag - (-2.5 * np.log10(flux + flux_err) + TESSMAG_ZERO_POINT), TESSMAG_ZERO_POINT_ERR)
    return mag, err, upper, lower


def exposure_table(frames: FrameTable, track: Track, geometry: FrameGeometry, timecorr: np.ndarray) -> Table:
    """Per-frame time, position, and geometry columns for the whole observation."""
    t_sc = frames.time - frames.barycorr
    time = t_sc + timecorr
    return Table({
        "TIME": time,
        "TIMECORR": timecorr,
        "ORIGINAL_TIME": frames.time,
        "ORIGINAL_TIMECORR": frames.barycorr,
        "TIME_EMIT": emission_time(time, timecorr, geometry.delta),
        "TSTART": frames.tstart - frames.barycorr + timecorr,
        "TSTOP": frames.tstop - frames.barycorr + timecorr,
        "CADENCENO": frames.cadenceno,
        "QUALITY": frames.quality,
        "EPHEM1": track.col,
        "EPHEM2": track.row,
        "RA_PRED": track.ra,
        "DEC_PRED": track.dec,
        "SUN_DISTANCE": geometry.r,
        "OBS_DISTANCE": geometry.delta,
        "STO_ANGLE": geometry.alpha,
        "VMAG_PRED": geometry.v,
    })


def weighted_ra(ra: np.ndarray, weights: np.ndarray) -> float:
    """Weighted mean RA across the 0/360 wrap."""
    unwrapped = ra[0] + (ra - ra[0] + 180.0) % 360.0 - 180.0
    return float(np.sum(weights * unwrapped) % 360.0)


AVERAGED = ("TIME", "TIMECORR", "ORIGINAL_TIME", "ORIGINAL_TIMECORR", "TIME_EMIT", "EPHEM1", "EPHEM2",
            "DEC_PRED", "SUN_DISTANCE", "OBS_DISTANCE", "STO_ANGLE", "VMAG_PRED")


@dataclass
class GroupRows:
    """Per-group values shared by the light-curve tables and EXTRAS."""

    extras: Table
    members: Table


def group_rows(phot: Photometry, exposures: Table) -> GroupRows:
    """Weighted group means of the exposure columns, and the member table."""
    rows, members = [], []
    for r in phot.groups:
        f = r.group.frames
        w = np.asarray(r.weights, float)
        e = exposures[f]
        row = {name: float(np.sum(w * np.asarray(e[name], float))) for name in AVERAGED}
        row["RA_PRED"] = weighted_ra(np.asarray(e["RA_PRED"], float), w)
        closest = int(np.argmin(np.abs(np.asarray(e["ORIGINAL_TIME"]) - row["ORIGINAL_TIME"])))
        row["CADENCENO"] = int(e["CADENCENO"][closest])
        row["GROUP_ID"] = r.group.index
        row["CORNER1"] = int(r.origins[closest][1]) if r.origins is not None else -1
        row["CORNER2"] = int(r.origins[closest][0]) if r.origins is not None else -1
        row["FRAME"] = int(f[closest])  # internal: the representative frame
        row["TIME_START"] = float(e["TSTART"][0])
        row["TIME_STOP"] = float(e["TSTOP"][-1])
        rows.append(row)
        m = e.copy()
        m["GROUP_ID"] = np.full(len(f), r.group.index)
        m["WEIGHT"] = w
        m["CORNER1"] = r.origins[:, 1] if r.origins is not None else -1
        m["CORNER2"] = r.origins[:, 0] if r.origins is not None else -1
        m["SSO_QUALITY"] = r.member_flags if r.member_flags is not None else 0
        members.append(m)
    from astropy.table import vstack

    extras = Table(rows=rows) if rows else Table()
    member_table = vstack(members) if members else Table()
    return GroupRows(extras=extras, members=member_table)


def _table_hdu(name: str, columns: Sequence[Column], data: dict, header: dict | None = None) -> fits.BinTableHDU:
    cols = [
        fits.Column(name=c.name, format=c.format, unit=c.unit, array=np.asarray(data[c.name]))
        for c in columns
    ]
    hdu = fits.BinTableHDU.from_columns(cols)
    hdu.header["EXTNAME"] = (name, "name of extension")
    for i, c in enumerate(columns, start=1):
        if c.comment:
            hdu.header.comments[f"TTYPE{i}"] = c.comment[:47]
    for key, value, comment in TIME_KEYWORDS:
        hdu.header[key] = (value, comment)
    for key, value in (header or {}).items():
        hdu.header[key] = value
    return hdu


TIME_COLUMNS = (
    Column("TIME", "D", BTJD_UNIT, "barycentric toward the asteroid, weighted mean"),
    Column("TIME_START", "D", BTJD_UNIT, "start of the first member exposure"),
    Column("TIME_STOP", "D", BTJD_UNIT, "end of the last member exposure"),
    Column("TIME_EMIT", "D", BTJD_UNIT, "emission time at the asteroid"),
    Column("CADENCENO", "J", None, "member exposure closest to TIME"),
    Column("GROUP_ID", "J", None, "group index, see GROUPS"),
    Column("SEGMENT", "I", None, "observing segment, see SEGMENTS"),
    Column("QUALITY", "J", None, "SPOC flags of the members, OR"),
    Column("SSO_QUALITY", "J", None, "tess_sso flags, see PRIMARY SSOBITn"),
)
FLUX_COLUMNS = (
    Column("FLUX", "E", "e-/s", "total asteroid flux"),
    Column("FLUX_ERR", "E", "e-/s", "random error: PIX, BKG, EXCESS in quadrature"),
    Column("FLUX_ERR_PIX", "E", "e-/s", "random: pixel noise"),
    Column("FLUX_ERR_BKG", "E", "e-/s", "random: temporal scattered-light term"),
    Column("FLUX_ERR_EXCESS", "E", "e-/s", "random: the segment's white excess"),
    Column("FLUX_ERR_SYS", "E", "e-/s", "coherent error: STAR and BKG_SYS"),
    Column("FLUX_ERR_STAR", "E", "e-/s", "coherent: stellar model"),
    Column("FLUX_ERR_BKG_SYS", "E", "e-/s", "coherent: spatial scattered-light term"),
    Column("TESSMAG", "E", "mag", "TESS magnitude, zero point TESSMAG0"),
    Column("TESSMAG_ERR", "E", "mag"),
    Column("TESSMAG_UERR", "E", "mag"),
    Column("TESSMAG_LERR", "E", "mag"),
    Column("FLUX_BKG", "E", "e-/s", "scattered light removed from FLUX"),
    Column("FLUX_BKG_ERR", "E", "e-/s", "scattered-light error, random and coherent"),
    Column("BKG_STD", "E", "e-/s", "RMS of the scattered-light fit residual"),
    Column("BKG_GRADIENT", "E", "e-/s/pixel", "scattered-light gradient at the asteroid"),
    Column("BKG_MAD", "E", "e-/s", "MAD of the scattered-light fit residual"),
    Column("BKG_CHI2", "E", None, "chi2 per pixel of that residual"),
    Column("BKG_USABLE", "E", None, "strip share near the asteroid in the fit"),
    Column("STAR_FLUX", "E", "e-/s", "stellar model removed from FLUX"),
    Column("STAR_RESIDUAL", "E", "e-/s", "static stellar residual projected on FLUX"),
    Column("MOM_CENTR1", "E", "pixel", "measured column"),
    Column("MOM_CENTR1_ERR", "E", "pixel"),
    Column("MOM_CENTR2", "E", "pixel", "measured row"),
    Column("MOM_CENTR2_ERR", "E", "pixel"),
    Column("PRF_SHIFT1", "E", "pixel", "column shift of the PRF from the prediction"),
    Column("PRF_SHIFT2", "E", "pixel", "row shift of the PRF from the prediction"),
    Column("RA", "D", "deg", "measured position"),
    Column("DEC", "D", "deg", "measured position"),
)
GROUP_COLUMNS = (
    Column("N_EXP", "I", None, "member exposures used"),
    Column("N_EXPECTED", "I", None, "exposure slots the group spans"),
    Column("COVERAGE", "E", None, "N_EXP / N_EXPECTED"),
    Column("EXPOSURE", "E", "d", "summed member exposure time"),
)
PRF_COLUMNS = TIME_COLUMNS + FLUX_COLUMNS + (
    Column("RED_CHI2", "E", None, "reduced chi2 of the joint PRF fit"),
    Column("PRF_FRACTION", "E", None, "PRF share on usable pixels"),
) + GROUP_COLUMNS
AP_COLUMNS = TIME_COLUMNS + FLUX_COLUMNS + (
    Column("FLUX_FRACTION", "E", None, "PRF share inside the usable aperture"),
    Column("NPIX", "I", None, "usable aperture pixels"),
) + GROUP_COLUMNS
EXTRAS_COLUMNS = (
    Column("TIME", "D", BTJD_UNIT, "barycentric toward the asteroid"),
    Column("TIMECORR", "D", "d", "barycentric correction toward the asteroid"),
    Column("ORIGINAL_TIME", "D", BTJD_UNIT, "SPOC time, CCD-centre correction"),
    Column("ORIGINAL_TIMECORR", "D", "d", "SPOC barycentric correction"),
    Column("TIME_EMIT", "D", BTJD_UNIT, "TIME - TIMECORR - OBS_DISTANCE / c"),
    Column("CADENCENO", "J", None, "member exposure closest to TIME"),
    Column("GROUP_ID", "J"),
    Column("CORNER1", "J", None, "column of the cutout's lower-left pixel"),
    Column("CORNER2", "J", None, "row of the cutout's lower-left pixel"),
    Column("EPHEM1", "E", "pixel", "predicted column"),
    Column("EPHEM2", "E", "pixel", "predicted row"),
    Column("RA_PRED", "D", "deg", "predicted position"),
    Column("DEC_PRED", "D", "deg", "predicted position"),
    Column("SUN_DISTANCE", "E", "AU"),
    Column("OBS_DISTANCE", "D", "AU", "distance from TESS"),  # float64: TIME_EMIT is exact from the columns
    Column("STO_ANGLE", "E", "deg", "Sun-target-observer angle"),
    Column("VMAG_PRED", "E", "mag", "predicted V magnitude"),
)
MEMBER_COLUMNS = (
    Column("GROUP_ID", "J"),
    Column("CADENCENO", "J"),
    Column("WEIGHT", "D", None, "share in every group mean"),
    Column("TIME", "D", BTJD_UNIT),
    Column("TSTART", "D", BTJD_UNIT),
    Column("TSTOP", "D", BTJD_UNIT),
    Column("TIMECORR", "D", "d"),
    Column("ORIGINAL_TIME", "D", BTJD_UNIT),
    Column("ORIGINAL_TIMECORR", "D", "d"),
    Column("TIME_EMIT", "D", BTJD_UNIT),
    Column("QUALITY", "J"),
    Column("SSO_QUALITY", "J"),
    Column("CORNER1", "J"),
    Column("CORNER2", "J"),
    Column("EPHEM1", "E", "pixel"),
    Column("EPHEM2", "E", "pixel"),
    Column("RA_PRED", "D", "deg"),
    Column("DEC_PRED", "D", "deg"),
    Column("SUN_DISTANCE", "E", "AU"),
    Column("OBS_DISTANCE", "D", "AU"),
    Column("STO_ANGLE", "E", "deg"),
)
SEGMENT_COLUMNS = (
    Column("SEGMENT", "I"),
    Column("TSTART", "D", BTJD_UNIT, "start of the first exposure"),
    Column("TSTOP", "D", BTJD_UNIT, "end of the last exposure"),
    Column("CADENCENO_START", "J"),
    Column("CADENCENO_STOP", "J"),
    Column("N_EXP", "J", None, "exposures in the segment"),
    Column("SOURCE", "8A", None, "boundary source: gap or user"),
    Column("OFFSET1", "E", "pixel", "column offset from the prediction"),
    Column("OFFSET2", "E", "pixel", "row offset from the prediction"),
    Column("OFFSET1_ERR", "E", "pixel"),
    Column("OFFSET2_ERR", "E", "pixel"),
    Column("OFFSET_NGROUPS", "J", None, "groups in the offset, after clipping"),
    Column("OFFSET_APPLIED", "L", None, "significant: shifts the PRF (segment mode)"),
    Column("EXCESS", "E", "e-/s", "white excess per exposure"),
    Column("MODEL_ERROR", "E", None, "stellar-model error fraction (epsilon)"),
)


def _light_curve_data(phot: Photometry, groups: GroupRows, frames: FrameTable, aperture: bool) -> dict:
    table = phot.to_table()
    ex = groups.extras
    prefix = "ap_" if aperture else ""
    flux = np.asarray(table[f"{prefix}flux"], float)
    flux_err = np.asarray(table[f"{prefix}flux_err"], float)
    mag, mag_err, mag_uerr, mag_lerr = tess_magnitude(flux, flux_err)
    drow, dcol = np.asarray(table["offset_row"], float), np.asarray(table["offset_col"], float)
    ra, dec = zip(*[
        pixel_offset_to_sky(frames.table, int(f), float(a), float(d), float(r), float(c))
        if np.isfinite(r) and np.isfinite(c) else (np.nan, np.nan)
        for f, a, d, r, c in zip(ex["FRAME"], ex["RA_PRED"], ex["DEC_PRED"], drow, dcol)
    ]) if len(table) else ((), ())
    if aperture:
        err_pix = np.array([r.aperture.err_pix for r in phot.groups])
        err_bkg, err_excess = np.array([r.ap_err_bkg for r in phot.groups]), np.array([r.ap_err_excess for r in phot.groups])
        err_star, err_bkg_sys = np.array([r.ap_err_star for r in phot.groups]), np.array([r.ap_err_bkg_sys for r in phot.groups])
        err_sys = np.array([r.ap_flux_err_sys for r in phot.groups])
    else:
        err_pix, err_bkg, err_excess, err_star, err_bkg_sys, err_sys = (
            np.asarray(table[c], float)
            for c in ("flux_err_pix", "flux_err_bkg", "flux_err_excess", "flux_err_star", "flux_err_bkg_sys", "flux_err_sys")
        )
    data = {
        "TIME": ex["TIME"], "TIME_START": ex["TIME_START"], "TIME_STOP": ex["TIME_STOP"], "TIME_EMIT": ex["TIME_EMIT"],
        "CADENCENO": ex["CADENCENO"], "GROUP_ID": ex["GROUP_ID"], "SEGMENT": table["segment"],
        "QUALITY": table["quality"], "SSO_QUALITY": table["sso_quality"],
        "FLUX": flux, "FLUX_ERR": flux_err, "FLUX_ERR_PIX": err_pix, "FLUX_ERR_BKG": err_bkg, "FLUX_ERR_EXCESS": err_excess,
        "FLUX_ERR_SYS": err_sys, "FLUX_ERR_STAR": err_star, "FLUX_ERR_BKG_SYS": err_bkg_sys,
        "TESSMAG": mag, "TESSMAG_ERR": mag_err, "TESSMAG_UERR": mag_uerr, "TESSMAG_LERR": mag_lerr,
        "FLUX_BKG": table[f"{prefix}bkg_flux"] if aperture else table["bkg_flux"], "FLUX_BKG_ERR": np.hypot(err_bkg, err_bkg_sys),
        "BKG_STD": table["bkg_std"], "BKG_GRADIENT": table["bkg_gradient"],
        "BKG_MAD": table["bkg_mad"], "BKG_CHI2": table["bkg_chi2"], "BKG_USABLE": table["bkg_usable"],
        "STAR_FLUX": table[f"{prefix}star_flux"],
        "STAR_RESIDUAL": table["ap_star_residual"] if aperture else table["star_residual"],
        "MOM_CENTR1": np.asarray(ex["EPHEM1"]) + dcol, "MOM_CENTR1_ERR": table["offset_col_err"],
        "MOM_CENTR2": np.asarray(ex["EPHEM2"]) + drow, "MOM_CENTR2_ERR": table["offset_row_err"],
        "PRF_SHIFT1": table["prf_shift_col"], "PRF_SHIFT2": table["prf_shift_row"],
        "RA": np.asarray(ra, float), "DEC": np.asarray(dec, float),
        "N_EXP": table["n_exp"], "N_EXPECTED": table["n_expected"], "COVERAGE": table["coverage"],
        "EXPOSURE": table["exposure"],
    }
    if aperture:
        data["FLUX_FRACTION"] = table["flux_fraction"]
        data["NPIX"] = table["ap_npix"]
    else:
        data["RED_CHI2"] = table["red_chi2"]
        data["PRF_FRACTION"] = table["prf_fraction"]
    return data


def _segments_data(segments: Sequence[Segment], exposures: Table, offsets: Sequence = (),
                   excess: dict | None = None, model_error: dict | None = None) -> dict:
    by_segment = {o.segment: o for o in offsets}

    def offset(s, name, missing):
        o = by_segment.get(s.index)
        return getattr(o, name) if o is not None else missing

    return {
        "SEGMENT": [s.index for s in segments],
        "TSTART": [float(exposures["TSTART"][s.first]) for s in segments],
        "TSTOP": [float(exposures["TSTOP"][s.last]) for s in segments],
        "CADENCENO_START": [int(exposures["CADENCENO"][s.first]) for s in segments],
        "CADENCENO_STOP": [int(exposures["CADENCENO"][s.last]) for s in segments],
        "N_EXP": [s.n_frames for s in segments],
        "SOURCE": [s.source for s in segments],
        "OFFSET1": [offset(s, "col", np.nan) for s in segments],
        "OFFSET2": [offset(s, "row", np.nan) for s in segments],
        "OFFSET1_ERR": [offset(s, "col_err", np.nan) for s in segments],
        "OFFSET2_ERR": [offset(s, "row_err", np.nan) for s in segments],
        "OFFSET_NGROUPS": [offset(s, "n_groups", 0) for s in segments],
        "OFFSET_APPLIED": [offset(s, "applied", False) for s in segments],
        "EXCESS": [(excess or {}).get(s.index, np.nan) for s in segments],
        "MODEL_ERROR": [(model_error or {}).get(s.index, np.nan) for s in segments],
    }


def _nanmedian(values) -> float | None:
    values = np.asarray(values, float)
    if not np.isfinite(values).any():
        return None
    return round(float(np.nanmedian(values)), 3)


def primary_header(
    phot: Photometry,
    groups: GroupRows,
    frames: FrameTable,
    object_name: str,
    object_fullname: str,
    provenance: dict,
    creator: str = "tess_sso lightcurve",
) -> fits.Header:
    """PRIMARY header: reference-pipeline keywords, time keywords, and tess_sso provenance.

    lightkurve recognizes a TESS file by ``TELESCOP`` and the product by ``CREATOR``:
    "lightcurve" for a light curve, "targetpixel" for a target-pixel file.
    """
    from tess_sso import __version__
    from tess_sso.quality import SsoQuality

    h = fits.Header()
    h["ORIGIN"] = ("tess_sso", "institution or software that created this file")
    h["DATE"] = (datetime.now(timezone.utc).strftime("%Y-%m-%d"), "file creation date")
    ex = groups.extras
    if len(ex):
        tstart, tstop = float(np.min(ex["TIME_START"])), float(np.max(ex["TIME_STOP"]))
        h["TELAPSE"] = (tstop - tstart, "[d] TSTOP - TSTART")
        h["TSTART"] = (tstart, "observation start time in BTJD of first group")
        h["TSTOP"] = (tstop, "observation stop time in BTJD of last group")
        h["DATE-OBS"] = (Time(tstart + BTJD0, format="jd", scale="tdb").utc.isot, "TSTART as UTC calendar date")
        h["DATE-END"] = (Time(tstop + BTJD0, format="jd", scale="tdb").utc.isot, "TSTOP as UTC calendar date")
    for key, value, comment in TIME_KEYWORDS:
        h[key] = (value, comment)
    h["RADESYS"] = ("ICRS", "reference frame of celestial coordinates")
    h["EQUINOX"] = (2000.0, "equinox of celestial coordinate system")
    h["CREATOR"] = (creator, "software used to produce this file")
    h["PROCVER"] = (__version__, "software version")
    h["FILEVER"] = (FILE_VERSION, "file format version")
    h["TIMVERSN"] = ("OGIP/93-003", "OGIP memo number for file format")
    h["TELESCOP"] = ("TESS", "telescope")
    h["INSTRUME"] = ("TESS Photometer", "detector type")
    fp = frames.provenance
    for key, comment in (("DATA_REL", "SPOC data release version number"), ("SECTOR", "Observing sector"),
                         ("CAMERA", "Camera number"), ("CCD", "CCD chip number")):
        if key in fp:
            h[key] = (fp[key], comment)
    h["OBJECT"] = (object_name, "object name")
    h["OBJNAME"] = (object_fullname[:68], "JPL full name")
    h["TESSMAG0"] = (TESSMAG_ZERO_POINT, "[mag] TESS zero-point magnitude")
    for key, column, comment in (("VMAG", "VMAG_PRED", "[mag] median predicted V magnitude"),
                                 ("SUN_DIST", "SUN_DISTANCE", "[AU] median distance from Sun"),
                                 ("OBS_DIST", "OBS_DISTANCE", "[AU] median distance from TESS"),
                                 ("STO_ANG", "STO_ANGLE", "[deg] median Sun-Target-Observer angle")):
        if len(ex):
            h[key] = (_nanmedian(ex[column]), comment)
    h["TIMECORR"] = (provenance.pop("TCORRSRC", "SPOC"), "source of the barycentric correction in TIME")
    h["TIMEEMIT"] = ("TIME - TIMECORR - OBS_DISTANCE / c", "TIME_EMIT definition")
    h["LTCORR"] = (False, "TIME is not corrected for light travel time")
    for key in ("OBJKEY", "CUBEDATE", "TESSCUBE"):
        if key in fp:
            h[key] = fp[key]
    for key, value in provenance.items():
        h[key] = value
    for flag in SsoQuality:
        h[f"SSOBIT{flag.bit_length() - 1}"] = (flag.name, "SSO_QUALITY bit")
    return h


def lightcurve_hdulist(
    phot: Photometry,
    frames: FrameTable,
    exposures: Table,
    segments: Sequence[Segment],
    object_name: str,
    object_fullname: str,
    provenance: dict,
    aperture_header: dict | None = None,
) -> fits.HDUList:
    """The light-curve file. ``exposures`` comes from :func:`exposure_table`."""
    groups = group_rows(phot, exposures)
    primary = fits.PrimaryHDU(header=primary_header(phot, groups, frames, object_name, object_fullname, dict(provenance)))
    primary.header["EXTNAME"] = ("PRIMARY", "name of extension")
    hdus = [primary]
    for name, columns, aperture, extra in (("LIGHTCURVE_PRF", PRF_COLUMNS, False, {}),
                                           ("LIGHTCURVE_AP", AP_COLUMNS, True, aperture_header or {})):
        data = _light_curve_data(phot, groups, frames, aperture) if len(phot.groups) else {c.name: [] for c in columns}
        header = dict(extra)
        mag = _nanmedian(data["TESSMAG"]) if len(phot.groups) else None
        header["TESSMAG"] = (mag, "[mag] median measured TESS magnitude")
        hdus.append(_table_hdu(name, columns, data, header))
    ex = groups.extras
    hdus.append(_table_hdu("EXTRAS", EXTRAS_COLUMNS, {c.name: ex[c.name] if len(ex) else [] for c in EXTRAS_COLUMNS}))
    m = groups.members
    hdus.append(_table_hdu("GROUPS", MEMBER_COLUMNS, {c.name: m[c.name] if len(m) else [] for c in MEMBER_COLUMNS}))
    hdus.append(_table_hdu("SEGMENTS", SEGMENT_COLUMNS, _segments_data(segments, exposures, phot.segment_offsets, phot.excess, phot.model_error)))
    return fits.HDUList(hdus)


def to_lightkurve(path, time: str = "TIME", hdu: str = "LIGHTCURVE_PRF", quality_bitmask: str | int = "default"):
    """A light-curve file as an ``lk.LightCurve`` (lightkurve must be installed).

    ``time`` is ``"TIME"`` (barycentric toward the asteroid) or
    ``"TIME_EMIT"`` (emission time at the asteroid). Every other column is
    kept, lower-cased, so the error components travel with the flux.
    Cadences with SPOC flags in ``quality_bitmask`` are dropped, as
    lightkurve's readers do.
    """
    import lightkurve as lk

    if time not in ("TIME", "TIME_EMIT"):
        raise ValueError(f"time must be 'TIME' or 'TIME_EMIT', got {time!r}")
    with fits.open(path) as hdul:
        table = Table(hdul[hdu].data)
        meta = dict(hdul[0].header)
    bad = lk.utils.TessQualityFlags.create_quality_mask(np.asarray(table["QUALITY"]), bitmask=quality_bitmask)
    table = table[bad]  # create_quality_mask returns True for cadences to keep
    lc = lk.LightCurve(
        time=Time(np.asarray(table[time], float), format="btjd", scale="tdb"),
        flux=np.asarray(table["FLUX"], float),
        flux_err=np.asarray(table["FLUX_ERR"], float),
        meta={**meta, "TIME_COLUMN": time, "HDU": hdu},
    )
    other = "TIME_EMIT" if time == "TIME" else "TIME"
    lc["time_emit" if other == "TIME_EMIT" else "time_barycentric"] = np.asarray(table[other], float)
    for name in table.colnames:
        if name not in ("FLUX", "FLUX_ERR", "TIME", "TIME_EMIT"):
            lc[name.lower()] = np.asarray(table[name])
    return lc
