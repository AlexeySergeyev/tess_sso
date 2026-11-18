"""Pixel product: the group stacks in the SPOC target-pixel-file layout, for inspection.

One row per group, in the order and with the ``CADENCENO`` of the light-curve
file. Each image is the group's stack on the grid centred on the pixel that
holds the asteroid, the grid the aperture photometry uses. Pixel (i, j) of a
row is CCD pixel (``CORNER2`` + i, ``CORNER1`` + j) of the row's
representative exposure (see ``EXTRAS``); the header's ``1CRV5P``/``2CRV5P``
give the first row's corner, since the cutout moves.

HDUs: ``PRIMARY``; ``PIXELS`` (the SPOC columns ``TIME``, ``TIMECORR``,
``CADENCENO``, then ``TIME_EMIT`` where SPOC has ``RAW_CNTS``, so ``FLUX`` is
column 5 as in SPOC files, ``FLUX_ERR``, ``FLUX_BKG``, ``FLUX_BKG_ERR``,
``QUALITY``, ``POS_CORR1``/``2``, then ``GROUP_ID``, ``SSO_QUALITY``, and
the images ``STAR_MODEL``, ``ASTEROID_MODEL``,
``RESIDUAL``, ``COVERAGE``, ``PIXEL_MASK``); ``APERTURE`` (bit 1 every pixel,
bit 2 the aperture of most groups, which lightkurve uses as the pipeline
aperture); ``EXTRAS``, as in the light-curve file.

``FLUX`` follows SPOC: the data minus the scattered light, stars left in.
``FLUX - STAR_MODEL - ASTEROID_MODEL = RESIDUAL``. There is no celestial WCS:
a single WCS cannot describe a moving cutout.
"""

from __future__ import annotations

import numpy as np
from astropy.io import fits
from astropy.table import Table

from tess_sso.output import EXTRAS_COLUMNS, TIME_KEYWORDS, _table_hdu, group_rows, primary_header
from tess_sso.photometry import Photometry
from tess_sso.quality import PixelMask
from tess_sso.storage.frames import FrameTable

# (name, FITS type, unit, comment)
SCALAR_COLUMNS_BEFORE = (
    ("TIME", "D", "BJD - 2457000, days", "barycentric toward the asteroid"),
    ("TIMECORR", "D", "d", "barycentric correction toward the asteroid"),
    ("CADENCENO", "J", None, "member exposure closest to TIME"),
    ("TIME_EMIT", "D", "BJD - 2457000, days", "emission time at the asteroid"),
)
IMAGE_COLUMNS = (
    ("FLUX", "E", "e-/s", "data minus scattered light, stars left in"),
    ("FLUX_ERR", "E", "e-/s", "pixel noise of the stack"),
    ("FLUX_BKG", "E", "e-/s", "scattered-light model"),
    ("FLUX_BKG_ERR", "E", "e-/s", "scattered-light model error"),
)
SCALAR_COLUMNS_AFTER = (
    ("QUALITY", "J", None, "SPOC flags of the members, OR"),
    ("POS_CORR1", "E", "pixel", "fitted column offset of the asteroid"),
    ("POS_CORR2", "E", "pixel", "fitted row offset of the asteroid"),
    ("GROUP_ID", "J", None, "group index, as in the light-curve file"),
    ("SSO_QUALITY", "J", None, "tess_sso flags"),
)
EXTRA_IMAGE_COLUMNS = (
    ("STAR_MODEL", "E", "e-/s", "stellar model (with the static correction)"),
    ("ASTEROID_MODEL", "E", "e-/s", "fitted flux times the stacked PRF"),
    ("RESIDUAL", "E", "e-/s", "FLUX - STAR_MODEL - ASTEROID_MODEL"),
    ("COVERAGE", "I", None, "member exposures usable at the pixel"),
    ("PIXEL_MASK", "I", None, "PixelMask bits, see PRIMARY PIXBITn"),
)


def _image_column(name: str, fmt: str, unit: str | None, images: np.ndarray) -> fits.Column:
    ny, nx = images.shape[1:]
    return fits.Column(name=name, format=f"{ny * nx}{fmt}", unit=unit, dim=f"({nx},{ny})", array=images)


def pixel_hdulist(
    phot: Photometry,
    frames: FrameTable,
    exposures: Table,
    object_name: str,
    object_fullname: str,
    provenance: dict,
) -> fits.HDUList:
    """The pixel product; ``exposures`` comes from :func:`tess_sso.output.exposure_table`."""
    if not phot.groups or phot.groups[0].images is None:
        raise ValueError("the photometry has no group images")
    groups = group_rows(phot, exposures)
    ex = groups.extras
    table = phot.to_table()

    def images(name: str) -> np.ndarray:
        return np.stack([r.images[name] for r in phot.groups])

    scalars = {
        "TIME": ex["TIME"], "TIMECORR": ex["TIMECORR"], "CADENCENO": ex["CADENCENO"], "TIME_EMIT": ex["TIME_EMIT"],
        "QUALITY": table["quality"], "POS_CORR1": table["offset_col"], "POS_CORR2": table["offset_row"],
        "GROUP_ID": ex["GROUP_ID"], "SSO_QUALITY": table["sso_quality"],
    }
    cols = [fits.Column(name=n, format=f, unit=u, array=np.asarray(scalars[n])) for n, f, u, _ in SCALAR_COLUMNS_BEFORE]
    cols += [_image_column(n, f, u, images(n)) for n, f, u, _ in IMAGE_COLUMNS]
    cols += [fits.Column(name=n, format=f, unit=u, array=np.asarray(scalars[n])) for n, f, u, _ in SCALAR_COLUMNS_AFTER]
    cols += [_image_column(n, f, u, images(n)) for n, f, u, _ in EXTRA_IMAGE_COLUMNS]
    pixels = fits.BinTableHDU.from_columns(cols)
    pixels.header["EXTNAME"] = ("PIXELS", "name of extension")
    comments = {n: c for n, _, _, c in SCALAR_COLUMNS_BEFORE + IMAGE_COLUMNS + SCALAR_COLUMNS_AFTER + EXTRA_IMAGE_COLUMNS}
    corner_col, corner_row = int(ex["CORNER1"][0]), int(ex["CORNER2"][0])
    for i, name in enumerate(pixels.columns.names, start=1):
        pixels.header.comments[f"TTYPE{i}"] = comments[name][:47]
        if name in {n for n, *_ in IMAGE_COLUMNS + EXTRA_IMAGE_COLUMNS}:
            # physical pixel coordinates of image[0, 0], as SPOC writes them (first row only: the cutout moves)
            for axis, ctype, corner in ((1, "RAWX", corner_col), (2, "RAWY", corner_row)):
                pixels.header[f"{axis}CTY{i}P"] = (ctype, "physical WCS axis type")
                pixels.header[f"{axis}CUN{i}P"] = ("PIXEL", "physical WCS axis unit")
                pixels.header[f"{axis}CRV{i}P"] = (corner, "CCD pixel of image[0, 0], first row")
                pixels.header[f"{axis}CDL{i}P"] = (1.0, "physical WCS step")
                pixels.header[f"{axis}CRP{i}P"] = (1, "reference pixel")
    for key, value, comment in TIME_KEYWORDS:
        pixels.header[key] = (value, comment)
    pixels.header["MOVING"] = (True, "cutout follows the asteroid: EXTRAS CORNER1/2")

    header = primary_header(phot, groups, frames, object_name, object_fullname, dict(provenance),
                            creator="tess_sso targetpixel")
    for flag in PixelMask:
        header[f"PIXBIT{flag.bit_length() - 1}"] = (flag.name, "PIXEL_MASK bit")
    primary = fits.PrimaryHDU(header=header)
    primary.header["EXTNAME"] = ("PRIMARY", "name of extension")

    in_aperture = (images("PIXEL_MASK") & PixelMask.APERTURE) > 0
    aperture = np.ones(in_aperture.shape[1:], np.int32) | np.where(in_aperture.mean(axis=0) >= 0.5, 2, 0).astype(np.int32)
    aperture_hdu = fits.ImageHDU(aperture, name="APERTURE")
    aperture_hdu.header["OBJECT"] = (object_name, "object name")
    aperture_hdu.header["APBITS"] = ("1 all, 2 aperture", "2: in the aperture of most groups")

    extras = _table_hdu("EXTRAS", EXTRAS_COLUMNS, {c.name: ex[c.name] for c in EXTRAS_COLUMNS})
    return fits.HDUList([primary, pixels, aperture_hdu, extras])
