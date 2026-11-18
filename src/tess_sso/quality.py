"""Quality flags: the SPOC bad-cadence bitmask and tess_sso's own SSO_QUALITY bits."""

from __future__ import annotations

from enum import IntFlag

import numpy as np

# SPOC QUALITY bits (1-based) treated as bad: lightkurve's TessQualityFlags.DEFAULT_BITMASK.
# AttitudeTweak, SafeMode, CoarsePoint, EarthPoint, Argabrightening, Desat,
# ManualExclude, ImpulsiveOutlier, BadCalibrationExclude.
SPOC_BAD_BITS = (1, 2, 3, 4, 5, 6, 8, 10, 15)
SPOC_BAD_BITMASK = sum(1 << (bit - 1) for bit in SPOC_BAD_BITS)  # 17087


def spoc_bad(quality: np.ndarray) -> np.ndarray:
    """True for frames with any bad SPOC bit set."""
    return (np.asarray(quality, dtype=np.int64) & SPOC_BAD_BITMASK) != 0


class SsoQuality(IntFlag):
    """Bits of the ``SSO_QUALITY`` column. A set bit flags a point; points are never removed."""

    WCS_FALLBACK = 1 << 0  # frame had no WCS; the average WCS was used
    OFF_SCIENCE = 1 << 1  # asteroid outside the CCD science area
    FEW_BACKGROUND_PIXELS = 1 << 2  # too few usable pixels for the scattered-light model
    BACKGROUND_GRADIENT = 1 << 3  # strong scattered-light gradient at the asteroid
    BACKGROUND_POOR_FIT = 1 << 4  # large scattered-light fit residuals
    STELLAR_MODEL_UNCERTAIN = 1 << 5  # stellar-model error or residual larger than the pixel-noise error
    STELLAR_VARIABILITY = 1 << 6  # star residuals differ before and after the crossing
    SATURATED_STAR = 1 << 7  # asteroid crosses a saturated star or bleed column
    MASKED_APERTURE_PIXELS = 1 << 8  # aperture lost pixels to masks; FLUX_FRACTION reduced
    LOW_GROUP_COVERAGE = 1 << 9  # group coverage below the threshold
    INCOMPLETE_GROUP = 1 << 10  # group cut short by a segment edge
    ONE_STAR_MASK_ONLY = 1 << 11  # under the asteroid: a source seen in the data but not in Gaia


def describe(value: int) -> list[str]:
    """Names of the SSO_QUALITY bits set in ``value``."""
    return [flag.name for flag in SsoQuality if value & flag]


class PixelMask(IntFlag):
    """Bits of the ``PIXEL_MASK`` images of the pixel product, per group stack."""

    STAR = 1 << 0  # in the star mask (Gaia or data) for some member exposure
    ASTEROID = 1 << 1  # in the asteroid footprint of some member exposure
    USED = 1 << 2  # usable in at least one member exposure (COVERAGE > 0)
    APERTURE = 1 << 3  # in the aperture drawn on the stack PRF
