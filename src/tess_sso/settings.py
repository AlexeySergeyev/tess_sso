"""Processing settings and their stable hash (names the results directory)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass


RESIDUAL_MODELS = ("local", "local_pointing", "static")


@dataclass(frozen=True)
class Settings:
    cutout_size: int = 11  # photometric cutout, pixels on a side (odd)
    strip_width: int = 31  # background strip across the track, pixels (odd, >= cutout_size)
    ephemeris_step: str = "1h"  # Horizons sampling; positions are interpolated to each frame
    group_minutes: float | None = None  # exposures per photometric point, by duration (main parameter)
    group_count: int | None = None  # ... or by number of exposures; neither: one exposure per point
    # Subtract the stellar model's mean residual on clean frames before photometry: it removes the
    # faint-star light counted twice by the scattered-light and stellar models (~3 % for asteroid 95).
    static_correction: bool = True
    # Where the asteroid PRF sits: "segment" shifts it by the segment's significant offset from the
    # prediction, "group" by each group's own offset, "predicted" not at all.
    prf_position: str = "segment"
    # The residual subtracted with the static correction: "local", a smooth function of time per pixel
    # fitted around the asteroid's passage with the asteroid left out; "static", one mean per pixel over
    # a segment (phase 12: the local residual takes 4337's slow noise from 0.090 to 0.048 mag);
    # "local_pointing" (the default), the local one also following the pointing shift where it helps:
    # on bright stars (3486, sector 45: the residual at crossings from -3.4 +- 18 to -0.1 +- 1.5 sigma).
    residual_model: str = "local_pointing"
    # A free constant background per group in the PRF fit, absorbing what the models leave in the box.
    local_background: bool = True

    def __post_init__(self):
        for name in ("cutout_size", "strip_width"):
            value = getattr(self, name)
            if value < 1 or value % 2 == 0:
                raise ValueError(f"{name} must be a positive odd number, got {value}")
        if self.group_minutes is not None and self.group_count is not None:
            raise ValueError("set group_minutes or group_count, not both")
        if self.group_minutes is not None and self.group_minutes <= 0:
            raise ValueError(f"group_minutes must be positive, got {self.group_minutes}")
        if self.group_count is not None and self.group_count < 1:
            raise ValueError(f"group_count must be >= 1, got {self.group_count}")
        if self.residual_model not in RESIDUAL_MODELS:
            raise ValueError(f"residual_model must be one of {RESIDUAL_MODELS}, got {self.residual_model!r}")
        if self.prf_position not in ("segment", "group", "predicted"):
            raise ValueError(f"prf_position must be 'segment', 'group', or 'predicted', got {self.prf_position!r}")
        if self.strip_width < self.cutout_size:
            raise ValueError(
                f"strip_width ({self.strip_width}) must be >= cutout_size ({self.cutout_size})"
            )

    def to_dict(self) -> dict:
        return asdict(self)

    def hash(self) -> str:
        """First 12 hex digits of the SHA-256 of the settings as sorted JSON."""
        text = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(text.encode()).hexdigest()[:12]
