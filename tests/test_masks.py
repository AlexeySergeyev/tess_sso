import numpy as np
import pytest
from astropy.io import fits

from conftest import FakeSource, GaussianPrf
from tess_sso.storage.download import download_pixels
from tess_sso.storage.store import PixelStore
from tess_sso.geometry import Track
from tess_sso.masks import (
    BIT_ASTEROID,
    BIT_USABLE,
    Grid,
    ObservationMasks,
    asteroid_footprints,
    asteroid_thresholds,
    build_segment_masks,
    data_driven_star_mask,
    ephemeris_margin_px,
    exposure_ends,
    gaia_star_mask,
    segment_medians,
    write_masks,
)
from tess_sso.prf import render_stars
from tess_sso.segments import Segment


def test_grid():
    grid = Grid.around([(10, 20), (12, 25)])
    assert grid.origin == (10, 20) and grid.shape == (3, 6)
    image = grid.image([(10, 20), (12, 25)])
    assert image[0, 0] and image[2, 5] and image.sum() == 2


def test_render_stars_conserves_flux_and_clips():
    prf = GaussianPrf()
    inside = render_stars(prf, np.array([20.0]), np.array([20.0]), np.array([1000.0]), (1, 1), (40, 40))
    edge = render_stars(prf, np.array([0.5]), np.array([20.0]), np.array([1000.0]), (1, 1), (40, 40))
    assert inside.sum() == pytest.approx(1000.0, rel=1e-6)
    assert edge.sum() == pytest.approx(500.0, rel=1e-6)  # centred on the image edge: half is off


def star_field(bump_at=(20, 20), flux=5000.0):
    image = np.full((40, 40), 100.0)
    image += render_stars(GaussianPrf(), np.array([bump_at[0] + 1.0]), np.array([bump_at[1] + 1.0]), np.array([flux]), (1, 1), (40, 40))
    return image


def test_data_driven_mask_finds_a_star_the_catalogue_lacks():
    image = star_field()
    valid = np.ones_like(image, dtype=bool)
    mask = data_driven_star_mask(image, valid)
    assert mask[20, 20] and not mask[5, 5]
    valid[18:23, 18:23] = False
    assert not data_driven_star_mask(image, valid)[20, 20]


def test_gaia_mask_threshold():
    star = np.array([[0.5, 2.0], [10.0, 0.0]])
    noise = np.full((2, 2), 1.0)
    assert gaia_star_mask(star, noise, k=1.0).tolist() == [[False, True], [True, False]]
    assert gaia_star_mask(star, noise, k=3.0).tolist() == [[False, False], [True, False]]


def track(n=5, on=True):
    t = 3234.0 + np.arange(n) * 200 / 86400
    return Track(time=t, ra=np.zeros(n), dec=np.zeros(n), row=20.0 + np.arange(n) * 0.5, col=np.full(n, 20.0),
                 wcs_fallback=np.zeros(n, bool), on_science=np.full(n, on))


def test_exposure_ends_follow_velocity():
    tr = track()
    r0, c0, r1, c1 = exposure_ends(tr, np.full(len(tr), 200 / 86400))
    np.testing.assert_allclose(r1 - r0, 0.5)
    np.testing.assert_allclose(c1 - c0, 0.0)


def test_asteroid_footprints():
    grid = Grid((1, 1), (40, 40))
    tr = track()
    exposure = np.full(len(tr), 200 / 86400)
    plain = asteroid_footprints(GaussianPrf(), tr, exposure, grid)
    wide = asteroid_footprints(GaussianPrf(), tr, exposure, grid, margin_px=2)
    centre = np.ravel_multi_index((19, 19), grid.shape)  # pixel (20, 20)
    assert centre in plain[0]
    assert len(wide[0]) > len(plain[0])
    off = asteroid_footprints(GaussianPrf(), track(on=False), exposure, grid)
    assert all(len(f) == 0 for f in off)


def test_segment_medians_skip_bad_frames(tmp_path):
    source = FakeSource(nframes=5)
    store = PixelStore(tmp_path / "95" / "s0071-2-2", 71, 2, 2)
    pixels = [(500, 1030), (501, 1031), (500, 1031)]
    download_pixels(store, pixels, source.fetch)
    pixels = pixels[:2]
    grid = Grid.around(pixels)
    strip = grid.image(pixels)
    segments = [Segment(0, 0, 2, 0.0, 1.0, "gap"), Segment(1, 3, 4, 2.0, 3.0, "gap")]
    bad = np.array([False, False, True, False, False])
    median, noise = segment_medians(store, grid, strip, segments, bad)
    assert median[0, 0, 0] == 500 * 1000 + 1030
    assert noise[0, 0, 0] == 0.5  # frames 0 and 1 (frame 2 is bad): error = frame index
    assert noise[1, 0, 0] == 3.5
    assert np.isnan(median[0, 0, 1])  # (500, 1031) is not in the strip


def test_build_masks_and_write(tmp_path):
    grid = Grid((1, 1), (40, 40))
    strip = np.ones(grid.shape, bool)
    median = star_field(bump_at=(10, 10))
    noise = np.full(grid.shape, 1.0)
    star_image = render_stars(GaussianPrf(), np.array([31.0]), np.array([31.0]), np.array([500.0]), (1, 1), (40, 40))
    footprints = asteroid_footprints(GaussianPrf(), track(), np.full(5, 200 / 86400), grid)
    m = build_segment_masks(0, strip, median, noise, star_image, footprints)
    assert m.star_gaia[30, 30] and not m.star_data[30, 30]  # catalogue star, not visible in the data
    assert m.star_data[10, 10] and not m.star_gaia[10, 10]  # visible star missing from the catalogue
    assert m.one_mask_only[30, 30] and m.one_mask_only[10, 10]
    assert m.asteroid[19, 19] and not m.usable[19, 19]
    assert m.usable[0, 39] and 0.5 < m.usable_fraction < 1.0
    seg = Segment(0, 0, 4, 0.0, 1.0, "gap")
    obs = ObservationMasks(grid, [seg], [m], footprints, median[None], noise[None], star_image, {"KSIGMA": 1.0})
    assert "segment 0: usable" in obs.describe()
    assert "data only (missing from Gaia) " in obs.describe()
    path = tmp_path / "masks.fits"
    write_masks(path, obs)
    with fits.open(path) as hdul:
        bits = hdul["MASKS"].data[0]
        assert bits[19, 19] & BIT_ASTEROID and not bits[19, 19] & BIT_USABLE
        assert hdul[0].header["KSIGMA"] == 1.0
        assert len(hdul["FOOTPRINTS"].data) == sum(len(f) for f in footprints)


def test_ephemeris_margin_rounds_to_whole_pixels():
    assert ephemeris_margin_px(0.011) == 0  # numbered asteroid: no margin
    assert ephemeris_margin_px(10.4) == 0
    assert ephemeris_margin_px(10.6) == 1
    assert ephemeris_margin_px(60.0) == 3
    assert ephemeris_margin_px(float("nan")) == 0


def test_bright_asteroid_gets_a_wider_footprint():
    thresholds = asteroid_thresholds(np.array([12.0, 20.0, np.nan]), noise=1.3)
    assert thresholds[0] < 0.001  # ~3800 e-/s: wings above 1.3 e-/s reach far below 1 %
    assert thresholds[1] == 0.01 and thresholds[2] == 0.01  # faint or unknown: 1 %
    grid = Grid((1, 1), (40, 40))
    tr = track()
    exposure = np.full(len(tr), 200 / 86400)
    plain = asteroid_footprints(GaussianPrf(), tr, exposure, grid)
    bright = asteroid_footprints(GaussianPrf(), tr, exposure, grid, np.full(len(tr), thresholds[0]))
    assert len(bright[0]) > len(plain[0])


def test_dense_field_raises_k_to_keep_part_of_the_strip():
    from tess_sso.masks import K_SIGMA, MIN_USABLE_FRACTION

    shape = (40, 40)
    strip = np.ones(shape, bool)
    noise = np.full(shape, 1.0)
    # unresolved stars everywhere: 20-200 x the noise, far above K_SIGMA
    star_image = np.linspace(20.0, 200.0, strip.size).reshape(shape)
    m = build_segment_masks(0, strip, np.full(shape, 100.0), noise, star_image, [])
    assert m.k > K_SIGMA
    assert MIN_USABLE_FRACTION <= m.usable_fraction < MIN_USABLE_FRACTION + 0.01
    assert star_image[m.usable].max() < star_image[m.star_gaia].min()  # the faintest pixels are kept


def test_sparse_field_keeps_k():
    from tess_sso.masks import K_SIGMA

    shape = (40, 40)
    star_image = render_stars(GaussianPrf(), np.array([31.0]), np.array([31.0]), np.array([500.0]), (1, 1), shape)
    m = build_segment_masks(0, np.ones(shape, bool), np.full(shape, 100.0), np.full(shape, 1.0), star_image, [])
    assert m.k == K_SIGMA
