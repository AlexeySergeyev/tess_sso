import numpy as np
import pytest
from astropy.table import Table

from conftest import WcsFakeSource
from tess_sso.asteroid import Asteroid, WcsDisagreementError, WcsDisagreementWarning
from tess_sso.designation import Target
from tess_sso.settings import Settings
from tess_sso.storage.ecsv_io import read_table

TARGET = Target("95", "95 Arethusa (A867 WA)")
RA, DEC = 72.67498, 24.41023


def horizons(designation, start, stop, step):
    n = 48
    return Table({"datetime_jd": start + np.arange(n) / 24, "RA": RA + 0.1 * np.arange(n) / 24,
                  "DEC": np.full(n, DEC), "V": np.full(n, 12.3), "r": np.full(n, 2.8),
                  "delta": np.full(n, 1.9), "lighttime": np.full(n, 15.6), "alpha": np.full(n, 17.3)})


def make(tmp_path, settings=Settings(), crval_shift_deg=0.0):
    sources = {}

    def source(sector, camera, ccd):
        sources[(sector, camera, ccd)] = WcsFakeSource(crval_shift_deg=crval_shift_deg)
        return sources[(sector, camera, ccd)]

    ast = Asteroid("95", root=tmp_path, settings=settings, resolve=lambda n: TARGET,
                   sectors=lambda d: [(71, 2, 2)], horizons=horizons, source=source)
    return ast, sources


def test_observations(tmp_path):
    ast, _ = make(tmp_path)
    table = ast.observations()
    assert len(table) == 1 and table["on_ccd"][0]
    assert repr(ast) == "Asteroid('95 Arethusa (A867 WA)')"


def test_plan_then_download_then_nothing_missing(tmp_path):
    ast, sources = make(tmp_path, Settings(cutout_size=5, strip_width=21))
    plan = ast.plan(71, 2, 2)
    assert plan.missing == sorted(plan.strip_pixels) and len(plan.strip_pixels) >= 21 * 21
    assert set(plan.cutout_pixels) <= set(plan.strip_pixels)
    assert "additional download required" in plan.describe()
    assert plan.missing_bytes == len(plan.missing) * 6 * 8
    report = ast.download(71, 2, 2)
    assert report.downloaded == len(plan.missing)
    stored = {tuple(p) for p in ast.store(71, 2, 2).coverage()}
    assert stored == set(plan.strip_pixels)  # exactly the strip, nothing more
    again = ast.plan(71, 2, 2)
    assert again.missing == [] and "nothing to download" in again.describe()


def test_stored_files_are_used_without_opening_the_cube(tmp_path):
    ast, _ = make(tmp_path, Settings(cutout_size=5, strip_width=21))
    ast.download(71, 2, 2)

    def offline(sector, camera, ccd):
        raise AssertionError("the cube was opened although everything is stored")

    again = Asteroid("95", root=tmp_path, settings=ast.settings, resolve=lambda n: TARGET,
                     sectors=lambda d: [(71, 2, 2)], horizons=horizons, source=offline)
    assert again.plan(71, 2, 2).missing == []
    report = again.download(71, 2, 2)
    assert report.downloaded == 0 and report.already_stored == len(again.plan(71, 2, 2).strip_pixels)
    assert again.segments(71, 2, 2)


def test_narrower_settings_need_nothing_new(tmp_path):
    ast, _ = make(tmp_path, Settings(cutout_size=5, strip_width=41))
    ast.download(71, 2, 2)
    narrow, _ = make(tmp_path, Settings(cutout_size=3, strip_width=21))
    assert narrow.plan(71, 2, 2).missing == []


def test_wider_strip_reports_only_new_pixels(tmp_path):
    ast, _ = make(tmp_path, Settings(cutout_size=5, strip_width=5))
    first = ast.plan(71, 2, 2)
    ast.download(71, 2, 2)
    wide, _ = make(tmp_path, Settings(cutout_size=5, strip_width=81))
    plan = wide.plan(71, 2, 2)
    assert set(plan.missing) == set(plan.strip_pixels) - set(first.strip_pixels)
    assert plan.missing


def test_results_dir_uses_settings_hash(tmp_path):
    ast, _ = make(tmp_path)
    assert ast.results_dir(71, 2, 2) == tmp_path / "95" / "s0071-2-2" / Settings().hash()
    assert not ast.results_dir(71, 2, 2).exists()
    created = ast.results_dir(71, 2, 2, create=True)
    assert '"strip_width": 31' in (created / "settings.json").read_text()


def test_everything_lives_in_the_asteroid_folder(tmp_path):
    ast, _ = make(tmp_path, Settings(cutout_size=5, strip_width=11))
    ast._orbit_times = lambda: ONE_ORBIT_FOR_LAYOUT
    ast.observations()
    ast.download(71, 2, 2)
    ast.segments(71, 2, 2)
    top = sorted(p.name for p in tmp_path.iterdir())
    assert top == ["95", "reference"]
    assert sorted(p.name for p in (tmp_path / "reference").iterdir()) == ["TESS_orbit_times.csv", "designations.json"]
    assert ast.directory == tmp_path / "95"
    assert (tmp_path / "95" / "observations.ecsv").exists()
    assert list((tmp_path / "95" / "ephemerides").glob("horizons_*.ecsv"))
    assert (tmp_path / "95" / "s0071-2-2" / "frames.fits").exists()
    assert list((tmp_path / "95" / "s0071-2-2" / "pixels").glob("r*.fits"))


ONE_ORBIT_FOR_LAYOUT = """Sector,Orbit,Start of Orbit,End of Orbit
71,149,2023-10-16 11:30:00,2023-10-16 12:30:00
"""


def test_wcs_disagreement_is_reported(tmp_path):
    ast, _ = make(tmp_path, crval_shift_deg=0.01)  # about 1.7 px
    with pytest.warns(WcsDisagreementWarning):
        ast.track(71, 2, 2)


def test_wcs_disagreement_of_several_pixels_stops(tmp_path):
    ast, _ = make(tmp_path, crval_shift_deg=0.05)  # about 8.6 px: the photometry would miss the asteroid
    with pytest.raises(WcsDisagreementError, match="s0071-2-2"):
        ast.track(71, 2, 2)


ONE_ORBIT = """Sector,Orbit,Start of Orbit,End of Orbit
71,149,2023-10-16 11:30:00,2023-10-16 12:30:00
"""
TWO_ORBITS = ONE_ORBIT + "71,149,2023-10-16 13:00:00,2023-10-16 14:00:00\n"


def test_segments_cross_checked(tmp_path, recwarn):
    ast, _ = make(tmp_path)
    ast._orbit_times = lambda: ONE_ORBIT
    segments = ast.segments(71, 2, 2)
    assert [(s.first, s.last) for s in segments] == [(0, 5)]
    assert not [w for w in recwarn if w.category.__name__ == "DownlinkMismatchWarning"]


def test_segment_mismatch_warns(tmp_path):
    from tess_sso.segments import DownlinkMismatchWarning

    ast, _ = make(tmp_path)
    ast._orbit_times = lambda: TWO_ORBITS
    with pytest.warns(DownlinkMismatchWarning, match="1 segment"):
        ast.segments(71, 2, 2)


def test_user_segments_override(tmp_path):
    ast, _ = make(tmp_path)
    frames_t = ast.segments(71, 2, 2, bounds=[(0.0, 5000.0)])
    assert frames_t[0].source == "user"


def test_masks_need_the_strip_then_run(tmp_path):
    from astropy.table import Table

    from conftest import GaussianPrf
    from tess_sso.storage.store import MissingPixelsError

    def gaia(adql):
        return Table({"source_id": [7], "ra": [RA + 0.01], "dec": [DEC], "pmra": [0.0], "pmdec": [0.0],
                      "phot_g_mean_mag": [11.0], "bp_rp": [0.8]})

    ast, _ = make(tmp_path, Settings(cutout_size=5, strip_width=21))
    ast._gaia = gaia
    ast._prf = lambda s, c, d: GaussianPrf()
    ast._orbit_times = lambda: ONE_ORBIT
    with pytest.raises(MissingPixelsError):
        ast.masks(71, 2, 2)
    ast.download(71, 2, 2)
    result = ast.masks(71, 2, 2)
    assert len(result.per_segment) == 1
    assert result.per_segment[0].asteroid.any()
    assert result.header["NSTARS"] == 1
    assert (ast.results_dir(71, 2, 2) / "masks.fits").exists()


def test_scattered_light_reproduces_the_background(tmp_path):
    from astropy.table import Table

    from conftest import GaussianPrf, pixel_value

    def gaia(adql):
        return Table({"source_id": [7], "ra": [RA + 0.01], "dec": [DEC], "pmra": [0.0], "pmdec": [0.0],
                      "phot_g_mean_mag": [11.0], "bp_rp": [0.8]})

    ast, _ = make(tmp_path, Settings(cutout_size=5, strip_width=21))
    ast._gaia = gaia
    ast._prf = lambda s, c, d: GaussianPrf()
    ast._orbit_times = lambda: ONE_ORBIT
    ast.download(71, 2, 2)
    model = ast.scattered_light(71, 2, 2)
    # FakeSource flux is row * 1000 + col: a plane, which the splines reproduce exactly,
    # including under the masked asteroid footprint
    masks = ast.masks(71, 2, 2, save=False)
    footprint = np.argwhere(masks.per_segment[0].asteroid) + np.array(masks.grid.origin)
    expected = np.array([pixel_value(r, c) for r, c in footprint])
    np.testing.assert_allclose(model.evaluate(footprint, [0])[0], expected, rtol=1e-6)
    assert (ast.results_dir(71, 2, 2) / "scattered_light.fits").exists()


def test_asteroid_mask_covers_frames_outside_the_time_range(tmp_path):
    from astropy.table import Table

    from conftest import GaussianPrf

    def gaia(adql):
        return Table({"source_id": [7], "ra": [RA + 0.01], "dec": [DEC], "pmra": [0.0], "pmdec": [0.0],
                      "phot_g_mean_mag": [11.0], "bp_rp": [0.8]})

    ast, _ = make(tmp_path, Settings(cutout_size=5, strip_width=21))
    ast._gaia, ast._prf, ast._orbit_times = gaia, (lambda s, c, d: GaussianPrf()), (lambda: ONE_ORBIT)
    track = ast.track(71, 2, 2)
    stretch = (track.time[2], track.time[3])  # frames 2-3 only
    ast.download(71, 2, 2, time_range=stretch)
    masks = ast.masks(71, 2, 2, time_range=stretch, save=False)
    assert all(len(masks.footprints[i]) > 0 for i in range(len(track)))  # every frame, not just 2-3


def test_stellar_model_runs_on_the_fake_cube(tmp_path):
    from astropy.table import Table

    from conftest import GaussianPrf

    ast, _ = make(tmp_path, Settings(cutout_size=5, strip_width=21))
    track = ast.track(71, 2, 2)
    # 5 px from the track: inside the fit region, but never under the asteroid footprint
    star_ra, star_dec = track.ra[0], track.dec[0] + 5 * 21 / 3600

    def gaia(adql):
        return Table({"source_id": [7], "ra": [star_ra], "dec": [star_dec], "pmra": [0.0], "pmdec": [0.0],
                      "phot_g_mean_mag": [11.0], "bp_rp": [0.8]})

    ast._gaia, ast._prf, ast._orbit_times = gaia, (lambda s, c, d: GaussianPrf()), (lambda: ONE_ORBIT)
    ast.download(71, 2, 2)
    model = ast.stellar_model(71, 2, 2)
    assert len(model.components) == 1 and len(model.segments) == 1
    seg = model.segments[0]
    assert seg.diagnostics.fitted[0] and seg.diagnostics.n_clean[0] > 0
    # the fake cube is a plane with no stars: the fitted flux is ~0, far below the Gaia prior
    assert abs(seg.fluxes[0]) < 0.01 * model.components.prior_flux[0]
    assert (ast.results_dir(71, 2, 2) / "stellar.fits").exists()


def test_star_under_the_asteroid_keeps_its_prior_and_is_flagged(tmp_path):
    from astropy.table import Table

    from conftest import GaussianPrf

    ast, _ = make(tmp_path, Settings(cutout_size=5, strip_width=21))
    track = ast.track(71, 2, 2)
    star_ra, star_dec = track.ra[0] + 0.001, track.dec[0]  # ~0.2 px: covered in every frame

    def gaia(adql):
        return Table({"source_id": [7], "ra": [star_ra], "dec": [star_dec], "pmra": [0.0], "pmdec": [0.0],
                      "phot_g_mean_mag": [11.0], "bp_rp": [0.8]})

    ast._gaia, ast._prf, ast._orbit_times = gaia, (lambda s, c, d: GaussianPrf()), (lambda: ONE_ORBIT)
    ast.download(71, 2, 2)
    seg = ast.stellar_model(71, 2, 2, save=False).segments[0]
    assert seg.diagnostics.n_clean[0] == 0 and seg.diagnostics.poorly_constrained[0]


def test_photometry_on_the_fake_cube(tmp_path):
    from astropy.table import Table

    from conftest import GaussianPrf

    ast, _ = make(tmp_path, Settings(cutout_size=5, strip_width=21, group_count=2))
    track = ast.track(71, 2, 2)
    star_ra, star_dec = track.ra[0], track.dec[0] + 5 * 21 / 3600

    def gaia(adql):
        return Table({"source_id": [7], "ra": [star_ra], "dec": [star_dec], "pmra": [0.0], "pmdec": [0.0],
                      "phot_g_mean_mag": [11.0], "bp_rp": [0.8]})

    ast._gaia, ast._prf, ast._orbit_times = gaia, (lambda s, c, d: GaussianPrf()), (lambda: ONE_ORBIT)
    ast.download(71, 2, 2)
    table = ast.photometry(71, 2, 2).to_table()
    assert list(table["n_exp"]) == [2, 2, 2]
    # the fake cube has no asteroid: the measured flux is ~0
    assert np.all(np.abs(table["flux"]) < 1.0)
    assert np.all(table["flux_err"] >= table["flux_err_pix"])
    saved = read_table(ast.results_dir(71, 2, 2) / "photometry.ecsv")
    assert len(saved) == 3 and len(read_table(ast.results_dir(71, 2, 2) / "photometry_groups.ecsv")) == 6
    assert saved.meta["PRFPOS"] == "segment"


def test_static_correction_shifts_the_flux_by_the_projected_residual(tmp_path):
    from astropy.table import Table

    from conftest import GaussianPrf

    ast, _ = make(tmp_path, Settings(cutout_size=5, strip_width=21))
    track = ast.track(71, 2, 2)
    star_ra, star_dec = track.ra[0], track.dec[0] + 5 * 21 / 3600

    def gaia(adql):
        return Table({"source_id": [7], "ra": [star_ra], "dec": [star_dec], "pmra": [0.0], "pmdec": [0.0],
                      "phot_g_mean_mag": [11.0], "bp_rp": [0.8]})

    ast._gaia, ast._prf, ast._orbit_times = gaia, (lambda s, c, d: GaussianPrf()), (lambda: ONE_ORBIT)
    ast.download(71, 2, 2)
    masks = ast.masks(71, 2, 2, save=False)
    scattered = ast.scattered_light(71, 2, 2, masks=masks, save=False)
    stellar = ast.stellar_model(71, 2, 2, masks=masks, scattered=scattered, save=False)
    stellar.segments[0].mean_residual[:] = 5.0  # a static residual of 5 e-/s on every pixel
    tables = {}
    for on in (False, True):
        ast.settings = Settings(cutout_size=5, strip_width=21, static_correction=on, residual_model="static",
                                local_background=False)
        tables[on] = ast.photometry(71, 2, 2, masks=masks, scattered=scattered, stellar=stellar, save=False).to_table()
    residual = np.array(tables[True]["star_residual"])
    assert np.all(residual[1:] > 0)  # frame 0 of the fake cube has zero errors, so no weight
    np.testing.assert_allclose(tables[True]["flux"], np.array(tables[False]["flux"]) - residual, rtol=1e-9, atol=1e-9)


def _one_star_asteroid(tmp_path):
    from astropy.table import Table

    from conftest import GaussianPrf

    ast, _ = make(tmp_path, Settings(cutout_size=5, strip_width=21))
    track = ast.track(71, 2, 2)
    star_ra, star_dec = track.ra[0], track.dec[0] + 5 * 21 / 3600

    def gaia(adql):
        return Table({"source_id": [7], "ra": [star_ra], "dec": [star_dec], "pmra": [0.0], "pmdec": [0.0],
                      "phot_g_mean_mag": [11.0], "bp_rp": [0.8]})

    ast._gaia, ast._prf, ast._orbit_times = gaia, (lambda s, c, d: GaussianPrf()), (lambda: ONE_ORBIT)
    ast.download(71, 2, 2)
    return ast


def test_no_background_model_stops_the_stellar_model(tmp_path):
    from tess_sso.asteroid import NoBackgroundModelError
    from tess_sso.scattered_light import ScatteredLightModel

    ast = _one_star_asteroid(tmp_path)
    with pytest.raises(NoBackgroundModelError, match="s0071-2-2: no segment has a scattered-light model"):
        ast.stellar_model(71, 2, 2, scattered=ScatteredLightModel([]), save=False)


def test_no_background_model_writes_no_light_curve(tmp_path):
    from tess_sso.asteroid import NoBackgroundModelError
    from tess_sso.scattered_light import ScatteredLightModel

    ast = _one_star_asteroid(tmp_path)
    stellar = ast.stellar_model(71, 2, 2, save=False)
    with pytest.raises(NoBackgroundModelError):
        ast.lightcurve(71, 2, 2, scattered=ScatteredLightModel([]), stellar=stellar)
    assert not ast.lightcurve_path(71, 2, 2).exists()


def _dense_field_asteroid(tmp_path):
    from astropy.table import Table

    from conftest import GaussianPrf

    ast, _ = make(tmp_path, Settings(cutout_size=5, strip_width=21))
    track = ast.track(71, 2, 2)
    step = 2 * 21 / 3600  # a star every 2 px over the whole strip
    dra, ddec = np.meshgrid(np.arange(-7, 8) * step / np.cos(np.radians(DEC)), np.arange(-7, 8) * step)

    def gaia(adql):
        n = dra.size
        return Table({"source_id": np.arange(n), "ra": track.ra[0] + dra.ravel(), "dec": track.dec[0] + ddec.ravel(),
                      "pmra": np.zeros(n), "pmdec": np.zeros(n), "phot_g_mean_mag": 11.0 + np.arange(n) / n,
                      "bp_rp": np.full(n, 0.8)})

    ast._gaia, ast._prf, ast._orbit_times = gaia, (lambda s, c, d: GaussianPrf()), (lambda: ONE_ORBIT)
    ast.download(71, 2, 2)
    return ast


def test_dense_field_raises_the_gaia_mask_threshold_and_warns(tmp_path):
    from tess_sso.masks import K_SIGMA, MIN_USABLE_FRACTION, DenseFieldWarning

    ast = _dense_field_asteroid(tmp_path)
    with pytest.warns(DenseFieldWarning, match="s0071-2-2, segment 0: Gaia mask threshold raised"):
        masks = ast.masks(71, 2, 2)
    assert masks.header["KSIGMA"] == K_SIGMA and masks.header["KSIG0"] > K_SIGMA
    assert masks.per_segment[0].usable_fraction >= MIN_USABLE_FRACTION
    assert "k = " in masks.describe()


@pytest.mark.filterwarnings("ignore::tess_sso.masks.DenseFieldWarning")  # the masks it builds first
def test_dense_field_stellar_model_is_tiled_and_matches_the_whole_fit(tmp_path, monkeypatch):
    from astropy.io import fits

    ast = _dense_field_asteroid(tmp_path)
    masks = ast.masks(71, 2, 2, save=False)
    scattered = ast.scattered_light(71, 2, 2, masks=masks, save=False)
    whole = ast.stellar_model(71, 2, 2, masks=masks, scattered=scattered, save=False)
    monkeypatch.setattr("tess_sso.asteroid.TILE_COMPONENTS", 40)
    tiled = ast.stellar_model(71, 2, 2, masks=masks, scattered=scattered)
    assert len(whole.segments[0].covariance.blocks) == 1 and len(tiled.segments[0].covariance.blocks) > 1
    err = np.sqrt(whole.segments[0].covariance.diag())  # some fluxes are ~0: compare in units of their errors
    assert np.max(np.abs(tiled.segments[0].fluxes - whole.segments[0].fluxes) / err) < 1e-4
    assert fits.getheader(ast.results_dir(71, 2, 2) / "stellar.fits")["TILECOMP"] == 40
    tables = [ast.photometry(71, 2, 2, masks=masks, scattered=scattered, stellar=model, save=False).to_table()
              for model in (whole, tiled)]
    for column in ("flux", "flux_err"):  # the asteroid's measurement does not see the tiles
        np.testing.assert_allclose(tables[1][column], tables[0][column], rtol=1e-6, equal_nan=True)


def test_stellar_fit_ignores_pixels_with_an_unconstrained_background(tmp_path, monkeypatch):
    ast = _one_star_asteroid(tmp_path)
    masks = ast.masks(71, 2, 2, save=False)
    scattered = ast.scattered_light(71, 2, 2, masks=masks, save=False)
    seg = scattered.segments[0]
    loose = lambda pixels: np.asarray(pixels)[:, 1] % 2 == 0  # every other column constrained
    monkeypatch.setattr(seg, "constrained", loose)
    captured = {}

    def spy(*args, **kwargs):
        captured["weight"] = args[6]
        return fit_segment(*args, **kwargs)

    from tess_sso.stellar import fit_segment

    monkeypatch.setattr("tess_sso.asteroid.fit_stellar_segment", spy)
    model = ast.stellar_model(71, 2, 2, masks=masks, scattered=scattered, save=False)
    free = ~loose(model.pixels)
    assert free.any() and not captured["weight"][:, free].any()
    assert captured["weight"][:, ~free].any()


@pytest.mark.filterwarnings("ignore::tess_sso.masks.DenseFieldWarning")
def test_segments_where_the_mask_threshold_was_raised_are_dense(tmp_path):
    assert all(s.dense for s in _dense_field_asteroid(tmp_path / "dense").stellar_model(71, 2, 2, save=False).segments)
    assert not any(s.dense for s in _one_star_asteroid(tmp_path / "sparse").stellar_model(71, 2, 2, save=False).segments)


def test_photometry_with_the_pointing_residual_runs_on_the_fake_cube(tmp_path):
    from astropy.io import fits

    ast = _one_star_asteroid(tmp_path)
    ast.settings = Settings(cutout_size=5, strip_width=21, group_count=2, residual_model="local_pointing")
    table = ast.photometry(71, 2, 2, save=False).to_table()
    assert len(table) == 3 and np.isfinite(table["flux"]).all()
    hdul = ast.lightcurve(71, 2, 2, save=False)
    assert hdul[0].header["RESMODEL"] == "local_pointing"


def test_verified_plan_reused_across_instances(tmp_path, monkeypatch):
    from tess_sso.storage import plan_cache

    ast, _ = make(tmp_path)
    visit = (71, 2, 2)
    ast.download(*visit)
    expected = ast.plan(*visit)
    again, sources = make(tmp_path)

    def unexpected(*args):
        pytest.fail("verified pixel geometry and coverage should come from disk")

    monkeypatch.setattr(plan_cache, "track_pixels", unexpected)
    monkeypatch.setattr(plan_cache.PixelStore, "missing", unexpected)
    assert again.plan(*visit) == expected
    assert again.download(*visit).already_stored == len(expected.strip_pixels)
    assert not sources


def test_verified_plan_detects_lost_block_and_different_store(tmp_path):
    ast, _ = make(tmp_path)
    visit = (71, 2, 2)
    ast.download(*visit)
    expected = ast.plan(*visit)
    store = ast.store(*visit)
    store.files()[0].unlink()
    assert ast.plan(*visit).missing
    store.pixels_dir = tmp_path / "other_drive"
    ast.store = lambda *args: store
    assert ast.plan(*visit).missing == expected.strip_pixels


def test_verified_plan_recovers_from_damaged_cache(tmp_path):
    ast, _ = make(tmp_path)
    visit = (71, 2, 2)
    expected = ast.plan(*visit)
    (ast.observation_dir(*visit) / "pixel_plan.npz").write_bytes(b"interrupted copy")
    assert ast.plan(*visit) == expected


def test_a_frame_with_a_wrong_wcs_uses_the_average_one(tmp_path):
    shifts = np.zeros(40)
    shifts[7] = 0.05  # one frame about 8.6 px off, as s0013-1-3 frame 547 is 9.2 px
    sources = {}

    def source(sector, camera, ccd):
        sources[(sector, camera, ccd)] = WcsFakeSource(nframes=40, crval_shift_deg=shifts)
        return sources[(sector, camera, ccd)]

    ast = Asteroid("95", root=tmp_path, resolve=lambda n: TARGET,
                   sectors=lambda d: [(71, 2, 2)], horizons=horizons, source=source)
    with pytest.warns(WcsDisagreementWarning, match="1 of 40 frames.*WCS_FALLBACK"):
        track = ast.track(71, 2, 2)
    assert np.flatnonzero(track.wcs_fallback).tolist() == [7]
    assert track.on_science.all()
    # the average WCS puts frame 7 where its neighbours' WCS would
    step = np.hypot(np.diff(track.row), np.diff(track.col))
    assert step.max() < 0.1


def test_several_frames_with_a_wrong_wcs_still_stop(tmp_path):
    ast, _ = make(tmp_path)
    shifts = np.zeros(6)
    shifts[2] = 0.05  # 1 of 6 frames: above WCS_FAIL_FRACTION
    ast._source = lambda sector, camera, ccd: WcsFakeSource(crval_shift_deg=shifts)
    with pytest.raises(WcsDisagreementError, match="1 of 6 frames"):
        ast.track(71, 2, 2)
