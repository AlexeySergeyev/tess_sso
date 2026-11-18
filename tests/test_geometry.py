import numpy as np
import pytest
from astropy.table import Table

from conftest import make_wcs_frames
from tess_sso.storage.frames import FrameTable
from tess_sso.ephemeris import Ephemeris
from tess_sso.geometry import (
    BTJD0,
    Track,
    box_pixels,
    frame_pixels,
    interpolate_radec,
    make_track,
    on_science,
    track_pixels,
    wcs_disagreement,
)

RA, DEC = 72.67498, 24.41023  # asteroid 95 on 2023-10-17, on sector 71 camera 2 CCD 2


def frames(nframes=6, **kwargs):
    return FrameTable(table=Table(make_wcs_frames(nframes, **kwargs).data), provenance={})


def ephemeris(ra0=RA, ra_rate=0.1, n=48, start_tdb=BTJD0 + 3233.5):
    t = start_tdb + np.arange(n) / 24
    ra = ra0 + ra_rate * (t - t[0])
    ones = np.ones(n)
    return Ephemeris(time_utc=t, time_tdb=t, ra=ra, dec=DEC * ones, v=12 * ones,
                     r=2.8 * ones, delta=1.9 * ones, lighttime=0.01 * ones, alpha=17 * ones,
                     ra_3sigma=0.01 * ones, dec_3sigma=0.01 * ones)


def test_box_pixels_centred_and_clipped():
    assert box_pixels(100.2, 200.6, 3) == [(r, c) for r in (99, 100, 101) for c in (200, 201, 202)]
    assert box_pixels(1.0, 1.0, 3) == [(1, 1), (1, 2), (2, 1), (2, 2)]


def test_on_science():
    assert on_science(np.array([1.0, 2048.0, 2049.0]), np.array([45.0, 2092.0, 100.0])).tolist() == [True, True, False]
    assert not on_science(np.array([100.0]), np.array([44.0]))[0]


def test_interpolation_uses_spacecraft_time():
    fr = frames()
    eph = ephemeris(ra_rate=1.0)  # 1 deg/day, so a time error is easy to see
    ra, dec = interpolate_radec(eph, fr)
    t_sc = fr.time + BTJD0 - fr.barycorr
    np.testing.assert_allclose(ra, RA + (t_sc - eph.time_tdb[0]), atol=1e-9)
    np.testing.assert_allclose(dec, DEC)


def test_interpolation_across_ra_zero():
    fr = frames()
    eph = ephemeris(ra0=359.99, ra_rate=0.5)
    ra, _ = interpolate_radec(eph, fr)
    assert np.all((ra >= 0) & (ra < 360))


def test_frame_pixels_match_tesswcs_and_follow_crval():
    from tesswcs import WCS

    x, y = WCS.from_sector(71, 2, 2).world_to_pixel_values(RA, DEC)
    row, col, fallback = frame_pixels(frames(3).table, np.full(3, RA), np.full(3, DEC))
    np.testing.assert_allclose(row, y + 1, atol=1e-3)
    np.testing.assert_allclose(col, x + 1, atol=1e-3)
    assert not fallback.any()
    shifted, _, _ = frame_pixels(frames(3, crval_shift_deg=0.01).table, np.full(3, RA), np.full(3, DEC))
    assert np.abs(shifted - row).max() > 0.1


def test_frames_without_wcs_fall_back_to_average():
    row, col, fallback = frame_pixels(frames(4, nan_frames=(2,)).table, np.full(4, RA), np.full(4, DEC))
    assert fallback.tolist() == [False, False, True, False]
    assert np.isfinite(row).all()
    assert row[2] == pytest.approx(row[0], abs=1e-6)


@pytest.mark.parametrize("nan_frames", [(), (1,)])
def test_frame_pixels_nonconvergence_excludes_only_failed_positions(nan_frames):
    from tess_sso.geometry import WcsProjectionWarning

    fr = frames(3, nan_frames=nan_frames)
    ra = np.array([RA, (RA + 30) % 360, RA])
    dec = np.full(3, DEC)
    with pytest.warns(WcsProjectionWarning, match=r"1 frame.*NaN"):
        row, col, fallback = frame_pixels(fr.table, ra, dec)
    assert np.isnan(row[1]) and np.isnan(col[1])
    assert not on_science(row, col)[1]
    expected_row, expected_col, _ = frame_pixels(fr.table[[0, 2]], ra[[0, 2]], dec[[0, 2]])
    np.testing.assert_allclose(row[[0, 2]], expected_row)
    np.testing.assert_allclose(col[[0, 2]], expected_col)
    assert fallback.tolist() == [False, bool(nan_frames), False]


def test_frame_pixels_never_uses_unconverged_best_solution(monkeypatch):
    from astropy.wcs import NoConvergence, WCS
    from tess_sso.geometry import WcsProjectionWarning

    def fail(self, *args, **kwargs):
        raise NoConvergence("diverging", best_solution=np.array([[100.0, 100.0]]),
                            accuracy=np.ones((1, 2)), niter=20, divergent=np.array([0]))

    fr = frames(3)
    monkeypatch.setattr(WCS, "all_world2pix", fail)
    with pytest.warns(WcsProjectionWarning, match=r"3 frame.*NaN") as caught:
        row, col, fallback = frame_pixels(fr.table, np.full(3, RA), np.full(3, DEC))
    assert len(caught) == 1
    assert np.isnan(row).all() and np.isnan(col).all()
    assert not on_science(row, col).any()
    assert not fallback.any()


def rereferenced(table, rows, shift=24.0):
    """``table`` with ``rows`` carrying the same sky-to-pixel mapping written about a reference pixel
    ``shift`` px further on both axes, as sector 91's first orbit is against its second."""
    import warnings

    from astropy.coordinates import SkyCoord
    from astropy.wcs.utils import fit_wcs_from_points
    from tesswcs import WCS

    wcs = WCS.from_sector(71, 2, 2)
    y, x = np.mgrid[0:2048:64, 44:2092:64].reshape(2, -1).astype(float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ra, dec = wcs.all_pix2world(x, y, 0)
        reference = wcs.all_pix2world([wcs.wcs.crpix - 1 + shift], 0)[0]
        fit = fit_wcs_from_points((x, y), SkyCoord(ra, dec, unit="deg"),
                                  proj_point=SkyCoord(*reference, unit="deg"), sip_degree=wcs.sip.a_order)
    table = table.copy()
    rows = list(rows)
    for k in (1, 2):
        table[f"CRPIX{k}"][rows] = fit.wcs.crpix[k - 1]
        table[f"CRVAL{k}"][rows] = fit.wcs.crval[k - 1]
    for (i, j), value in np.ndenumerate(fit.wcs.cd):
        table[f"CD{i + 1}_{j + 1}"][rows] = value
    for name, matrix in (("A", fit.sip.a), ("B", fit.sip.b)):
        for (p, q), value in np.ndenumerate(matrix):
            if 2 <= p + q <= fit.sip.a_order:
                table[f"{name}_{p}_{q}"][rows] = value
    return table


def test_frames_with_another_reference_pixel_keep_their_own():
    """Most frames on another CRPIX must not move the rest: each frame uses its own reference pixel and the
    SIP of the frames that share it."""
    from tesswcs import WCS

    from tess_sso.geometry import frame_wcs, sky_to_pixels

    x, y = WCS.from_sector(71, 2, 2).world_to_pixel_values(RA, DEC)
    table = rereferenced(frames(5).table, rows=(2, 3, 4))
    row, col, fallback = frame_pixels(table, np.full(5, RA), np.full(5, DEC))
    assert not fallback.any()
    np.testing.assert_allclose(row, y + 1, atol=0.1)
    np.testing.assert_allclose(col, x + 1, atol=0.1)
    for frame in (0, 3):
        fx, fy = frame_wcs(table, frame).all_world2pix([[RA, DEC]], 0)[0]
        assert (fy + 1, fx + 1) == pytest.approx((y + 1, x + 1), abs=0.1)
    rows, cols = sky_to_pixels(table, np.array([RA]), np.array([DEC]))
    assert (rows[0], cols[0]) == pytest.approx((y + 1, x + 1), abs=0.1)


@pytest.mark.parametrize("unused_value", [np.nan, np.inf, 1e20])
def test_average_wcs_ignores_coefficients_above_declared_order(unused_value):
    from tess_sso.geometry import average_wcs

    table = frames(3).table
    expected = average_wcs(table)
    order = int(table["A_ORDER"][0])
    for prefix in ("A", "B", "AP", "BP"):
        table[f"{prefix}_ORDER"] = np.full(3, order)
        table[f"{prefix}_{order + 1}_0"] = np.full(3, unused_value)
    wcs = average_wcs(table)
    assert (wcs.sip.a_order, wcs.sip.b_order, wcs.sip.ap_order, wcs.sip.bp_order) == (order,) * 4
    np.testing.assert_allclose(wcs.all_world2pix([[RA, DEC]], 0), expected.all_world2pix([[RA, DEC]], 0))


@pytest.mark.parametrize("higher_order_frames", [(2, 3, 4), (4,)])
def test_mixed_sip_orders_keep_each_groups_coefficients(higher_order_frames):
    from tess_sso.geometry import average_wcs, frame_wcs, sky_to_pixels

    fr = frames(6, nan_frames=(5,))
    table = fr.table
    expected_row, expected_col, _ = frame_pixels(table, np.full(6, RA), np.full(6, DEC))
    order = int(table["A_ORDER"][0])
    higher = list(higher_order_frames)
    coefficient = 1e-20
    for prefix in ("A", "B", "AP", "BP"):
        table[f"{prefix}_ORDER"] = np.full(6, order)
        table[f"{prefix}_ORDER"][higher] = order + 1
        table[f"{prefix}_{order + 1}_0"] = np.full(6, np.nan)
        table[f"{prefix}_{order + 1}_0"][higher] = coefficient
    row, col, fallback = frame_pixels(table, np.full(6, RA), np.full(6, DEC))
    assert np.isfinite(row).all() and np.isfinite(col).all()
    assert fallback.tolist() == [False] * 5 + [True]
    np.testing.assert_allclose(row[0], expected_row[0])
    np.testing.assert_allclose(col[0], expected_col[0])
    for i in range(5):
        wcs = frame_wcs(table, i)
        assert wcs.sip.a_order == order + (i in higher)
        if i in higher:
            for matrix in (wcs.sip.a, wcs.sip.b, wcs.sip.ap, wcs.sip.bp):
                assert matrix[order + 1, 0] == pytest.approx(coefficient, rel=1e-12, abs=0)
        x, y = wcs.all_world2pix([[RA, DEC]], 0)[0]
        assert (row[i], col[i]) == pytest.approx((y + 1, x + 1))
    assert average_wcs(table).sip.a_order == order + (len(higher) > 2)
    rows, cols = sky_to_pixels(table, np.array([RA]), np.array([DEC]))
    assert np.isfinite(rows).all() and np.isfinite(cols).all()


def test_track_and_pixels():
    fr = frames()
    track = make_track(ephemeris(), fr)
    assert len(track) == 6 and track.on_science.all()
    assert wcs_disagreement(track, 71, 2, 2) < 1e-3
    pixels = track_pixels(track, 3)
    assert set(box_pixels(track.row[0], track.col[0], 3)) <= set(pixels)


@pytest.mark.parametrize("science_offset", [0.0, 8.0])
def test_wcs_disagreement_ignores_off_science_positions(science_offset):
    from tesswcs import WCS

    x, y = WCS.from_sector(71, 2, 2).world_to_pixel_values(RA, DEC)
    # A large off-detector discrepancy must not hide a real on-detector offset.
    row = np.array([y + 1 + science_offset, 5735.0])
    col = np.full(2, x + 1)
    track = Track(time=np.arange(2), ra=np.full(2, RA), dec=np.full(2, DEC),
                  row=row, col=col, wcs_fallback=np.zeros(2, bool), on_science=on_science(row, col))
    assert track.on_science.tolist() == [True, False]
    assert wcs_disagreement(track, 71, 2, 2) == pytest.approx(science_offset, abs=1e-6)


def test_wcs_disagreement_without_science_positions():
    track = make_track(ephemeris(), frames())
    track = track.within(track.time[-1] + 1, track.time[-1] + 2)
    assert not track.on_science.any()
    assert np.isnan(wcs_disagreement(track, 71, 2, 2))


def test_track_pixels_union_skips_off_science():
    n = 6
    track = Track(time=np.arange(n), ra=np.zeros(n), dec=np.zeros(n),
                  row=100.0 + np.arange(n), col=np.full(n, 200.0),
                  wcs_fallback=np.zeros(n, bool), on_science=np.array([True] * 5 + [False]))
    assert len(track_pixels(track, 3)) == 7 * 3  # rows 99-105, columns 199-201


def test_within_keeps_frames_but_limits_use():
    track = make_track(ephemeris(), frames())
    part = track.within(track.time[1], track.time[3])
    assert len(part) == len(track)
    assert part.on_science.tolist() == [False, True, True, True, False, False]


def test_sky_to_pixels_matches_frame_pixels():
    from tess_sso.geometry import sky_to_pixels

    fr = frames(3)
    rows, cols = sky_to_pixels(fr.table, np.array([RA]), np.array([DEC]))
    row, col, _ = frame_pixels(fr.table, np.array([RA]), np.array([DEC]))
    assert rows[0] == pytest.approx(row[0], abs=1e-6) and cols[0] == pytest.approx(col[0], abs=1e-6)


def test_geometry_is_interpolated_at_spacecraft_time():
    from tess_sso.geometry import interpolate_geometry

    fr = frames()
    eph = ephemeris()
    eph = Ephemeris(**{**eph.__dict__, "delta": 1.9 + 0.1 * (eph.time_tdb - eph.time_tdb[0])})
    geo = interpolate_geometry(eph, fr)
    t_sc = fr.time + BTJD0 - fr.barycorr
    np.testing.assert_allclose(geo.delta, 1.9 + 0.1 * (t_sc - eph.time_tdb[0]), atol=1e-9)
    np.testing.assert_allclose(geo.r, 2.8)
    np.testing.assert_allclose(geo.alpha, 17.0)


def test_pixel_offset_to_sky_matches_the_frame_wcs():
    from tess_sso.geometry import pixel_offset_to_sky

    fr = frames(3)
    row, col, _ = frame_pixels(fr.table, np.array([RA]), np.array([DEC]))
    ra, dec = pixel_offset_to_sky(fr.table, 1, RA, DEC, 0.3, -0.2)
    back_row, back_col, _ = frame_pixels(fr.table, np.array([ra]), np.array([dec]))
    assert back_row[0] - row[0] == pytest.approx(0.3, abs=1e-4)
    assert back_col[0] - col[0] == pytest.approx(-0.2, abs=1e-4)
