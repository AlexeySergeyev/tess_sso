import numpy as np
import pytest

from tess_sso.segments import (
    check_segments,
    find_segments,
    gap_threshold,
    load_orbit_times,
    parse_orbit_times,
    segment_ids,
    user_segments,
)

CADENCE = 200 / 86400


def times_with_gaps(gaps_after=(99, 199), gap=0.213, n=300):
    t = np.arange(n) * CADENCE
    for i in gaps_after:
        t[i + 1 :] += gap
    return 3234.03 + t


def test_threshold():
    assert gap_threshold(np.arange(10) * CADENCE) == pytest.approx(0.1)
    assert gap_threshold(np.arange(10) * 0.05) == pytest.approx(0.15)


def test_split_at_real_gaps_only():
    t = times_with_gaps()
    t[150:] += 2 * CADENCE  # a two-frame hole (e.g. dropped frames) is not a boundary
    segs = find_segments(t)
    assert [(s.first, s.last) for s in segs] == [(0, 99), (100, 199), (200, 299)]
    assert all(s.source == "gap" for s in segs)
    assert segs[1].start == pytest.approx(t[100]) and segs[1].n_frames == 100


def test_segment_ids():
    segs = find_segments(times_with_gaps())
    ids = segment_ids(segs, 300)
    assert ids[0] == 0 and ids[150] == 1 and ids[-1] == 2 and (ids >= 0).all()


def test_unsorted_times_rejected():
    with pytest.raises(ValueError):
        find_segments(np.array([1.0, 0.5, 2.0]))


def test_user_segments():
    t = times_with_gaps(gaps_after=())
    mid = t[149]
    segs = user_segments(t, [(t[0], mid), (mid + CADENCE / 2, t[-1])])
    assert [(s.first, s.last, s.source) for s in segs] == [(0, 149, "user"), (150, 299, "user")]
    with pytest.raises(ValueError, match="exactly one"):
        user_segments(t, [(t[0], mid)])


ORBITS = """Sector,Orbit,Start of Orbit,End of Orbit
71,149,2023-10-16 12:50:00,2023-10-23 00:00:00
71,149,2023-10-23 05:00:00,2023-10-29 12:20:00
72,151,2023-11-11 16:00:00,2023-11-18 00:00:00
"""


def test_parse_converts_utc_to_spacecraft_btjd():
    table = parse_orbit_times(ORBITS)
    assert table["start"][0] == pytest.approx(3234.0347 + 69.184 / 86400, abs=1e-4)


def test_check_segments():
    orbits = parse_orbit_times(ORBITS)
    orbits = orbits[orbits["sector"] == 71]
    t = np.concatenate([np.arange(orbits["start"][0] + 0.001, orbits["stop"][0] - 0.005, CADENCE),
                        np.arange(orbits["start"][1] + 0.001, orbits["stop"][1] - 0.005, CADENCE)])
    good = check_segments(find_segments(t), orbits)
    assert good.ok, good.message
    bad = check_segments(find_segments(t[t < orbits["stop"][0]]), orbits)
    assert not bad.ok and "1 segment(s) in the data, 2 in the orbit table" in bad.message


def test_orbit_times_cached_and_refreshed_for_new_sector(tmp_path):
    calls = []

    def fetch():
        calls.append(1)
        return ORBITS

    assert len(load_orbit_times(71, tmp_path, fetch)) == 2
    assert len(load_orbit_times(72, tmp_path, fetch)) == 1
    assert len(calls) == 1
    assert len(load_orbit_times(99, tmp_path, fetch)) == 0
    assert len(calls) == 2  # unknown sector triggers one refresh
