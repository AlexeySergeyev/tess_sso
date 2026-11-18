import pytest

from tess_sso.settings import Settings


def test_defaults_are_valid():
    s = Settings()
    assert (s.cutout_size, s.strip_width, s.ephemeris_step) == (11, 31, "1h")


@pytest.mark.parametrize("kwargs", [{"cutout_size": 10}, {"strip_width": 0}, {"cutout_size": 11, "strip_width": 9}])
def test_invalid_settings_rejected(kwargs):
    with pytest.raises(ValueError):
        Settings(**kwargs)


def test_hash_is_stable_and_sensitive():
    assert Settings().hash() == Settings().hash()
    assert len(Settings().hash()) == 12 and int(Settings().hash(), 16) >= 0
    assert Settings(strip_width=33).hash() != Settings().hash()
    assert Settings(ephemeris_step="30m").hash() != Settings().hash()


def test_grouping_settings():
    assert Settings(group_minutes=30).group_minutes == 30
    assert Settings(group_count=3).group_count == 3
    for bad in ({"group_minutes": 30, "group_count": 3}, {"group_minutes": 0}, {"group_count": 0}):
        with pytest.raises(ValueError):
            Settings(**bad)


def test_static_correction_is_on_by_default_and_part_of_the_hash():
    assert Settings().static_correction is True
    assert Settings(static_correction=False).hash() != Settings().hash()


def test_prf_position_defaults_to_segment_and_is_checked():
    assert Settings().prf_position == "segment"
    assert Settings(prf_position="group").hash() != Settings().hash()
    with pytest.raises(ValueError):
        Settings(prf_position="measured")


def test_local_residual_and_background_are_on_by_default_and_checked():
    s = Settings()
    assert s.residual_model == "local_pointing" and s.local_background is True
    assert Settings(residual_model="static").hash() != s.hash()
    assert Settings(local_background=False).hash() != s.hash()
    with pytest.raises(ValueError, match="residual_model"):
        Settings(residual_model="spline")


def test_the_residual_follows_the_pointing_by_default():
    assert Settings().residual_model == "local_pointing"
    assert Settings(residual_model="local").hash() != Settings().hash()
