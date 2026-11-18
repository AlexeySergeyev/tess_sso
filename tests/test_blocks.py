import pytest

from tess_sso.storage.blocks import block_start, by_block, check_pixels, file_name, parse_file_name, runs_for


def test_block_start_is_one_based():
    assert [block_start(r) for r in (1, 16, 17, 705, 720, 721)] == [1, 1, 17, 705, 705, 721]


def test_by_block_groups_and_sorts():
    groups = by_block([(721, 5), (705, 9), (720, 3), (705, 9)])
    assert groups == {705: [(705, 9), (720, 3)], 721: [(721, 5)]}


def test_runs_cover_exactly_the_pixels():
    pixels = [(10, 5), (10, 6), (10, 7), (10, 9), (11, 6)]
    assert runs_for(pixels) == [
        {"row": 10, "start_column": 5, "ncolumns": 3},
        {"row": 10, "start_column": 9, "ncolumns": 1},
        {"row": 11, "start_column": 6, "ncolumns": 1},
    ]


def test_file_names():
    assert file_name(705, 1) == "r0705-0720_1.fits"
    assert parse_file_name("r0705-0720_12.fits") == (705, 12)
    assert parse_file_name("r0705-0720_1.fits.part") is None
    assert parse_file_name("frames.fits") is None


@pytest.mark.parametrize("pixel", [(0, 5), (2079, 5), (5, 0), (5, 2137)])
def test_pixels_outside_the_ccd_rejected(pixel):
    with pytest.raises(ValueError, match="outside the CCD"):
        check_pixels([pixel])
