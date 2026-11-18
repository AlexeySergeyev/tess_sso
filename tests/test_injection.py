import numpy as np
import pytest
from astropy.io import fits

from tess_sso import validation as V
from tess_sso.injection import BOX, InjectedSignal, InjectedStore, Injection, dips, injection_for, sinusoid
from tess_sso.settings import Settings


class GaussPrf:
    """A round Gaussian PRF (sigma 1 px) that ignores the motion."""

    def smeared(self, row0, col0, row1, col1, origin, shape):
        r = origin[0] + np.arange(shape[0])[:, None] - 0.5 * (row0 + row1)
        c = origin[1] + np.arange(shape[1])[None, :] - 0.5 * (col0 + col1)
        image = np.exp(-0.5 * (r**2 + c**2))
        return image / (2 * np.pi)


def test_factors():
    t = np.array([0.0, 0.25, 0.5, 1.0])
    assert sinusoid(0.05, 1.0)(t) == pytest.approx([1.0, 1.05, 1.0, 1.0])
    assert dips([(0.2, 0.6)], 0.02)(t) == pytest.approx([1.0, 0.98, 0.98, 1.0])


def test_injection_adds_the_scaled_prf_at_the_shifted_position():
    times = np.array([0.0, 0.25, 0.5])
    flux = np.array([1000.0, 1000.0, np.nan])  # no flux measured: nothing injected
    ends = (np.full(3, 50.0), np.full(3, 60.0), np.full(3, 50.0), np.full(3, 60.0))
    shifts = np.array([[0.0, 0.0], [1.0, -1.0], [0.0, 0.0]])
    inj = Injection.build(sinusoid(0.1, 1.0), times, flux, ends, shifts, GaussPrf())
    assert set(inj.images) == {1}  # frame 0: factor 1; frame 2: no flux
    pixels = [(r, c) for r in range(40, 62) for c in range(50, 72)]
    out = inj.add(np.zeros((3, len(pixels))), pixels, first_frame=0)
    assert out[0].sum() == 0 and out[2].sum() == 0
    assert out[1].sum() == pytest.approx(100.0, rel=1e-3)  # (A - 1) x F, the PRF sums to 1
    assert pixels[int(np.argmax(out[1]))] == (51, 59)
    # the frame index is absolute: the same image lands in row 0 when reading from frame 1
    assert inj.add(np.zeros((1, len(pixels))), pixels, first_frame=1)[0] == pytest.approx(out[1])


def test_injected_store_adds_to_the_pixels_only():
    class Store:
        sector = 71

        def read_pixels(self, pixels, frames=None):
            n = 3 if frames is None else frames.stop - frames.start
            return np.ones((n, len(pixels)), np.float32), np.full((n, len(pixels)), 2.0, np.float32)

    image = np.zeros((BOX, BOX))
    image[BOX // 2, BOX // 2] = 5.0
    store = InjectedStore(Store(), Injection({2: ((0, 0), image)}))
    flux, err = store.read_pixels([(7, 7), (0, 0)], frames=slice(1, 3))
    assert flux.dtype == np.float32
    assert flux.tolist() == [[1.0, 1.0], [6.0, 1.0]]
    assert err.tolist() == [[2.0, 2.0], [2.0, 2.0]]
    assert store.sector == 71


def test_injected_signal():
    signal = InjectedSignal(crossing=((1.0, 1.1),), isolated=((2.0, 2.1),))
    t = np.array([0.0, 0.0625, 1.0, 2.0625])
    assert signal(t) == pytest.approx([1.0, 1.05, 0.98, 0.98 * 1.05])
    assert signal.windows == [(1.0, 1.1), (2.0, 2.1)]
    assert signal.label == "sin 0.05 6 h x 2 dips 0.02"


def test_injected_signal_places_dips_at_the_crossings_and_between_them():
    t = 3235.0 + np.arange(0, 6.0, 0.01)
    star = np.where(np.abs(t - 3237.0) < 0.02, 100.0, 1.0)
    star[np.argmin(np.abs(t - 3237.01))] = 300.0  # the brightest point
    lc = fits.BinTableHDU.from_columns([
        fits.Column(name="TIME", format="D", array=t),
        fits.Column(name="STAR_FLUX", format="D", array=star),
        fits.Column(name="SEGMENT", format="I", array=np.zeros(t.size, int)),
    ], name="LIGHTCURVE_PRF")
    extras = fits.BinTableHDU.from_columns([fits.Column(name="ORIGINAL_TIME", format="D", array=t - 0.001)], name="EXTRAS")
    signal = InjectedSignal.for_lightcurve(fits.HDUList([fits.PrimaryHDU(), lc, extras]))
    assert len(signal.crossing) == 1
    assert np.mean(signal.crossing[0]) == pytest.approx(3237.01 - 0.001, abs=1e-6)
    assert signal.crossing[0][1] - signal.crossing[0][0] == pytest.approx(1 / 24)
    # every 1.5 days from 0.5 d after the start, except the one within 0.1 d of the crossing window
    assert [round(float(np.mean(w)), 3) for w in signal.isolated] == [3235.499, 3238.499, 3239.999]


def test_injected_pipeline_writes_apart_and_recovers_the_factor(tmp_path):
    from test_output import SETTINGS, pipeline

    ast = pipeline(tmp_path)
    ast.lightcurve(71, 2, 2)
    with pytest.raises(ValueError, match="single-exposure"):  # groups of two
        injection_for(ast, 71, 2, 2, sinusoid(0.1, 1.0), ast.lightcurve_path(71, 2, 2))
    ast.settings = Settings(cutout_size=SETTINGS.cutout_size, strip_width=SETTINGS.strip_width)
    plain = ast.lightcurve(71, 2, 2)
    plain_path = ast.lightcurve_path(71, 2, 2)
    # the fake pixels are ~7e5 (float32 steps of 0.06) and its asteroid 0.15 e-/s: a huge factor to be seen
    factor = sinusoid(1000.0, 6 * 200 / 86400)
    ast.injection = injection_for(ast, 71, 2, 2, factor, plain_path, label="test sine")
    assert ast.results_dir(71, 2, 2) == plain_path.parent / "injection"
    injected = ast.lightcurve(71, 2, 2)
    assert ast.lightcurve_path(71, 2, 2).exists() and injected[0].header["INJECTED"] == "test sine"
    assert "INJECTED" not in fits.getheader(plain_path)
    ratio = injected[1].data["FLUX"] / plain[1].data["FLUX"]
    expected = V.expected_factor(injected, factor)
    assert np.ptp(expected) > 100
    measured = np.isfinite(ratio)  # a frame without plain flux gets nothing injected
    assert measured.sum() >= 5
    assert (ratio[measured] - 1) / (expected[measured] - 1) == pytest.approx(1.0, abs=0.02)
