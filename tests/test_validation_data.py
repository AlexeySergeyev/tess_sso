"""The phase 10 acceptance criteria on real data: runs examples/validation.py (~0.9 GB on a fresh data folder).

Needs binaries' reference files (``BINARIES_DATA``, default ../binaries/data/asteroids).
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = Path(os.environ.get("BINARIES_DATA", ROOT.parent / "binaries" / "data" / "asteroids"))


@pytest.mark.validation
def test_validation_criteria_pass():
    if not (REFERENCE / "95" / "tess-95-s0071-2-2-tpf.fits").exists():
        pytest.skip(f"reference files not found under {REFERENCE} (set BINARIES_DATA)")
    env = dict(os.environ, BINARIES_DATA=str(REFERENCE), MPLBACKEND="Agg")
    result = subprocess.run([sys.executable, str(ROOT / "examples" / "validation.py")], env=env,
                            capture_output=True, text=True, timeout=3600)
    assert result.returncode == 0, result.stderr[-2000:]
    summary = [line for line in result.stdout.splitlines() if line.startswith("criterion ")]
    assert len(summary) == 3 and all(line.endswith("passed") for line in summary), "\n".join(summary)  # 1, 2, 4


@pytest.mark.validation
def test_demonstration_script_runs(tmp_path):
    """Runs examples/demo.py (asteroid 95, sector 71, every frame: ~2.2 GB, 10-15 minutes) in a fresh data directory."""
    script = ROOT / "examples" / "demo.py"
    env = dict(os.environ, TESS_SSO_DATA=str(tmp_path), MPLBACKEND="Agg")
    result = subprocess.run([sys.executable, str(script)], env=env, capture_output=True, text=True, timeout=5400)
    assert result.returncode == 0, result.stderr[-2000:]
    assert "single-exposure points" in result.stdout and "thirty-minute groups" in result.stdout
    assert (tmp_path / "95" / "s0071-2-2" / "frames.fits").exists()


@pytest.mark.validation
def test_injection_criteria_pass():
    """Runs examples/injection.py: criteria 3 and 5 by injection into the three observations (~20 minutes)."""
    env = dict(os.environ, MPLBACKEND="Agg")
    result = subprocess.run([sys.executable, str(ROOT / "examples" / "injection.py")], env=env,
                            capture_output=True, text=True, timeout=5400)
    assert result.returncode == 0, result.stderr[-2000:]
    summary = [line for line in result.stdout.splitlines() if line.startswith("criterion ")]
    assert len(summary) == 2 and all(line.endswith("passed") for line in summary), "\n".join(summary)  # 3, 5


@pytest.mark.validation
def test_binaries_example_runs():
    """Runs examples/binaries.py: binaries' analysis of the three asteroids with tess_sso photometry.

    Needs binaries' package (tessbin) and the network for its stage 1 (Tesscut, Miriade); ~20 minutes.
    """
    pytest.importorskip("tessbin")
    env = dict(os.environ, MPLBACKEND="Agg")
    result = subprocess.run([sys.executable, str(ROOT / "examples" / "binaries.py")], cwd=ROOT / "examples",
                            env=env, capture_output=True, text=True, timeout=5400)
    assert result.returncode == 0, result.stderr[-2000:]
    assert "period h" in result.stdout and "flagged points dropped" in result.stdout
