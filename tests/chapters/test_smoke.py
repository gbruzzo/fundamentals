"""Headless smoke tests for every chapter orchestrator script.

Each script is launched via ``subprocess`` with ``--save`` so we exercise the
exact CLI path a user would. ``MPLBACKEND=Agg`` keeps the GIF/PNG renderers
off any display.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]   # tests/chapters/ → repo root
CHAPTER_DIRS = {
    1: REPO_ROOT / "chapters" / "chapter_01",
    2: REPO_ROOT / "chapters" / "chapter_02",
    3: REPO_ROOT / "chapters" / "chapter_03",
    4: REPO_ROOT / "chapters" / "chapter_04",
    5: REPO_ROOT / "chapters" / "chapter_05",
    6: REPO_ROOT / "chapters" / "chapter_06",
    7: REPO_ROOT / "chapters" / "chapter_07",
    8: REPO_ROOT / "chapters" / "chapter_08",
    9: REPO_ROOT / "chapters" / "chapter_09",
    10: REPO_ROOT / "chapters" / "chapter_10",
}


def _is_interactive(p: Path) -> bool:
    """Skip GUI-only scripts in headless tests."""
    return "interactive" in p.name


def _chapter_number(script: Path) -> int:
    """Infer the numeric chapter directory that owns ``script``."""
    return int(script.parent.name.removeprefix("chapter_"))


def _fresh_data_pairs(data_dir: Path, started_ns: int) -> list[tuple[Path, Path]]:
    """Return JSON/NPZ sidecars written after ``started_ns``."""
    pairs: list[tuple[Path, Path]] = []
    for json_path in data_dir.glob("*.json"):
        npz_path = json_path.with_suffix(".npz")
        if not npz_path.exists():
            continue
        if json_path.stat().st_mtime_ns >= started_ns or npz_path.stat().st_mtime_ns >= started_ns:
            pairs.append((npz_path, json_path))
    return pairs


def _run(
    script: Path,
    *extra: str,
    env_extra: dict | None = None,
    timeout: int = 120,
) -> None:
    """Run one chapter script and require fresh figure-adjacent raw-data output."""
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["PYTHONWARNINGS"] = "error"
    env["PYTHONPATH"] = str(REPO_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    if env_extra:
        env.update(env_extra)
    chapter = _chapter_number(script)
    data_dir = REPO_ROOT / "output" / "data" / f"chapter_{chapter:02d}"
    started_ns = time.time_ns()
    cmd = [sys.executable, str(script), "--save", *extra]
    result = subprocess.run(cmd, capture_output=True, text=True, env=env,
                            timeout=timeout)
    if result.returncode != 0:
        raise AssertionError(
            f"Script {script.name} failed.\nSTDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
    pairs = _fresh_data_pairs(data_dir, started_ns)
    assert pairs, f"Script {script.name} did not write a fresh NPZ+JSON data export"


CHAPTER_1_SCRIPTS = sorted(CHAPTER_DIRS[1].glob("0*.py"))
CHAPTER_2_EXAMPLES = sorted(p for p in CHAPTER_DIRS[2].glob("example_*.py")
                            if not _is_interactive(p))
CHAPTER_2_AUX = [CHAPTER_DIRS[2] / "visualize_generative_model.py"]
CHAPTER_2_ANIMATIONS = sorted(CHAPTER_DIRS[2].glob("animation_*.py"))
CHAPTER_3_EXAMPLES = sorted(CHAPTER_DIRS[3].glob("example_*.py"))
CHAPTER_3_ANIMATIONS = sorted(CHAPTER_DIRS[3].glob("animation_*.py"))
CHAPTER_3_VISUALIZATIONS = sorted(
    p for p in CHAPTER_DIRS[3].glob("visualize_*.py")
    if not _is_interactive(p)
)
CHAPTER_4_EXAMPLES = sorted(CHAPTER_DIRS[4].glob("example_*.py"))
CHAPTER_4_ANIMATIONS = sorted(CHAPTER_DIRS[4].glob("animation_*.py"))
CHAPTER_4_VISUALIZATIONS = sorted(
    p for p in CHAPTER_DIRS[4].glob("visualize_*.py")
    if not _is_interactive(p)
)
CHAPTER_5_EXAMPLES = sorted(CHAPTER_DIRS[5].glob("example_*.py"))
CHAPTER_5_ANIMATIONS = sorted(CHAPTER_DIRS[5].glob("animation_*.py"))
CHAPTER_6_EXAMPLES = sorted(CHAPTER_DIRS[6].glob("example_*.py"))
CHAPTER_7_EXAMPLES = sorted(CHAPTER_DIRS[7].glob("example_*.py"))
CHAPTER_8_EXAMPLES = sorted(CHAPTER_DIRS[8].glob("example_*.py"))
CHAPTER_8_ANIMATIONS = sorted(CHAPTER_DIRS[8].glob("animation_*.py"))
CHAPTER_8_VISUALIZATIONS = sorted(
    p for p in CHAPTER_DIRS[8].glob("visualize_*.py")
    if not _is_interactive(p)
)
CHAPTER_9_EXAMPLES = sorted(CHAPTER_DIRS[9].glob("example_*.py"))
CHAPTER_9_ANIMATIONS = sorted(CHAPTER_DIRS[9].glob("animation_*.py"))
CHAPTER_10_EXAMPLES = sorted(CHAPTER_DIRS[10].glob("example_*.py"))
CHAPTER_10_ANIMATIONS = sorted(CHAPTER_DIRS[10].glob("animation_*.py"))
CHAPTER_10_VISUALIZATIONS = sorted(CHAPTER_DIRS[10].glob("visualize_*.py"))


@pytest.mark.parametrize("script", CHAPTER_1_SCRIPTS,
                         ids=[s.name for s in CHAPTER_1_SCRIPTS])
def test_chapter_1_scripts_run(script: Path) -> None:
    _run(script)


@pytest.mark.parametrize("script", CHAPTER_2_EXAMPLES,
                         ids=[s.name for s in CHAPTER_2_EXAMPLES])
def test_chapter_2_scripts_run(script: Path) -> None:
    _run(script)


@pytest.mark.parametrize("script", CHAPTER_2_AUX,
                         ids=[s.name for s in CHAPTER_2_AUX])
def test_chapter_2_auxiliary_scripts(script: Path) -> None:
    _run(script)


@pytest.mark.parametrize("script", CHAPTER_2_ANIMATIONS,
                         ids=[s.name for s in CHAPTER_2_ANIMATIONS])
def test_chapter_2_animations(script: Path) -> None:
    _run(script, timeout=240)


@pytest.mark.parametrize("script", CHAPTER_3_EXAMPLES,
                         ids=[s.name for s in CHAPTER_3_EXAMPLES])
def test_chapter_3_scripts_run(script: Path) -> None:
    _run(script)


@pytest.mark.parametrize("script", CHAPTER_3_ANIMATIONS,
                         ids=[s.name for s in CHAPTER_3_ANIMATIONS])
def test_chapter_3_animations(script: Path) -> None:
    _run(script, timeout=240)


@pytest.mark.parametrize("script", CHAPTER_3_VISUALIZATIONS,
                         ids=[s.name for s in CHAPTER_3_VISUALIZATIONS])
def test_chapter_3_visualizations(script: Path) -> None:
    # Diagnostic visualizations run a small Monte Carlo loop; allow some headroom.
    _run(script, timeout=240)


@pytest.mark.parametrize("script", CHAPTER_4_EXAMPLES,
                         ids=[s.name for s in CHAPTER_4_EXAMPLES])
def test_chapter_4_scripts_run(script: Path) -> None:
    # VFE grid sweeps (contour surfaces) are integration-heavy; allow headroom.
    _run(script, timeout=240)


@pytest.mark.parametrize("script", CHAPTER_4_ANIMATIONS,
                         ids=[s.name for s in CHAPTER_4_ANIMATIONS])
def test_chapter_4_animations(script: Path) -> None:
    _run(script, timeout=240)


@pytest.mark.parametrize("script", CHAPTER_4_VISUALIZATIONS,
                         ids=[s.name for s in CHAPTER_4_VISUALIZATIONS])
def test_chapter_4_visualizations(script: Path) -> None:
    _run(script, timeout=240)


@pytest.mark.parametrize("script", CHAPTER_5_EXAMPLES,
                         ids=[s.name for s in CHAPTER_5_EXAMPLES])
def test_chapter_5_scripts_run(script: Path) -> None:
    # Predictive-coding recognition dynamics + grid sweeps; allow headroom.
    _run(script, timeout=240)


@pytest.mark.parametrize("script", CHAPTER_5_ANIMATIONS,
                         ids=[s.name for s in CHAPTER_5_ANIMATIONS])
def test_chapter_5_animations(script: Path) -> None:
    # Recognition-descent + hierarchical-convergence GIFs (rendering is slower).
    _run(script, timeout=240)


@pytest.mark.parametrize("script", CHAPTER_6_EXAMPLES,
                         ids=[s.name for s in CHAPTER_6_EXAMPLES])
def test_chapter_6_scripts_run(script: Path) -> None:
    # Generalized filtering for perception (online dynamic simulation).
    _run(script, timeout=240)


@pytest.mark.parametrize("script", CHAPTER_7_EXAMPLES,
                         ids=[s.name for s in CHAPTER_7_EXAMPLES])
def test_chapter_7_scripts_run(script: Path) -> None:
    # Active inference: the coupled action-perception loop.
    _run(script, timeout=240)


@pytest.mark.parametrize("script", CHAPTER_8_EXAMPLES,
                         ids=[s.name for s in CHAPTER_8_EXAMPLES])
def test_chapter_8_scripts_run(script: Path) -> None:
    # Learning, attention, and hierarchy in continuous state spaces.
    _run(script, timeout=240)


@pytest.mark.parametrize("script", CHAPTER_8_ANIMATIONS,
                         ids=[s.name for s in CHAPTER_8_ANIMATIONS])
def test_chapter_8_animations(script: Path) -> None:
    _run(script, timeout=240)


@pytest.mark.parametrize("script", CHAPTER_8_VISUALIZATIONS,
                         ids=[s.name for s in CHAPTER_8_VISUALIZATIONS])
def test_chapter_8_visualizations(script: Path) -> None:
    _run(script, timeout=240)


@pytest.mark.parametrize("script", CHAPTER_9_EXAMPLES,
                         ids=[s.name for s in CHAPTER_9_EXAMPLES])
def test_chapter_9_scripts_run(script: Path) -> None:
    # Discrete POMDP active inference (categorical state inference).
    _run(script, timeout=240)


@pytest.mark.parametrize("script", CHAPTER_9_ANIMATIONS,
                         ids=[s.name for s in CHAPTER_9_ANIMATIONS])
def test_chapter_9_animations(script: Path) -> None:
    # Dynamic categorical filtering GIFs (§9.2).
    _run(script, timeout=240)


@pytest.mark.parametrize("script", CHAPTER_10_EXAMPLES,
                         ids=[s.name for s in CHAPTER_10_EXAMPLES])
def test_chapter_10_scripts_run(script: Path) -> None:
    # Learning the POMDP parameters (Dirichlet A/B/D + novelty, Alg. 10.1.1).
    _run(script, timeout=240)


@pytest.mark.parametrize("script", CHAPTER_10_ANIMATIONS,
                         ids=[s.name for s in CHAPTER_10_ANIMATIONS])
def test_chapter_10_animations(script: Path) -> None:
    # Dirichlet learning, precision sweep, and two-armed bandit animations (GIFs).
    _run(script, timeout=240)


@pytest.mark.parametrize("script", CHAPTER_10_VISUALIZATIONS,
                         ids=[s.name for s in CHAPTER_10_VISUALIZATIONS])
def test_chapter_10_visualizations(script: Path) -> None:
    # Factorial likelihood structure heatmaps (§10.3, Fig 10.3.4).
    _run(script, timeout=240)
