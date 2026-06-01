"""Path conventions for figures and serialized data produced by chapter scripts."""

from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]


def default_figure_dir() -> Path:
    """Return the repository-level directory where chapter figures are rendered."""
    return _REPO_ROOT / "output" / "figures"


def default_data_dir() -> Path:
    """Return the repository-level directory where raw NPZ/JSON data is exported."""
    return _REPO_ROOT / "output" / "data"


def ensure_dir(path: Path) -> Path:
    """Create ``path`` (and parents) if missing; return it unchanged."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
