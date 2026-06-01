"""Contracts for chapter raw-data sidecars and their validator."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from active_inference.utils import (
    data_paths_for_figure,
    extract_figure_data,
    save_chapter_data,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_save_chapter_data_writes_npz_and_json_manifest(tmp_path: Path) -> None:
    """Persist numeric arrays plus machine-readable provenance and statistics."""
    npz_path, json_path = save_chapter_data(
        8,
        "example_8_1_learning_attention",
        arrays={
            "time": np.linspace(0.0, 1.0, 5),
            "beliefs": np.eye(3),
        },
        metadata={
            "script": "example_8_1_learning_attention.py",
            "args": {"save": True, "seed": 7},
            "seed": 7,
            "summary": {"final_error": 0.125},
        },
        figures=[Path("output/figures/chapter_08/example_8_1_learning_attention.png")],
        root=tmp_path,
    )

    assert npz_path == tmp_path / "chapter_08" / "example_8_1_learning_attention.npz"
    assert json_path == tmp_path / "chapter_08" / "example_8_1_learning_attention.json"
    assert npz_path.exists()
    assert json_path.exists()

    with np.load(npz_path) as data:
        assert set(data.files) == {"time", "beliefs"}
        np.testing.assert_allclose(data["time"], np.linspace(0.0, 1.0, 5))

    manifest = json.loads(json_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["chapter"] == 8
    assert manifest["script"] == "example_8_1_learning_attention.py"
    assert manifest["seed"] == 7
    assert manifest["figures"] == [
        "output/figures/chapter_08/example_8_1_learning_attention.png"
    ]
    assert manifest["arrays"]["time"]["shape"] == [5]
    assert manifest["arrays"]["beliefs"]["dtype"].startswith("float")
    assert manifest["summary"]["time"]["finite_fraction"] == pytest.approx(1.0)
    assert manifest["metadata"]["summary"]["final_error"] == pytest.approx(0.125)


@pytest.mark.parametrize(
    "arrays",
    [
        {"empty": np.array([])},
        {"nonfinite": np.array([0.0, np.inf])},
        {"object": np.array([{"bad": "shape"}], dtype=object)},
        {"string": np.array(["not", "numeric"])},
    ],
    ids=["empty", "nonfinite", "object", "string"],
)
def test_save_chapter_data_rejects_invalid_arrays(
    tmp_path: Path,
    arrays: dict[str, np.ndarray],
) -> None:
    """Reject arrays that cannot support reproducible numeric reconstruction."""
    with pytest.raises(ValueError):
        save_chapter_data(1, "bad_export", arrays=arrays, root=tmp_path)


def test_extract_figure_data_captures_plot_reconstruction_arrays() -> None:
    """Extract line, image, scatter, and figure geometry arrays from a figure."""
    fig, ax = plt.subplots(figsize=(4, 3), constrained_layout=True)
    x = np.linspace(0.0, 1.0, 4)
    ax.plot(x, x**2, label="curve")
    ax.scatter(x, x + 1.0, label="samples")
    ax.imshow(np.arange(4, dtype=float).reshape(2, 2), alpha=0.2)
    arrays, metadata = extract_figure_data(fig)

    assert "figure_size_inches" in arrays
    assert "axes_0_line_0_x" in arrays
    assert "axes_0_line_0_y" in arrays
    assert any(name.endswith("_image_0") for name in arrays)
    assert any(name.endswith("_collection_0_offsets") for name in arrays)
    assert metadata["axes"][0]["xlabel"] == ""
    assert "curve" in metadata["axes"][0]["legend_labels"]
    plt.close(fig)


def test_extract_figure_data_handles_bar_container_legends() -> None:
    """Extract metadata from bar charts whose legend handles are containers."""
    fig, ax = plt.subplots(figsize=(4, 3), constrained_layout=True)
    ax.bar([0, 1, 2], [1.0, 2.0, 1.5], label="bars")
    ax.legend()
    arrays, metadata = extract_figure_data(fig)

    assert arrays["axes_count"].shape == (1,)
    assert any(name.endswith("_patch_0_bounds") for name in arrays)
    assert "bars" in metadata["axes"][0]["legend_labels"]
    plt.close(fig)


def test_data_paths_for_figure_maps_output_figures_to_output_data() -> None:
    """Map a rendered artifact path to the matching chapter data sidecar paths."""
    npz_path, json_path = data_paths_for_figure(
        REPO_ROOT / "output" / "figures" / "chapter_08" / "demo.png"
    )
    assert npz_path == REPO_ROOT / "output" / "data" / "chapter_08" / "demo.npz"
    assert json_path == REPO_ROOT / "output" / "data" / "chapter_08" / "demo.json"


def test_validate_raw_data_exports_accepts_good_and_rejects_bad(
    tmp_path: Path,
) -> None:
    """Exercise the CLI validator on valid and intentionally corrupted fixtures."""
    save_chapter_data(
        2,
        "good",
        arrays={"x": np.arange(3, dtype=float)},
        metadata={"script": "good.py"},
        root=tmp_path,
    )
    validator = REPO_ROOT / "scripts" / "validate_raw_data_exports.py"
    good = subprocess.run(
        [sys.executable, str(validator), "--root", str(tmp_path), "--chapters", "2"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert good.returncode == 0, good.stderr

    bad_json = tmp_path / "chapter_02" / "good.json"
    manifest = json.loads(bad_json.read_text(encoding="utf-8"))
    manifest["arrays"]["x"]["shape"] = [99]
    bad_json.write_text(json.dumps(manifest), encoding="utf-8")

    bad = subprocess.run(
        [sys.executable, str(validator), "--root", str(tmp_path), "--chapters", "2"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert bad.returncode != 0
    assert "shape" in bad.stderr
