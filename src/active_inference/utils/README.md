# src/active_inference/utils/ — Utilities

Small helpers used across the package: grid construction, logging, output-path
conventions, and raw NPZ+JSON exports for saved chapter artifacts.

## Files

| File | What it defines |
|---|---|
| [`grids.py`](grids.py) | `make_grid`, `make_2d_grid` |
| [`logging.py`](logging.py) | `get_logger` |
| [`io.py`](io.py) | `default_figure_dir`, `default_data_dir`, `ensure_dir` |
| [`export.py`](export.py) | `save_chapter_data`, `extract_figure_data`, `extract_animation_data`, `data_paths_for_figure` |
| `__init__.py` | Re-exports all public names |

## Public API

```python
from active_inference.utils.grids import make_grid, make_2d_grid
from active_inference.utils.logging import get_logger
from active_inference.utils.io import default_figure_dir, default_data_dir, ensure_dir
from active_inference.utils import save_chapter_data
```

Also available from the top-level package:
```python
from active_inference import make_grid, get_logger
```

### `grids.py`

- `make_grid(low, high, n_points=500)` → evenly-spaced 1-D `np.ndarray`.
  Validates finite bounds, `low < high`, `n_points >= 2`.
- `make_2d_grid(x_low, x_high, y_low, y_high, n_x=200, n_y=200)` →
  `(x_array, y_array)` tuple for 2-D joint density grids.

### `logging.py`

- `get_logger(name="active_inference", level=logging.INFO)` → configured
  `logging.Logger` with a `StreamHandler(stdout)` and a consistent format
  (`[HH:MM:SS] LEVEL  name: message`). Idempotent — repeated calls for the
  same name return the same logger without adding duplicate handlers.

### `io.py`

- `default_figure_dir()` → `Path("output/figures")` (relative to repo root).
- `default_data_dir()` → `Path("output/data")` (relative to repo root).
- `ensure_dir(path)` → create `path` (and parents) if missing; return it.

### `export.py`

- `save_chapter_data(chapter, stem, arrays, metadata, figures=...)` writes a
  compressed `NPZ` array bundle and paired `JSON` manifest under
  `output/data/chapter_NN/`.
- `extract_figure_data(fig)` and `extract_animation_data(anim)` provide the
  automatic raw-data capture used by shared visualization save helpers.
- `data_paths_for_figure(path)` maps a saved figure path to its raw-data
  sidecar paths.

## Design Decisions

- **`__file__`-relative paths:** `io.py` computes the repo root as
  `Path(__file__).resolve().parents[3]`, so `default_figure_dir()` always
  points to the right place regardless of the working directory.
- **Validated raw data**: export arrays must be finite, numeric, non-empty, and
  non-object; manifests include shapes, dtypes, figures, seed when present, and
  summary statistics.
- **Minimal, focused functions** — each does one thing well.

## Testing

Each module has a dedicated test file under `tests/utils/`:
`test_grids.py`, `test_io.py`, and `test_logging.py`. The helpers are also
exercised indirectly through every chapter script and the rest of the suite.
