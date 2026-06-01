# `chapters/chapter_05/` — Predictive Coding

Chapter 5 scripts implement **predictive coding**: gradient descent on the MAP/Laplace
free energy, driven by precision-weighted prediction errors. The univariate recognition
dynamics, multivariate generalization, and hierarchical stack all reduce to one descent
law (Eq. 16). The linear fixed point equals Chapter 4's grid posterior mean — the
cross-chapter oracle that verifies the runs.

## Scripts

| Script | Lines | What it shows |
|---|---|---|
| [`example_5_1_prediction_errors.py`](example_5_1_prediction_errors.py) | ~100 | Flat-prior MLE + the MAP free energy as two precision-weighted prediction errors (Fig. 5.1.2). |
| [`example_5_3_multivariate.py`](example_5_3_multivariate.py) | ~80 | Multivariate PC (vector state, Jacobian `g`) converges; reduces to scalar. |
| [`example_5_4_recognition_dynamics.py`](example_5_4_recognition_dynamics.py) | ~100 | Recognition dynamics (Alg. 5.2.1); `--linear` → Ch.4 posterior mean 2.4. |
| [`example_5_7_hierarchical.py`](example_5_7_hierarchical.py) | ~70 | Hierarchical PC → `[2, 1, 0]`, all `ε → 0`, `Σ F = 0` (Fig. 5.4.4). |
| [`animation_recognition_descent.py`](animation_recognition_descent.py) | ~105 | GIF: `μ_x` descending onto the oracle, errors decaying, `𝓕` falling. |
| [`animation_hierarchical.py`](animation_hierarchical.py) | ~75 | GIF: layer beliefs settling to `[2, 1, 0]`, errors → 0, `Σ F → 0`. |

## Running

```bash
# Single script (headless — saves to output/figures/chapter_05/)
python chapters/chapter_05/example_5_4_recognition_dynamics.py --linear --save

# Every Chapter 5 script at once
python scripts/run_all_figures.py --chapters 5
```

Each script accepts `--save` for headless rendering. `example_5_4` takes `--linear`
(oracle cross-check) and `animation_recognition_descent` takes `--nonlinear`.

## Library Usage

```python
from active_inference import (
    LinearFunction, QuadraticFunction, PredictiveCodingModel,
    predictive_coding_inference, multivariate_predictive_coding,
    hierarchical_predictive_coding, HierarchicalPCModel,
    pc_linear_fixed_point, pc_curvature_linear,
    GridBayesianInference, oracle_agreement,
)
```

## Smoke Tests

`tests/chapters/test_smoke.py` runs each script via `subprocess` with `--save` and
asserts exit code 0 (`test_chapter_5_scripts_run`, `test_chapter_5_animations`). Unit
tests for the methods live in `tests/core/test_predictive_coding.py` and
`tests/estimators/test_predictive_coding.py`.

## Key Concepts

- **Recognition dynamics** `μ ← μ − κ(λ_x ε_x − λ_y ε_y g'(μ))` (Eq. 16). The gradient
  sign is *derived* (the book's convention is inconsistent) and verified by finite
  difference.
- **Conditional stability.** `F` is quadratic with curvature `L = λ_x + g'²λ_y`;
  fixed-step descent converges iff `κ < 2/L` (`pc_curvature_linear`).
- **Cross-chapter oracle.** For linear `g` the fixed point is the exact Gaussian
  posterior mean (`pc_linear_fixed_point`), so PC, variational inference (Ch.4) and grid
  Bayes (Ch.1) all converge on `μ = 2.4`.
- **Hierarchical PC** stacks `L+1` layers with `μ^{[0]}=y` clamped and an unconstrained
  top (`m_x=0`); unit variances make `Σ F = 0` exactly at convergence.
