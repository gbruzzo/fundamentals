# Chapter 5 — Predictive Coding

Chapter 5 commits to the simplest variational family — a point belief — and lets
the brain *descend* the free energy in continuous time. The result is **predictive
coding**: gradient descent on the MAP/Laplace form of VFE, organized around
**precision-weighted prediction errors**. The deeper concept map is
[`docs/chapters/chapter_05.md`](../../docs/chapters/chapter_05.md).

This folder contains four example orchestrators and two animations. The headline
result: the linear predictive-coding fixed point lands on `μ = 2.4` — the same
belief Chapter 4's variational inference and the grid Bayesian posterior reach, so
the cross-chapter oracle (`pc_linear_fixed_point` / `GridBayesianInference`) verifies
every run.

## Scripts

### Numbered examples

| Script | Mirrors | What it shows |
|--------|---------|---------------|
| `example_5_1_prediction_errors.py`   | §5.1 / Fig. 5.1.2 | Flat-prior MLE and the MAP free energy as two precision-weighted prediction errors. |
| `example_5_3_multivariate.py`        | §5.3        | Multivariate predictive coding (vector state, Jacobian `g`) converges. |
| `example_5_4_recognition_dynamics.py`| Alg. 5.2.1  | Recognition dynamics (Eq. 16); `--linear` lands on the Ch.4 grid posterior mean `2.4`. |
| `example_5_7_hierarchical.py`        | Example 5.7 / §5.4 | Hierarchical PC converges to `[2, 1, 0]`, all layer errors → 0, `Σ F = 0` (Fig. 5.4.4). |

### Animations (GIFs)

| Script | What it shows |
|--------|---------------|
| `animation_recognition_descent.py` | `μ_x` descending onto the oracle, errors decaying, `𝓕` falling (`--nonlinear` for Fig. 5.2.3). |
| `animation_hierarchical.py`        | Layer beliefs settling to `[2, 1, 0]`, errors → 0, `Σ F → 0` (Fig. 5.4.4 in motion). |

## Running

```bash
# from the repo root
uv sync                                                       # one-time setup
uv run python chapters/chapter_05/example_5_4_recognition_dynamics.py --linear --save

# or via the top-level menu
./run.sh --chapter 5
```

Pass `--save` to dump figures (and GIFs) into `output/figures/chapter_05/`. Without
it the scripts open interactive windows. `example_5_4` defaults to the nonlinear
model; pass `--linear` for the cross-chapter oracle cross-check.

## Programmatic usage

```python
from active_inference import (
    LinearFunction, PredictiveCodingModel,
    predictive_coding_inference, pc_linear_fixed_point,
)

model = PredictiveCodingModel(g=LinearFunction(2.0, 3.0), sigma2_y=0.25, m_x=4.0, s2_x=0.25)
res = predictive_coding_inference(model, 7.0, kappa=0.02, n_iter=2000)
print(res.mu_star)                          # ≈ 2.4
print(pc_linear_fixed_point(model, 7.0))    # 2.4 exactly — the closed-form oracle
```
