# Chapter 5 — concept map

Chapter 4 turned posterior estimation into the minimization of variational free
energy (VFE). Chapter 5 asks: what does that minimization look like when we commit
to the simplest possible variational family — a point belief — and let the brain
*descend* the free energy in continuous time? The answer is **predictive coding
(PC)**: gradient descent on the MAP/Laplace form of VFE, organized entirely around
**precision-weighted prediction errors**.

The running model is the same linear-Gaussian example used since Chapter 3
(`y = β₀ + β₁x + noise`, prior `x ~ N(4, 0.25)`), so the *exact* grid posterior
`N(2.4, 0.05)` is still available as an **oracle**. The headline result of this
chapter's code is that the PC fixed point lands on `μ = 2.4` — i.e. predictive
coding, variational inference, and grid Bayes all converge on the same belief.

## From VFE to the prediction-error form

Fix the variational belief to a point mass at `μ_x` (the MAP/Laplace move, book
§5.1). The free energy collapses to the negative log joint at `μ_x`:

```
F_MAP = −log p(y | x=μ_x) − log p(x=μ_x)
      = ½(λ_y ε_y² + λ_x ε_x² + log σ_y² + log s_x²) + const     (Eq. 7a)
```

with two **prediction errors** and their **precisions** `λ = 1/σ²`:

| Symbol | Definition | Eq. | Meaning |
|--------|-----------|-----|---------|
| `ε_y` | `y − g(μ_x)` | 6a | **sensory** error: data minus prediction |
| `ε_x` | `μ_x − m_x` | 6b | **state** error: belief minus prior |
| `λ_y` | `1/σ_y²` | — | sensory precision (reliability of data) |
| `λ_x` | `1/s_x²` | — | prior precision (confidence in prior) |

`F` trades the two squared, precision-weighted errors against each other; its
minimum is the MAP estimate. `plot_prediction_errors` reproduces Fig. 5.1.2: the
parabola `F(μ_x)` with its minimum between the prior mean `m_x=4` and the
data-consistent state, and the two weighted errors `λ_x ε_x²`, `λ_y ε_y²` crossing.

## Recognition dynamics (the perception equation)

Perception is gradient descent on `F`:

```
μ_x ← μ_x − κ ∂F/∂μ_x,   ∂F/∂μ_x = λ_x ε_x − λ_y ε_y g'(μ_x)     (Eq. 15/16)
```

This is **the simplest perception equation in continuous active inference**.

> **A derived, not transcribed, gradient.** The book's sign convention for this
> gradient is internally inconsistent across Eq. 6b, the p.283 text, and Eq. 16.
> Rather than copy a sign, the companion *derives* it by the chain rule
> (`d/dμ · ½λ_y(y−g(μ))² = −λ_y ε_y g'(μ)`; `d/dμ · ½λ_x(μ−m_x)² = λ_x ε_x`) and
> **verifies it against a central finite difference** at several off-equilibrium
> points (`test_analytic_grad_matches_fd`), with a wrong-sign mutation test proving
> the check is not vacuous (`test_gradient_check_large_for_wrong_grad`).

> **Step-size is conditionally stable.** `F` is quadratic in `μ` with curvature
> `L = λ_x + β₁²λ_y`; fixed-step descent converges iff `κ < 2/L` (the book's
> κ-sensitivity warning, made precise). The fixed point is κ-independent. The
> cross-chapter sweep picks `κ = 0.4/L` so descent is stable for every config.

## Three algorithms

Each is a configurable building block in
`active_inference.estimators.predictive_coding`, demonstrated by a thin
orchestrator in `chapters/chapter_05/`.

| Algorithm | Key API | Book | Orchestrator |
|-----------|---------|------|--------------|
| Prediction-error visualization | `plot_prediction_errors(model, y, mu_grid, ...)` | §5.1 | `example_5_1_prediction_errors.py` |
| Univariate PC (perception) | `predictive_coding_inference(model, y, ...)` | Alg. 5.2.1 | `example_5_4_recognition_dynamics.py` |
| Multivariate PC | `multivariate_predictive_coding(g, jacobian, y, m_x, ...)` | §5.3 | `example_5_3_multivariate.py` |
| Hierarchical PC | `hierarchical_predictive_coding(model, y, ...)` | §5.4, Ex. 5.7 | `example_5_7_hierarchical.py` |

* **Univariate PC** descends Eq. 16 from the prior mean. For a linear `g` the fixed
  point is the exact posterior mean; for a nonlinear `g` it is the grid argmin of
  `F` (both verified as oracles).
* **Multivariate PC** generalizes to vector states with update
  `μ ← μ − κ(Π_x ε_x − Jᵀ Π_y ε_y)`, where `J` is the Jacobian of `g` and `Π` are
  precision matrices. Reduces exactly to the scalar case (tested to 1e-6). For a
  linear `g(x)=Ax+b` the fixed point has a closed form,
  `μ* = (Π_x + AᵀΠ_yA)⁻¹(Π_x m_x + AᵀΠ_y(y−b))` — the vector counterpart of
  `pc_linear_fixed_point`, exposed as `pc_multivariate_linear_fixed_point` and used
  as the figure's oracle (it reduces to the least-squares inverse `A⁻¹(y−b)` under a
  flat prior). The result also records per-iteration prediction-error traces so the
  errors can be plotted, exactly like the scalar case.
* **Hierarchical PC** stacks `L+1` layers with `μ^{[0]}=y` clamped. Each layer
  predicts the one below (`ε^{[l]} = μ^{[l]} − g^{[l+1]}(μ^{[l+1]})`); the top is
  **unconstrained** (`m_x=0 ⇒ ε^{[L]}=μ^{[L]}`, book p.306). The summed VFE
  `F = Σ_l ½(λ^{[l]} ε^{[l]}² + log σ²_{[l]})` (Eq. 34) is minimized by simultaneous
  (Jacobi) updates of all hidden layers. Example 5.7 converges to `[2,1,0]` with
  `Σ F = 0` — exact, because every error vanishes and the unit variances make
  `log σ² = 0`.

## Verification — the cross-chapter oracle

Predictive coding is verified against an **independent** source of truth, not its
own internals:

* **`μ* = 2.40000` = Ch.4 grid posterior mean** for the linear model
  (`oracle_agreement`, err ≈ 4.6e-6). This is not a single lucky point:
  `TestLinearOracleSweep` runs **144 configurations** (varying β₀, β₁, σ_y², m_x,
  s_x², y) and asserts the PC fixed point equals *both* the closed-form Gaussian
  posterior mean *and* Ch.4's `GridBayesianInference` posterior mean across all of
  them (oracle spread > 1 unit).
* **Gradient analytic == finite difference** (`gradient_check`, ~1e-9), with the
  wrong-sign falsification test.
* **Hierarchical reproduces Example 5.7** `[2,1,0]`, `Σ F = 0`, all errors → 0.
* **Free energy is monotone non-increasing** along every descent
  (`convergence_report`).

## Reusable building blocks

* **`active_inference.core.predictive_coding`** — `GenerativeFunction` (ABC) and
  the concrete `LinearFunction`, `QuadraticFunction`, `GenericFunction` (with
  finite-difference derivative); `PredictiveCodingModel`; the error functions
  `sensory_prediction_error`, `state_prediction_error`; `PCFreeEnergy` +
  `predictive_coding_free_energy` (Eq. 7a); and the gradients `pc_free_energy_grad`
  (analytic, Eq. 16) and `pc_free_energy_grad_fd` (numerical oracle).
* **`active_inference.estimators.predictive_coding`** — `predictive_coding_inference`,
  `multivariate_predictive_coding`, `hierarchical_predictive_coding`,
  `HierarchicalPCModel`, and the result traces `PredictiveCodingResult`,
  `MultivariatePCResult`, `HierarchicalPCResult`.
* **`active_inference.core.diagnostics`** — the statistics / validation layer:
  `gradient_check` (analytic-vs-FD), `convergence_report` → `ConvergenceReport`
  (monotonicity, empirical convergence rate), `oracle_agreement` → `OracleAgreement`
  (estimate-vs-oracle).
* **`active_inference.visualizations.unified`** — the streamlined shared layer
  (see below).

## The unified visualization layer

`active_inference.visualizations.unified` is the "best streamlined unified
visualizations" system shared across Chapters 4 and 5. Instead of each chapter
re-specifying figure sizes, palettes, grids and legends, every inference *result*
renders through one vocabulary:

| Function | Reproduces | Accepts |
|----------|------------|---------|
| `plot_recognition_dynamics(result, ...)` | descent figures | **either** a Ch.4 `FixedFormResult` (2 panels) **or** a Ch.5 `PredictiveCodingResult` (3 panels) — duck-typed on `.mus` / `.free_energies` / `.eps_*` |
| `plot_prediction_errors(model, y, grid, ...)` | Fig. 5.1.2 | a `PredictiveCodingModel` |
| `plot_multivariate_pc(result, truth=, oracle=)` | §5.3 (3 panels) | a `MultivariatePCResult` — per-component belief with truth/oracle lines, ‖ε_x‖/‖ε_y‖ decay, free-energy descent |
| `plot_hierarchical_pc(result, ...)` | Fig. 5.4.4 | a `HierarchicalPCResult` |
| `panel_grid`, `finalize`, `layer_colors` | — | styling primitives every panel routes through |

Every panel is **annotated and quantified**, not just drawn:

* **Bigger type** — the package `DEFAULT_RC` uses bold 16 pt titles, 15 pt axis
  labels and 12 pt legends, sized for slides and print.
* **Statistics boxes** — each panel carries a monospace readout: `μ₀ → μ*`, the
  oracle and `|μ*−oracle|`, iteration count / convergence flag, `F₀ → F*`, `ΔF`,
  and the empirical convergence rate.
* **Analytical landmarks** — the fixed point `μ*` is marked on the belief curve
  with a callout; `plot_prediction_errors` reports the *closed-form* minimizer
  (`pc_linear_fixed_point`) and curvature `L` (`pc_curvature_linear`) alongside the
  grid argmin; hierarchical layers are tagged with their final `μ^{[l]}`.
* **More legends** — every panel has a legend, and reference lines carry their
  values (`truth x*=2`, `oracle=2.4`, `−log p(y)=7.431`).

`test_unified.py` asserts the shared path *programmatically* (panel counts, that
the plotted belief line is the result's `μ` trace, axis labels, that each panel has
a legend, and that the stat boxes quote the *real* `μ*` and oracle error) on real
result objects from both chapters — so "the figure looks right" is a quoted
artifact, not an eyeball, and the shared refactor cannot silently break Chapter 4.

## Animations

Two GIF orchestrators bring the dynamics to life, both built on the **composable**
animators in `visualizations.animations` (same `result`-in → animation-out contract
as the static plotters, same bold style, a `stride` knob for size):

| Script | Animator | Shows |
|--------|----------|-------|
| `animation_recognition_descent.py` (`--nonlinear`) | `animate_recognition_dynamics` | `μ_x` descending onto the oracle (linear → Ch.4 posterior mean 2.4), errors decaying, `𝓕` falling — with moving markers and a live `iter / μ / 𝓕` box |
| `animation_hierarchical.py` | `animate_hierarchical_pc` | Example 5.7 settling to `[2,1,0]`, every `ε^{[l]} → 0`, and `Σ F → 0` (Fig. 5.4.4 in motion) |

`animate_recognition_dynamics` is the animation twin of `plot_recognition_dynamics`:
the *same* function animates a Chapter 4 `FixedFormResult` (2 panels) and a Chapter 5
`PredictiveCodingResult` (3 panels), duck-typed on the result's traces. `test_animations.py`
validates panel counts, per-panel legends, and that strided GIFs render fewer frames.

## Where the book takes this next

Chapter 5's recognition dynamics are *perception only* — inference about a static
cause. Part II adds **time**: generalized coordinates and filtering (continuous
state estimation), then **action** (active generalized filtering), where the same
prediction-error machinery drives motor output, not just belief updates. The
`GenerativeFunction` / precision interface here is exactly what those chapters
extend.
