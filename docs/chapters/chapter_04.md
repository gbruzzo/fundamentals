# Chapter 4 — concept map

Chapter 4 is the hinge of the book. Chapters 1–3 computed posteriors either in
closed form (conjugate Gaussians, the LGS) or by point estimation (MLE/MAP/EM).
Chapter 4 asks what to do when the posterior has *no* closed form, and answers
with **variational Bayesian inference (VBI)**: turn posterior estimation into an
optimization problem. Pick a tractable variational density `q(x)` and adjust it
until it matches the true posterior `p(x | y)`, by minimizing **variational free
energy** (VFE, `𝓕`).

Everything is built on the same linear-Gaussian running example as Chapter 3
(`y = β₀ + β₁x + noise`, prior `x ~ N(4, 0.25)`), so the *exact* grid posterior
`N(2.4, 0.05)` and the exact log-evidence are available as an **oracle** to
verify every variational result against.

## The one objective, five forms

VFE is a single number with five algebraically-equivalent decompositions
(book §4.2–4.4). `variational_free_energy(...)` computes them all at once and
`VFEComponents.check()` asserts they agree to grid precision.

| Form | Decomposition | Eq. | Intuition |
|------|---------------|-----|-----------|
| **G** (generative) | `𝓕 = E_q[log q] − E_q[log p(x,y)]` | 10 | distance from `q` to the *unnormalized* model |
| **D** (divergence) | `𝓕 = D_KL(q‖p(x\|y)) − log p(y)` | 18/27 | divergence-from-posterior + surprisal |
| **C** (complexity) | `𝓕 = D_KL(q‖p(x)) − E_q[log p(y\|x)]` | 28 | complexity − accuracy (the Occam term) |
| **E** (energy) | `𝓕 = −H[q] − E_q[log p(x,y)]` | 29 | (negative) entropy + average energy |
| **MAP / MLE** | point-mass special cases | 30/31 | recover MAP/MLE estimation |

Two facts proved in §4.3 anchor all verification:

* **Upper bound on surprisal** (Eq. 26): `𝓕 ≥ −log p(y)` for *any* `q`.
* **Tight at the posterior**: `𝓕 = −log p(y)` exactly when `q(x) = p(x | y)`
  (the D-form divergence vanishes).

## Three algorithms for minimizing VFE

Each is a configurable building block in `active_inference.estimators.variational`,
demonstrated by a thin orchestrator in `chapters/chapter_04/`.

| Algorithm | Key API | Book | Orchestrator |
|-----------|---------|------|--------------|
| Coordinate search (zero-order) | `coordinate_search_vfe(model, y, x_grid, ...)` | Alg. 4.2.1, Ex. 4.1 | `example_4_1_coordinate_search.py` |
| Fixed-form VI (gradient on `(μ, σ²)`) | `fixed_form_vi(model, y, x_grid, ...)` | Alg. 4.6.1, Ex. 4.7 | `example_4_7_fixed_form.py` |
| Free-form mean-field CAVI | `free_form_cavi(y, cfg=...)` | Alg. 4.5.1, Ex. 4.5/4.6 | `example_4_6_free_form_cavi.py` |

Additional script coverage: `example_4_2_surprisal.py` renders the surprisal
relationship, `example_4_3_vfe_forms.py` compares the five decompositions,
`animation_vfe_descent.py` animates `q(x)` approaching the posterior,
`visualize_kl_loss.py`, `visualize_vfe_intuition.py`, and
`visualize_model_comparison.py` provide supporting figures, and
`interactive_vfe_explorer.py` exposes the same surface with sliders.

* **Coordinate search** evaluates VFE at the eight `(μ ± κ, σ² ± κ)` neighbours
  and jumps to the lowest — VFE *is* a usable loss with no gradient at all. With
  the book's `κ=0.01, 20` iterations it deliberately stops short of the minimum
  (§4.4, Fig. 4.4.1); pass `--extended` to converge fully.
* **Fixed-form VI** fixes `q` to Gaussian and follows the gradient of VFE w.r.t.
  `(μ, σ²)`. The book used PyTorch autodiff; this companion stays dependency-light
  and uses central finite differences — same method, no torch. Converges exactly
  to `N(2.4, 0.05)` with `𝓕 → −log p(y)`.
* **Free-form CAVI** drops the Gaussian assumption and runs mean-field
  coordinate-ascent on the three unknowns `(x, β₀, β₁)`, using the fundamental
  theorem of mean-field VI (Eq. 43). For this conjugate model every update is
  closed-form Gaussian, and VFE is guaranteed non-increasing per sweep.

## Reusable building blocks

* **`active_inference.core.variational`** — `GaussianBelief` (the parametric
  `q`), `variational_free_energy` → `VFEComponents` (all five forms + the
  decomposition terms `divergence`, `surprisal`, `complexity`, `accuracy`,
  `neg_entropy`, `energy`), the thin wrappers `vfe_g_form` / `vfe_d_form` /
  `vfe_c_form` / `vfe_e_form` / `vfe_map_form` / `vfe_mle_form`, and the oracle
  helpers `log_model_evidence`, `surprisal`, `free_energy_bound_gap`.
* **`active_inference.estimators.variational`** — `coordinate_search_vfe`,
  `fixed_form_vi`, `free_form_cavi`, their result traces
  (`CoordinateSearchResult`, `FixedFormResult`, `CAVIResult`) and the CAVI model
  config `MeanFieldConfig`.
* **`active_inference.visualizations.variational`** — `vfe_surface`,
  `plot_vfe_contour` (Fig. 4.2.2/4.6.1 left), `plot_density_evolution`
  (right panel), `plot_vfe_decomposition` (the G/C/E panels of Fig. 4.4.1/4.6.2),
  and `plot_surprisal_relationship` (Fig. 4.3.1).
* **`active_inference.visualizations.animations`** — `animate_vfe_descent`, the
  dynamic version of the descent figures (pillow GIF, no FFmpeg).

## Extra orchestrators (beyond the numbered examples)

| Script | Reproduces | What it shows |
|--------|------------|---------------|
| `example_4_2_surprisal.py` | Fig. 4.3.1 | surprisal `−log p(y)` vs `y` and vs `p(y)` |
| `example_4_3_vfe_forms.py` | Fig. 4.4.1 | all five forms agree; G/C/E decompositions over a descent |
| `visualize_kl_loss.py` | §4.1 | the KL loss surface vs the VFE surface (same minimum, no posterior needed) |
| `visualize_vfe_intuition.py` | Fig. 4.2.3 | G-form intuition: `q(x)`, `p(x,y)`, and the posterior |
| `visualize_model_comparison.py` | Fig. 4.3.2/4.3.3 | model evidence of a good vs bad model against the true input distribution |
| `interactive_vfe_explorer.py` | — | GUI sliders to drag `(μ, σ²)` and watch the decomposition trade off (skipped headless) |

## Where the book takes this next

Chapter 5 builds the *generative model* of perception on top of this VFE
machinery, and later chapters add action (active inference proper) by extending
free energy to *expected* free energy over policies. The `GenerativeModel`
interface (`log_likelihood`, `log_prior`) that VFE consumes here is exactly the
interface those chapters extend, so the variational core composes forward
unchanged.
