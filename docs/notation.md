# Notation cheatsheet

Quick reference for the symbols used throughout the codebase. The notation
follows the conventions established in the book and Appendix B; this file is
descriptive — it explains how the symbols map to identifiers in the package.

## Variables

| Symbol     | Identifier        | Meaning |
|------------|-------------------|---------|
| `x*`       | `x_star`, `x_true`| External (true) state of the environment. |
| `x`        | `x`               | Agent's *hidden state* — its belief about `x*`. |
| `y`        | `y`, `y_obs`      | Observation produced by the generative process. |
| `Y`        | `Y`               | Stacked observations as an `(N, D)` data matrix. |
| `X`        | `X`               | Design matrix `(N, C)` (or `(N, C+1)` with intercept). |
| `Θ` / `θ`  | `Theta` / `theta` | Mixing matrix (LGS / FA) or parameter vector (regression). |
| `ζ`        | `mu_zeta`         | Log precision for Chapter 8 attention / second-order parameter learning. |
| `μ̃_x`      | `mu_tilde`, `mus` | Generalized-coordinate hidden-state belief; public result tensors use `(time, embedding_order, variable)`. |
| `ỹ`        | `y_tilde`, `y_tildes` | Generalized measurements built from observations and their finite-difference derivatives. |
| `a`        | `action`, `actions` | Continuous action/control signal in Chapter 7 active generalized filtering. |
| `b`        | `b`               | Optional offset added to the linear generator. |
| `g_E`      | `process.mean()`  | Generating function inside the environment. |
| `g_M`      | `model.predict_mean()` | Agent's generating function. |
| `psi(x)`   | `psi`             | Optional nonlinear transform of `x`. |
| `omega_y`  | (sampled noise)   | Zero-mean Gaussian observation noise with variance `sigma2_y` (uni) or `cov_y` (multi). |

## Parameters

| Symbol     | Identifier            | Meaning |
|------------|------------------------|---------|
| `beta0`    | `beta0`                | Linear intercept of the generating function. |
| `beta1`    | `beta1`                | Linear slope. |
| `sigma2_y` | `sigma2_y`             | Variance of the likelihood (observation noise). |
| `m_x`      | `m_x`                  | Prior mean over the hidden state. |
| `s2_x`     | `s2_x`                 | Prior variance over the hidden state. |
| `theta*`   | `process.theta`        | Parameters of the generative process. |
| `theta`    | model attributes       | Parameters of the generative model. |
| `λ` / `Π`  | `lambda_*`, `precision_*` | Precision (inverse variance/covariance); Chapter 8 uses `exp(mu_zeta)`. |
| `Π̃(γ)`    | `correlated_embedding_precision(...)` | Generalized precision with correlated embedding orders. |
| `S(γ)`     | `gaussian_temporal_covariance(...)` | Gaussian temporal covariance over embedding orders. |
| `γ`        | `gamma`                 | Smoothness/roughness parameter controlling embedding-order correlations. |

## Densities

| Symbol            | Identifier in code             | Notes |
|-------------------|--------------------------------|-------|
| `p(y \| x)`       | `model.log_likelihood(...)`    | Likelihood / observation model. |
| `p(x)`            | `model.log_prior(...)`         | Hidden-state prior. |
| `p(x, y)`         | `inferer.joint_density(...)`   | Joint distribution. |
| `log p(y)`        | `result.log_evidence`          | Log marginal likelihood (model evidence); `surprisal = −log p(y)`. |
| `p(x \| y)`       | `result.posterior`             | Posterior on the grid. |

## Algorithms

| Acronym | Identifier | Where |
|---------|------------|-------|
| MLE     | `mle_analytic_linear`, `mle_loss`, `mle_grad_x`     | `estimators/mle.py` |
| MAP     | `map_analytic_linear`, `map_loss`, `map_grad_x`     | `estimators/map.py` |
| GD      | `gradient_descent`                                  | `estimators/gradient_descent.py` |
| OLS / multiple regression | `mle_linear_regression`           | `estimators/linear_regression.py` |
| BLR     | `BayesianLinearRegression`                          | `estimators/linear_regression.py` |
| LGS     | `LinearGaussianSystem`                              | `core/lgs.py` |
| EM (FA) | `fit_factor_analysis`, `factor_analysis_e_step`/`_m_step` | `estimators/em.py` |
| MVN     | `mvn_pdf`, `mvn_log_pdf`, `mvn_sample`, `mahalanobis_squared` | `core/distributions.py` |
| Pipeline | `Pipeline`, `Pipeline.linear_gaussian`             | `core/compose.py` |
| Running stats | `running_stats`, `RunningPosteriorStats`       | `core/compose.py` |
| VFE (Ch.4) | `variational_free_energy`, `coordinate_search_vfe`, `fixed_form_vi`, `free_form_cavi` | `core/variational.py`, `estimators/variational.py` |
| PC (Ch.5) | `predictive_coding_inference`, `multivariate_predictive_coding`, `hierarchical_predictive_coding`, `pc_free_energy_grad`, `pc_linear_fixed_point` | `core/predictive_coding.py`, `estimators/predictive_coding.py` |
| GF (Ch.6) | `generalized_filter`, `generalized_filter_gc`, `generalized_vector_filter`, `correlated_embedding_precision` | `core/generalized_filtering.py`, `estimators/generalized_filtering.py` |
| AIF (Ch.7) | `simulate_active_inference`, `simulate_multivariate_active_inference`, `multivariate_action_gradient` | `core/active_inference.py`, `estimators/active_inference.py` |
| AGF learning/attention (Ch.8) | `learning_attention_free_energy`, `simulate_learning_attention`, `hierarchical_message_terms` | `core/continuous_learning.py`, `estimators/continuous_learning.py` |

## Information-theoretic / FEP quantities

| Symbol      | Identifier | Notes |
|-------------|-----------|-------|
| ``H[p]``    | `grid_entropy`, `gaussian_entropy_univariate/_mvn`, `InferenceResult.entropy` | Differential entropy in nats. |
| ``KL[p‖q]`` | `grid_kl_divergence`, `gaussian_kl_univariate/_mvn`, `InferenceResult.kl_from_prior` | Non-negative; asymmetric. |
| ``F[q]``    | derived: `−result.log_evidence + result.kl_from_prior()` | Variational free energy. |
| log-score   | `log_score_gaussian` | Higher is better. |
| CRPS        | `crps_gaussian` | Lower is better. |
| ESS         | `effective_sample_size` | Kish ESS in log space. |

## Conventions in the code

- A trailing ``_grid`` denotes a 1-D NumPy array of points.
- Functions that work in log-space are suffixed ``_log`` (or named
  ``log_likelihood`` / ``log_prior``).
- Parameters that the book denotes with a star (``*``) — i.e. quantities of
  the *generative process* — are spelled out in code as ``x_star``,
  ``beta0_true``, etc., to avoid confusion with Python's unpacking operator.
