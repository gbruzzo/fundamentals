"""§5.3 — Multivariate predictive coding.

Run::

    python chapters/chapter_05/example_5_3_multivariate.py [--save]

Generalizes the scalar recognition dynamics to a vector hidden state with
precision *matrices* and a Jacobian matrix (book §5.3). Here a 2-D state drives a
2-D observation through a linear map ``g(x) = A x + b`` (constant Jacobian ``A``);
the recognition dynamics ``μ ← μ − κ(Π_x ε_x − Aᵀ Π_y ε_y)`` recover the state.
With a near-flat prior the fixed point is the least-squares inverse ``A⁻¹(y−b)``.
On a 1-D restriction this routine reproduces the scalar predictive coding result
exactly (asserted in the test-suite).
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from active_inference import (
    get_logger,
    multivariate_predictive_coding,
    pc_multivariate_linear_fixed_point,
)
from active_inference.utils.io import default_figure_dir, ensure_dir
from active_inference.visualizations import plot_multivariate_pc, save_or_show

LOG = get_logger("ch5.ex3")


def parse_args() -> argparse.Namespace:
    """Parse command-line options for this executable entry point."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--save", action="store_true")
    return p.parse_args()


def main() -> None:
    """Run the chapter orchestrator and render or display its outputs."""
    args = parse_args()
    A = np.array([[2.0, 0.5], [-0.3, 1.5]])
    b = np.array([1.0, -1.0])
    x_true = np.array([2.0, -1.0])
    y = A @ x_true + b
    precision_y = np.array([1.0, 1.0])
    precision_x = np.array([1e-4, 1e-4])  # near-flat prior → least-squares

    res = multivariate_predictive_coding(
        g=lambda x: A @ x + b,
        jacobian=lambda x: A,
        y=y,
        m_x=np.zeros(2),
        precision_y=precision_y,
        precision_x=precision_x,
        kappa=0.05,
        n_iter=5000,
    )
    # Independent oracle: the closed-form multivariate linear fixed point — the
    # vector counterpart of pc_linear_fixed_point (≈ A⁻¹(y−b) under a flat prior).
    oracle = pc_multivariate_linear_fixed_point(
        A, b, y, np.zeros(2), precision_y=precision_y, precision_x=precision_x)
    LOG.info("multivariate mu*=%s | closed-form oracle=%s | ‖μ*−o‖=%.2e | converged=%s",
             np.round(res.mu_star, 4), np.round(oracle, 4),
             float(np.linalg.norm(res.mu_star - oracle)), res.converged)

    fig = plot_multivariate_pc(res, truth=x_true, oracle=oracle)

    if args.save:
        out = ensure_dir(default_figure_dir() / "chapter_05")
        save_or_show(fig, out / "example_5_3_multivariate.png")
        LOG.info("Saved to %s", out)
    else:
        plt.show()


if __name__ == "__main__":
    main()
