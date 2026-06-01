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

from active_inference import get_logger, multivariate_predictive_coding
from active_inference.utils.io import default_figure_dir, ensure_dir
from active_inference.visualizations import finalize, panel_grid, save_or_show
from active_inference.visualizations.style import COLORS

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

    res = multivariate_predictive_coding(
        g=lambda x: A @ x + b,
        jacobian=lambda x: A,
        y=y,
        m_x=np.zeros(2),
        precision_y=np.array([1.0, 1.0]),
        precision_x=np.array([1e-4, 1e-4]),  # near-flat prior → least-squares
        kappa=0.05,
        n_iter=5000,
    )
    expected = np.linalg.solve(A, y - b)
    LOG.info("multivariate mu*=%s | least-squares inverse=%s | converged=%s",
             np.round(res.mu_star, 4), np.round(expected, 4), res.converged)

    mus = res.mus
    it = np.arange(mus.shape[0])
    fig, axes = panel_grid(2, figsize=(11, 4.2),
                           title="§5.3 · multivariate predictive coding")
    ax = axes[0]
    for d in range(mus.shape[1]):
        ax.plot(it, mus[:, d], lw=2, label=rf"$\mu_{{{d + 1}}}$")
        ax.axhline(x_true[d], color=COLORS["neutral"], ls="--", lw=1)
    finalize(ax, xlabel="iteration", ylabel=r"$\mu$", title="state estimate")

    ax = axes[1]
    ax.plot(it, res.free_energies, color=COLORS["data"], lw=2, label=r"$\mathcal{F}$")
    finalize(ax, xlabel="iteration", ylabel=r"$\mathcal{F}$", title="free energy")

    if args.save:
        out = ensure_dir(default_figure_dir() / "chapter_05")
        save_or_show(fig, out / "example_5_3_multivariate.png")
        LOG.info("Saved to %s", out)
    else:
        plt.show()


if __name__ == "__main__":
    main()
