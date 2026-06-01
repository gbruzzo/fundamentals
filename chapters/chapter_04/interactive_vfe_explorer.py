"""Interactive — explore variational free energy by hand (Chapter 4, bonus).

Run::

    python chapters/chapter_04/interactive_vfe_explorer.py     # opens a GUI window

Two matplotlib sliders move the variational density ``q(x)=N(μ_x, σ²_x)``. The
top panel overlays ``q(x)`` on the exact posterior; the bottom panel shows the
live decomposition of variational free energy (free energy, divergence from the
posterior, complexity, accuracy) so you can *feel* how the terms trade off as you
drag the belief around. Free energy bottoms out exactly when ``q`` sits on the
posterior — at which point divergence is zero and ``F = −log p(y)``.

This script is GUI-only (matplotlib ``Slider`` widgets) and is skipped by the
headless smoke tests.
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider

from active_inference import (
    GaussianBelief,
    GridBayesianInference,
    LinearGaussianModel,
    variational_free_energy,
)
from active_inference.visualizations.style import COLORS


def parse_args() -> argparse.Namespace:
    """Parse command-line options for this executable entry point."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--save", action="store_true",
                   help="accepted for CLI symmetry; this script is interactive")
    p.add_argument("--y", type=float, default=7.0)
    return p.parse_args()


def main() -> None:
    """Run the chapter orchestrator and render or display its outputs."""
    args = parse_args()
    if args.save:  # headless / CI invocation — nothing to render, exit cleanly.
        print("interactive_vfe_explorer is GUI-only; nothing to save.")
        return

    x_grid = np.linspace(-6.0, 12.0, 1201)
    model = LinearGaussianModel(beta0=3.0, beta1=2.0, sigma2_y=0.25,
                                m_x=4.0, s2_x=0.25)
    exact = GridBayesianInference(model=model, x_grid=x_grid).infer(args.y)
    le, post = float(exact.log_evidence), np.asarray(exact.posterior)

    fig, (ax_q, ax_bar) = plt.subplots(2, 1, figsize=(8, 7))
    plt.subplots_adjust(bottom=0.22, hspace=0.35)
    ax_q.plot(x_grid, post, color="black", ls="--", lw=2, label="posterior")
    (line_q,) = ax_q.plot([], [], color=COLORS["posterior"], lw=2, label="q(x)")
    ax_q.set_xlim(0, 5)
    ax_q.set_xlabel("food size x")
    ax_q.set_ylabel("density")
    ax_q.legend(loc="upper right")

    labels = ["F", "divergence", "complexity", "accuracy"]
    bars = ax_bar.bar(labels, [0, 0, 0, 0],
                      color=[COLORS["prior"], COLORS["posterior"],
                             "#9b59b6", "#e67e22"])
    ax_bar.set_ylabel("nats")
    ax_bar.axhline(-le, color=COLORS["likelihood"], ls=":",
                   label=f"−log p(y) = {-le:.2f}")
    ax_bar.legend(loc="upper right", fontsize=9)

    ax_mu = plt.axes([0.15, 0.10, 0.7, 0.03])
    ax_var = plt.axes([0.15, 0.05, 0.7, 0.03])
    s_mu = Slider(ax_mu, r"$\mu_x$", 0.0, 5.0, valinit=4.0)
    s_var = Slider(ax_var, r"$\sigma_x^2$", 0.02, 2.0, valinit=0.25)

    def update(_=None):
        """Update interactive or animated artists for the current state."""
        q = GaussianBelief(s_mu.val, max(s_var.val, 1e-3))
        line_q.set_data(x_grid, q.pdf(x_grid))
        c = variational_free_energy(q, model, args.y, x_grid,
                                    log_evidence=le, posterior=post)
        for bar, val in zip(bars, [c.free_energy, c.divergence,
                                   c.complexity, c.accuracy]):
            bar.set_height(val)
        ax_bar.relim()
        ax_bar.autoscale_view()
        fig.canvas.draw_idle()

    s_mu.on_changed(update)
    s_var.on_changed(update)
    update()
    fig.suptitle("Interactive VFE explorer — drag μ and σ² onto the posterior")
    plt.show()


if __name__ == "__main__":
    main()
