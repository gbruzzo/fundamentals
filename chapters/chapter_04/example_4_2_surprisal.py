"""Example 4.2 — Surprisal and the marginal likelihood (§4.3).

Run::

    python chapters/chapter_04/example_4_2_surprisal.py [--save]

Reproduces Figure 4.3.1. Surprisal is ``−log p(y)``: a high-probability (expected)
observation has *low* surprisal, a low-probability (unexpected) observation has
*high* surprisal. Variational free energy is an upper bound on surprisal
(Eq. 26), so minimizing VFE pushes the agent toward a model under which its
sensory data are unsurprising — equivalently, toward higher model evidence.
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from active_inference import (
    GridBayesianInference,
    LinearGaussianModel,
    get_logger,
    surprisal,
)
from active_inference.utils.io import default_figure_dir, ensure_dir
from active_inference.visualizations import plot_surprisal_relationship, save_or_show

LOG = get_logger("ch4.ex2")


def parse_args() -> argparse.Namespace:
    """Parse command-line options for this executable entry point."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--save", action="store_true")
    return p.parse_args()


def main() -> None:
    """Run the chapter orchestrator and render or display its outputs."""
    args = parse_args()
    fig = plot_surprisal_relationship(mu=0.0, sigma2=0.64,
                                      title="Example 4.2 · evidence p(y) vs surprisal")

    # Numerical sanity: a typical observation is less surprising than an outlier.
    x_grid = np.linspace(-6.0, 12.0, 2001)
    model = LinearGaussianModel(beta0=3.0, beta1=2.0, sigma2_y=0.25,
                                m_x=4.0, s2_x=0.25)
    inf = GridBayesianInference(model=model, x_grid=x_grid)
    for y in (7.0, 0.0):
        s = surprisal(model, y, x_grid)
        LOG.info("ŷ=%.1f → log p(y)=%.3f  surprisal=%.3f",
                 y, inf.infer(y).log_evidence, s)

    if args.save:
        out = ensure_dir(default_figure_dir() / "chapter_04")
        save_or_show(fig, out / "example_4_2_surprisal.png")
        LOG.info("Saved to %s", out)
    else:
        plt.show()


if __name__ == "__main__":
    main()
