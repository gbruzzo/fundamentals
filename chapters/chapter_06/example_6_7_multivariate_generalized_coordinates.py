"""Example 6.7 — multivariate generalized filtering in generalized coordinates (§6.6).

Run::

    python chapters/chapter_06/example_6_7_multivariate_generalized_coordinates.py [--save]

This repeats the Hooke's-law oscillator from Example 6.2, but the agent now filters
generalized measurements ``ỹ`` and uses correlated embedding-order precisions
``Π̃(γ)``. The figure compares the ordinary vector filter's lag against the
generalized-coordinate estimate, then shows recovered first-order motion and VFE.
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from active_inference import (
    GeneralizedVectorModel,
    LinearVectorFunction,
    MultivariateDynamicModel,
    MultivariateDynamicProcess,
    correlated_embedding_precision,
    generalized_measurements_from_series,
    generalized_vector_filter,
    get_logger,
    multivariate_generalized_filter,
    save_chapter_data,
    simulate_multivariate_process,
)
from active_inference.utils.io import default_figure_dir, ensure_dir
from active_inference.visualizations import plot_generalized_vector_filter, save_or_show

LOG = get_logger("ch6.ex7")

K, MASS, V0, THETA_Y = 4.0, 3.0, 5.0, 3.0


def hooke_functions() -> tuple[LinearVectorFunction, LinearVectorFunction]:
    """Return Hooke's-law flow and shifted identity observation functions."""
    a_f = np.array([[0.0, 1.0], [-K / MASS, 0.0]])
    b_f = np.array([0.0, (K / MASS) * V0])
    return LinearVectorFunction(a_f, b_f), LinearVectorFunction(np.eye(2), np.array([-THETA_Y, -THETA_Y]))


def parse_args() -> argparse.Namespace:
    """Parse command-line options for Example 6.7."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-steps", type=int, default=1000)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--embedding-dim", type=int, default=3)
    parser.add_argument("--gamma", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    """Run Example 6.7 and render or save the educational figure."""
    args = parse_args()
    f, g = hooke_functions()
    process = MultivariateDynamicProcess(f=f, g=g, omega_x=0.0, omega_y=0.0)
    trace = simulate_multivariate_process(
        process,
        x0=np.array([0.0, 5.0]),
        n_steps=args.n_steps,
        dt=args.dt,
        rng=np.random.default_rng(args.seed),
    )
    base_model = MultivariateDynamicModel(f=f, g=g, precision_x=0.5, precision_y=10.0, dim_x=2, dim_y=2)
    y_tilde = generalized_measurements_from_series(trace.ys, embedding_dim=args.embedding_dim, dt=args.dt)
    model = GeneralizedVectorModel(
        f=f,
        g=g,
        precision_x=correlated_embedding_precision(np.eye(2) * 0.5, args.embedding_dim, gamma=args.gamma),
        precision_y=correlated_embedding_precision(np.eye(2) * 10.0, args.embedding_dim, gamma=args.gamma),
        embedding_dim=args.embedding_dim,
        dim_x=2,
        dim_y=2,
    )
    ordinary = multivariate_generalized_filter(base_model, trace.ys, dt=args.dt, kappa=1.0, mu0=np.array([8.0, 8.0]))
    generalized = generalized_vector_filter(model, y_tilde, dt=args.dt, kappa=1.0, mu0_tilde=np.zeros((args.embedding_dim, 2)))
    burn = args.n_steps // 3
    LOG.info(
        "Example 6.7 tracking error: ordinary=%.4f | generalized=%.4f",
        ordinary.tracking_error(trace.xs, burn_in=burn),
        generalized.tracking_error(trace.xs, burn_in=burn),
    )

    fig = plot_generalized_vector_filter(generalized, trace.xs, ordinary=ordinary, dt=args.dt)
    if args.save:
        out = ensure_dir(default_figure_dir() / "chapter_06")
        figure = out / "example_6_7_multivariate_generalized_coordinates.png"
        save_or_show(fig, figure)
        save_chapter_data(
            6,
            figure.stem,
            {
                "time": np.arange(trace.xs.shape[0]) * args.dt,
                "x_true": trace.xs,
                "observations": trace.ys,
                "generalized_measurements": y_tilde,
                "ordinary_beliefs": ordinary.mus,
                "generalized_beliefs": generalized.mus,
                "generalized_eps_x": generalized.eps_x,
                "generalized_eps_y": generalized.eps_y,
                "ordinary_free_energy": ordinary.free_energies,
                "generalized_free_energy": generalized.free_energies,
                "precision_x": model.Pi_x,
                "precision_y": model.Pi_y,
            },
            {"seed": args.seed, "dt": args.dt, "gamma": args.gamma, "embedding_dim": args.embedding_dim},
            figures=[figure],
        )
        LOG.info("Saved Example 6.7 to %s", figure)
    else:
        save_or_show(fig, None)
        plt.show()


if __name__ == "__main__":
    main()
