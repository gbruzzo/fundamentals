"""Example 7.5 — multivariate active generalized filtering in generalized coordinates.

Run::

    python chapters/chapter_07/example_7_5_multivariate_active_inference.py [--save]

A 2-D environment is pulled toward an exogenous attractor ``v* = [6, -4]``. The
agent's generative model prefers ``v = [0, 0]`` and acts through a vector control
channel. The figure contrasts the no-action baseline with active control and shows
the coupled perception/action traces from Algorithm 7.5.1.
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from active_inference import (
    GeneralizedVectorModel,
    LinearVectorFunction,
    MultivariateActiveEnvironment,
    MultivariateActiveInferenceAgent,
    correlated_embedding_precision,
    get_logger,
    save_chapter_data,
    simulate_multivariate_active_inference,
)
from active_inference.utils.io import default_figure_dir, ensure_dir
from active_inference.visualizations import plot_multivariate_active_inference, save_or_show

LOG = get_logger("ch7.ex5")

PREFERENCE = np.array([0.0, 0.0])
EXOGENOUS = np.array([6.0, -4.0])


def build_agent_env(kappa_a: float, *, embedding_dim: int, gamma: float):
    """Create the vector agent and environment used in the Chapter 7 §7.5 example."""
    model = GeneralizedVectorModel(
        f=LinearVectorFunction(-np.eye(2), PREFERENCE),
        g=LinearVectorFunction(np.eye(2), np.zeros(2)),
        precision_x=correlated_embedding_precision(np.eye(2), embedding_dim, gamma=gamma),
        precision_y=correlated_embedding_precision(np.eye(2) * 20.0, embedding_dim, gamma=gamma),
        embedding_dim=embedding_dim,
        dim_x=2,
        dim_y=2,
    )
    forward = np.zeros((embedding_dim * 2, 2))
    forward[[0, embedding_dim], :] = np.eye(2)
    agent = MultivariateActiveInferenceAgent(
        perception_model=model,
        forward_model=forward,
        kappa_x=0.4,
        kappa_a=kappa_a,
    )
    env = MultivariateActiveEnvironment(
        drift=LinearVectorFunction(-np.eye(2), EXOGENOUS),
        g=LinearVectorFunction(np.eye(2), np.zeros(2)),
        action_matrix=np.eye(2),
        omega_x=0.0,
        omega_y=0.0,
    )
    return agent, env


def parse_args() -> argparse.Namespace:
    """Parse command-line options for Example 7.5."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-steps", type=int, default=2500)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--embedding-dim", type=int, default=2)
    parser.add_argument("--gamma", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    """Run vector active inference and render or save its figure/data."""
    args = parse_args()
    action_start = args.n_steps // 5
    agent, env = build_agent_env(0.5, embedding_dim=args.embedding_dim, gamma=args.gamma)
    passive_agent, _, = build_agent_env(0.0, embedding_dim=args.embedding_dim, gamma=args.gamma)
    active = simulate_multivariate_active_inference(
        agent,
        env,
        x0=np.array([4.0, -2.0]),
        mu0_tilde=np.zeros((args.embedding_dim, 2)),
        n_steps=args.n_steps,
        dt=args.dt,
        action_start=action_start,
        rng=np.random.default_rng(args.seed),
    )
    passive = simulate_multivariate_active_inference(
        passive_agent,
        env,
        x0=np.array([4.0, -2.0]),
        mu0_tilde=np.zeros((args.embedding_dim, 2)),
        n_steps=args.n_steps,
        dt=args.dt,
        action_start=action_start,
        rng=np.random.default_rng(args.seed),
    )
    LOG.info(
        "settled preference error: active=%.4f | no-action=%.4f",
        active.preference_error(PREFERENCE),
        passive.preference_error(PREFERENCE),
    )
    fig = plot_multivariate_active_inference(
        active,
        preference=PREFERENCE,
        exogenous=EXOGENOUS,
        baseline=passive,
        dt=args.dt,
    )
    if args.save:
        out = ensure_dir(default_figure_dir() / "chapter_07")
        figure = out / "example_7_5_multivariate_active_inference.png"
        save_or_show(fig, figure)
        save_chapter_data(
            7,
            figure.stem,
            {
                "time": np.arange(active.xs.shape[0]) * args.dt,
                "active_xs": active.xs,
                "active_beliefs": active.mus,
                "active_actions": active.actions,
                "active_observations": active.ys,
                "active_generalized_measurements": active.y_tildes,
                "active_eps_y": active.eps_y,
                "active_free_energy": active.free_energies,
                "passive_xs": passive.xs,
                "passive_beliefs": passive.mus,
                "passive_free_energy": passive.free_energies,
                "preference": PREFERENCE,
                "exogenous": EXOGENOUS,
            },
            {"seed": args.seed, "dt": args.dt, "gamma": args.gamma, "action_start": action_start},
            figures=[figure],
        )
        LOG.info("Saved Example 7.5 to %s", figure)
    else:
        save_or_show(fig, None)
        plt.show()


if __name__ == "__main__":
    main()
