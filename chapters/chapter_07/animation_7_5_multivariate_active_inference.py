"""Animation — Chapter 7 §7.5 multivariate active generalized filtering.

Run::

    python chapters/chapter_07/animation_7_5_multivariate_active_inference.py [--save]

The GIF shows the 2-D active-inference loop in motion: the external state leaves the
exogenous attractor and approaches the preference as vector action suppresses
generalized sensory prediction error.
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
from active_inference.visualizations import animate_multivariate_active_inference, save_animation

LOG = get_logger("ch7.anim5")

PREFERENCE = np.array([0.0, 0.0])
EXOGENOUS = np.array([6.0, -4.0])


def build_agent_env(kappa_a: float, *, embedding_dim: int, gamma: float):
    """Create the vector agent and environment used in the animation."""
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
    agent = MultivariateActiveInferenceAgent(model, forward_model=forward, kappa_x=0.4, kappa_a=kappa_a)
    env = MultivariateActiveEnvironment(
        drift=LinearVectorFunction(-np.eye(2), EXOGENOUS),
        g=LinearVectorFunction(np.eye(2), np.zeros(2)),
        action_matrix=np.eye(2),
        omega_x=0.0,
        omega_y=0.0,
    )
    return agent, env


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the Chapter 7 §7.5 animation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-steps", type=int, default=1600)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--embedding-dim", type=int, default=2)
    parser.add_argument("--gamma", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    """Create and optionally save the multivariate active-inference GIF."""
    args = parse_args()
    action_start = args.n_steps // 5
    agent, env = build_agent_env(0.5, embedding_dim=args.embedding_dim, gamma=args.gamma)
    result = simulate_multivariate_active_inference(
        agent,
        env,
        x0=np.array([4.0, -2.0]),
        mu0_tilde=np.zeros((args.embedding_dim, 2)),
        n_steps=args.n_steps,
        dt=args.dt,
        action_start=action_start,
        rng=np.random.default_rng(args.seed),
    )
    anim = animate_multivariate_active_inference(
        result,
        preference=PREFERENCE,
        exogenous=EXOGENOUS,
        dt=args.dt,
        frame_stride=25,
        title="Fig. 7.5 animated · vector action-perception loop",
    )
    if args.save:
        out = "output/figures/chapter_07/animation_7_5_multivariate_active_inference.gif"
        save_animation(anim, out, fps=12, dpi=110)
        save_chapter_data(
            7,
            "animation_7_5_multivariate_active_inference",
            {
                "time": np.arange(result.xs.shape[0]) * args.dt,
                "xs": result.xs,
                "beliefs": result.mus,
                "actions": result.actions,
                "generalized_measurements": result.y_tildes,
                "eps_y": result.eps_y,
                "free_energy": result.free_energies,
                "preference": PREFERENCE,
                "exogenous": EXOGENOUS,
            },
            {"seed": args.seed, "dt": args.dt, "gamma": args.gamma, "action_start": action_start},
            figures=[out],
        )
        LOG.info("Saved Chapter 7 §7.5 animation to %s", out)
    else:
        plt.show()


if __name__ == "__main__":
    main()
