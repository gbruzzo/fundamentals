"""Pedagogical free-energy form decompositions for extras topics.

The book's Part III and Appendix D collect several free-energy variants. This
module keeps the companion implementation deliberately conservative: each
function returns a transparent numeric decomposition rather than claiming a
single universal convention for every literature variant.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


def _finite_scalar(name: str, value: float) -> float:
    """Validate and return a finite scalar used in a free-energy decomposition."""
    scalar = float(value)
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar


def _finite_vector(name: str, value: np.ndarray | list[float]) -> np.ndarray:
    """Validate and return a finite one-dimensional vector."""
    array = np.asarray(value, dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty 1-D vector")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _probabilities(value: np.ndarray | list[float]) -> np.ndarray:
    """Validate and normalize non-negative categorical weights."""
    probs = _finite_vector("probabilities", value)
    if np.any(probs < 0.0):
        raise ValueError("probabilities must be non-negative")
    total = float(np.sum(probs))
    if total <= 0.0:
        raise ValueError("probabilities must have positive mass")
    return probs / total


@dataclass(frozen=True)
class FreeEnergyForm:
    """A named free-energy total with its additive terms.

    Parameters
    ----------
    name : str
        Human-readable name of the form.
    total : float
        Scalar free-energy value.
    terms : Mapping[str, float]
        Additive or signed terms used to construct ``total``.
    """

    name: str
    total: float
    terms: Mapping[str, float]

    def __post_init__(self) -> None:
        """Validate decomposition fields after dataclass initialization."""
        object.__setattr__(self, "total", _finite_scalar("total", self.total))
        cleaned = {str(k): _finite_scalar(str(k), v) for k, v in self.terms.items()}
        if not cleaned:
            raise ValueError("terms must not be empty")
        object.__setattr__(self, "terms", cleaned)

    def term_vector(self, order: tuple[str, ...] | None = None) -> np.ndarray:
        """Return decomposition terms as a numeric vector in a stable order."""
        names = order if order is not None else tuple(sorted(self.terms))
        return np.array([self.terms[name] for name in names], dtype=float)


def expected_free_energy_form(
    risk: float,
    ambiguity: float,
    epistemic_value: float = 0.0,
) -> FreeEnergyForm:
    """Return the common pedagogical EFE form ``G = risk + ambiguity - epistemic``."""
    risk_v = _finite_scalar("risk", risk)
    ambiguity_v = _finite_scalar("ambiguity", ambiguity)
    epistemic_v = _finite_scalar("epistemic_value", epistemic_value)
    total = risk_v + ambiguity_v - epistemic_v
    return FreeEnergyForm(
        "expected_free_energy",
        total,
        {"risk": risk_v, "ambiguity": ambiguity_v, "epistemic_value": -epistemic_v},
    )


def free_energy_of_future(
    present_free_energy: float,
    expected_future_free_energy: float,
) -> FreeEnergyForm:
    """Return a simple FEF bridge as present plus expected future free energy."""
    present = _finite_scalar("present_free_energy", present_free_energy)
    future = _finite_scalar("expected_future_free_energy", expected_future_free_energy)
    return FreeEnergyForm(
        "free_energy_of_future",
        present + future,
        {"present_free_energy": present, "expected_future_free_energy": future},
    )


def observed_predicted_free_energy(
    observed_free_energy: float,
    predicted_free_energy: float,
    observation_weight: float = 0.5,
) -> FreeEnergyForm:
    """Blend observed and predicted free-energy terms with a validated weight."""
    weight = _finite_scalar("observation_weight", observation_weight)
    if weight < 0.0 or weight > 1.0:
        raise ValueError("observation_weight must lie in [0, 1]")
    observed = _finite_scalar("observed_free_energy", observed_free_energy)
    predicted = _finite_scalar("predicted_free_energy", predicted_free_energy)
    total = weight * observed + (1.0 - weight) * predicted
    return FreeEnergyForm(
        "observed_predicted_free_energy",
        total,
        {
            "observed_component": weight * observed,
            "predicted_component": (1.0 - weight) * predicted,
        },
    )


def generalized_free_energy_form(
    present_free_energy: float,
    future_free_energy: float,
    information_gain: float = 0.0,
) -> FreeEnergyForm:
    """Return ``present + future - information_gain`` as a GFE teaching form."""
    present = _finite_scalar("present_free_energy", present_free_energy)
    future = _finite_scalar("future_free_energy", future_free_energy)
    information = _finite_scalar("information_gain", information_gain)
    return FreeEnergyForm(
        "generalized_free_energy",
        present + future - information,
        {
            "present_free_energy": present,
            "future_free_energy": future,
            "information_gain": -information,
        },
    )


def bethe_free_energy_form(
    factor_energy: float,
    variable_entropy: float,
    consistency_penalty: float = 0.0,
) -> FreeEnergyForm:
    """Return a Bethe-style factor energy minus variable entropy decomposition.

    The optional consistency penalty is caller supplied, making this a
    transparent teaching form rather than a full loopy-belief-propagation
    implementation.
    """
    energy = _finite_scalar("factor_energy", factor_energy)
    entropy = _finite_scalar("variable_entropy", variable_entropy)
    penalty = _finite_scalar("consistency_penalty", consistency_penalty)
    return FreeEnergyForm(
        "bethe_free_energy",
        energy - entropy + penalty,
        {
            "factor_energy": energy,
            "negative_variable_entropy": -entropy,
            "consistency_penalty": penalty,
        },
    )


def renyi_bound(
    probabilities: np.ndarray | list[float],
    energies: np.ndarray | list[float],
    alpha: float,
) -> float:
    """Return a Renyi-style exponential certainty-equivalent energy.

    The quantity is ``log E_p[exp((1-alpha) E)] / (1-alpha)``. As
    ``alpha`` approaches one this approaches ``E_p[E]``; callers should use
    :func:`renyi_limit_energy` for that exact limiting case.
    """
    probs = _probabilities(probabilities)
    energy = _finite_vector("energies", energies)
    if energy.shape != probs.shape:
        raise ValueError("probabilities and energies must have the same shape")
    a = _finite_scalar("alpha", alpha)
    if np.isclose(a, 1.0):
        raise ValueError("alpha must differ from 1; use renyi_limit_energy")
    scaled = (1.0 - a) * energy
    shift = float(np.max(scaled))
    moment = float(np.sum(probs * np.exp(scaled - shift)))
    return float((np.log(moment) + shift) / (1.0 - a))


def renyi_limit_energy(
    probabilities: np.ndarray | list[float],
    energies: np.ndarray | list[float],
) -> float:
    """Return the alpha-to-one limit of :func:`renyi_bound`.

    In that limit the Renyi certainty-equivalent collapses to ordinary
    probability-weighted expected energy.
    """
    probs = _probabilities(probabilities)
    energy = _finite_vector("energies", energies)
    if energy.shape != probs.shape:
        raise ValueError("probabilities and energies must have the same shape")
    return float(probs @ energy)


def free_energy_variant_table(
    risk: np.ndarray | list[float],
    ambiguity: np.ndarray | list[float],
    epistemic_value: np.ndarray | list[float],
) -> dict[str, np.ndarray]:
    """Return comparable FEF/EFE/GFE arrays for a policy-indexed sweep."""
    r = _finite_vector("risk", risk)
    a = _finite_vector("ambiguity", ambiguity)
    e = _finite_vector("epistemic_value", epistemic_value)
    if r.shape != a.shape or r.shape != e.shape:
        raise ValueError("risk, ambiguity, and epistemic_value must share a shape")
    efe = r + a - e
    fef = efe + 0.5 * r
    gfe = efe - 0.25 * e
    return {
        "risk": r,
        "ambiguity": a,
        "epistemic_value": e,
        "expected_free_energy": efe,
        "free_energy_of_future": fef,
        "generalized_free_energy": gfe,
    }


__all__ = [
    "FreeEnergyForm",
    "expected_free_energy_form",
    "free_energy_of_future",
    "observed_predicted_free_energy",
    "generalized_free_energy_form",
    "bethe_free_energy_form",
    "renyi_bound",
    "renyi_limit_energy",
    "free_energy_variant_table",
]
