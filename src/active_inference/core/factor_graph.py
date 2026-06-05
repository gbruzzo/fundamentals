"""Small categorical factor-graph helpers used by extras orchestrators."""

from __future__ import annotations

import numpy as np


def _finite_array(name: str, value: np.ndarray | list[float]) -> np.ndarray:
    """Validate and return a finite numeric array."""
    array = np.asarray(value, dtype=float)
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def normalize_message(message: np.ndarray | list[float]) -> np.ndarray:
    """Normalize a non-negative one-dimensional categorical message.

    The helper rejects empty, negative, non-finite, or zero-mass messages before
    returning a unit-sum vector.
    """
    msg = _finite_array("message", message)
    if msg.ndim != 1:
        raise ValueError("message must be one-dimensional")
    if np.any(msg < 0.0):
        raise ValueError("message must be non-negative")
    total = float(np.sum(msg))
    if total <= 0.0:
        raise ValueError("message must have positive mass")
    return msg / total


def categorical_factor_message(
    factor: np.ndarray | list[float],
    incoming: list[np.ndarray] | tuple[np.ndarray, ...],
    target_axis: int,
) -> np.ndarray:
    """Compute a sum-product message from a categorical factor to one variable.

    ``incoming`` must provide one message for every factor axis except
    ``target_axis``. Axes are multiplied by the corresponding incoming message
    and then all non-target axes are summed out.
    """
    fac = _finite_array("factor", factor)
    axis = int(target_axis)
    if axis < 0:
        axis += fac.ndim
    if axis < 0 or axis >= fac.ndim:
        raise ValueError("target_axis is out of bounds")
    if len(incoming) != fac.ndim - 1:
        raise ValueError("incoming must have one message for every non-target axis")
    work = fac.copy()
    incoming_i = 0
    for dim in range(fac.ndim):
        if dim == axis:
            continue
        msg = normalize_message(incoming[incoming_i])
        if msg.shape[0] != fac.shape[dim]:
            raise ValueError("incoming message length does not match factor axis")
        shape = [1] * fac.ndim
        shape[dim] = msg.shape[0]
        work = work * msg.reshape(shape)
        incoming_i += 1
    summed = np.sum(work, axis=tuple(dim for dim in range(fac.ndim) if dim != axis))
    return normalize_message(summed)


def sum_product_chain(
    prior: np.ndarray | list[float],
    transitions: np.ndarray | list[float],
    likelihoods: np.ndarray | list[float],
) -> np.ndarray:
    """Run normalized forward messages for a categorical state-space chain.

    Parameters
    ----------
    prior : array, shape (S,)
        Initial state message.
    transitions : array, shape (T-1, S, S) or (S, S)
        Column-stochastic transition matrices ``P(s_t | s_{t-1})``.
    likelihoods : array, shape (T, S)
        Per-time likelihood messages over states.
    """
    belief = normalize_message(prior)
    like = _finite_array("likelihoods", likelihoods)
    if like.ndim != 2:
        raise ValueError("likelihoods must have shape (T, S)")
    if like.shape[1] != belief.size:
        raise ValueError("likelihood state dimension must match prior")
    trans = _finite_array("transitions", transitions)
    if trans.ndim == 2:
        trans = np.repeat(trans[None, :, :], max(0, like.shape[0] - 1), axis=0)
    if trans.ndim != 3 or trans.shape[1:] != (belief.size, belief.size):
        raise ValueError("transitions must have shape (T-1, S, S)")
    if trans.shape[0] != max(0, like.shape[0] - 1):
        raise ValueError("transition count must be T-1")
    out = []
    for t in range(like.shape[0]):
        if t > 0:
            belief = normalize_message(trans[t - 1] @ belief)
        belief = normalize_message(belief * normalize_message(like[t]))
        out.append(belief)
    return np.asarray(out)


def variational_message_update(
    log_factor: np.ndarray | list[float],
    incoming_expectations: list[np.ndarray] | tuple[np.ndarray, ...],
    target_axis: int,
) -> np.ndarray:
    """Compute a categorical VMP-style softmax update for one variable.

    Incoming expectations weight every non-target axis before marginalization,
    and the returned categorical vector is normalized through
    :func:`normalize_message`.
    """
    log_fac = _finite_array("log_factor", log_factor)
    axis = int(target_axis)
    if axis < 0:
        axis += log_fac.ndim
    if axis < 0 or axis >= log_fac.ndim:
        raise ValueError("target_axis is out of bounds")
    if len(incoming_expectations) != log_fac.ndim - 1:
        raise ValueError("incoming_expectations must cover non-target axes")
    expected = log_fac.copy()
    incoming_i = 0
    for dim in range(log_fac.ndim):
        if dim == axis:
            continue
        msg = normalize_message(incoming_expectations[incoming_i])
        if msg.shape[0] != log_fac.shape[dim]:
            raise ValueError("incoming expectation length does not match factor axis")
        shape = [1] * log_fac.ndim
        shape[dim] = msg.shape[0]
        expected = expected * msg.reshape(shape)
        incoming_i += 1
    logits = np.sum(expected, axis=tuple(dim for dim in range(log_fac.ndim) if dim != axis))
    logits = logits - float(np.max(logits))
    probs = np.exp(logits)
    return normalize_message(probs)


__all__ = [
    "normalize_message",
    "categorical_factor_message",
    "sum_product_chain",
    "variational_message_update",
]
