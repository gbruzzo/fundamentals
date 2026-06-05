"""Tests for categorical factor-graph helper functions."""

from __future__ import annotations

import numpy as np
import pytest

from active_inference import (
    categorical_factor_message,
    normalize_message,
    sum_product_chain,
    variational_message_update,
)


def test_normalize_message_returns_categorical_vector() -> None:
    """Message normalization preserves ratios and sums to one."""
    msg = normalize_message([2.0, 1.0, 1.0])
    assert msg.sum() == pytest.approx(1.0)
    assert msg.tolist() == pytest.approx([0.5, 0.25, 0.25])
    with pytest.raises(ValueError):
        normalize_message([0.0, 0.0])


def test_categorical_factor_message_sums_out_non_target_axes() -> None:
    """Factor-to-variable messages agree with direct marginalization."""
    factor = np.array([[0.9, 0.1], [0.2, 0.8]])
    incoming = [np.array([0.75, 0.25])]
    msg = categorical_factor_message(factor, incoming, target_axis=0)
    direct = normalize_message(factor @ normalize_message(incoming[0]))
    assert msg.tolist() == pytest.approx(direct.tolist())


def test_sum_product_chain_filters_each_time_step() -> None:
    """Forward messages remain normalized over a categorical state chain."""
    prior = np.array([0.6, 0.4])
    transition = np.array([[0.8, 0.3], [0.2, 0.7]])
    likelihoods = np.array([[0.9, 0.1], [0.4, 0.7], [0.2, 0.9]])
    beliefs = sum_product_chain(prior, transition, likelihoods)
    assert beliefs.shape == (3, 2)
    assert np.allclose(beliefs.sum(axis=1), 1.0)
    assert beliefs[-1, 1] > beliefs[0, 1]


def test_variational_message_update_softmaxes_expected_log_factor() -> None:
    """VMP-style update returns a normalized categorical message."""
    log_factor = np.log(np.array([[0.8, 0.2], [0.1, 0.9]]))
    msg = variational_message_update(log_factor, [np.array([0.5, 0.5])], target_axis=0)
    assert msg.shape == (2,)
    assert msg.sum() == pytest.approx(1.0)
