"""Tests for VaR backtesting (Kupiec POF, Christoffersen)."""

from __future__ import annotations

import numpy as np
import pytest

from mibel_risk.backtesting.christoffersen import (
    christoffersen_conditional_coverage,
    christoffersen_independence,
)
from mibel_risk.backtesting.kupiec import kupiec_pof


def _bernoulli_exceptions(n: int, rate: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random(n) < rate


def test_kupiec_accepts_well_calibrated_model() -> None:
    """Exceptions drawn at exactly the nominal rate should not be rejected."""
    exc = _bernoulli_exceptions(5_000, rate=0.05, seed=0)
    res = kupiec_pof(exc, alpha=0.95)
    assert not res.reject
    assert res.p_value > 0.05


def test_kupiec_rejects_underestimated_risk() -> None:
    """Far too many exceptions for a 99% VaR model => reject."""
    exc = _bernoulli_exceptions(5_000, rate=0.05, seed=0)
    res = kupiec_pof(exc, alpha=0.99)  # model claims 1% but realises ~5%
    assert res.reject
    assert res.p_value < 0.01


def test_kupiec_zero_exceptions_branch() -> None:
    exc = np.zeros(1_000, dtype=bool)
    res = kupiec_pof(exc, alpha=0.95)
    assert res.n_exceptions == 0
    assert res.lr_stat >= 0.0
    assert res.expected_exceptions == pytest.approx(50.0)


def test_kupiec_counts() -> None:
    exc = np.array([True, False, True, False, False])
    res = kupiec_pof(exc, alpha=0.80)
    assert res.n_obs == 5
    assert res.n_exceptions == 2


def test_christoffersen_independence_accepts_iid() -> None:
    exc = _bernoulli_exceptions(5_000, rate=0.05, seed=1)
    res = christoffersen_independence(exc)
    assert not res.reject


def test_christoffersen_independence_rejects_clustered() -> None:
    """A clearly clustered series (long runs) breaks independence."""
    block = np.array([True] * 30 + [False] * 270)
    exc = np.tile(block, 10)
    res = christoffersen_independence(exc)
    assert res.reject


def test_christoffersen_independence_degenerate_no_exceptions() -> None:
    exc = np.zeros(500, dtype=bool)
    res = christoffersen_independence(exc)
    assert res.lr_stat == 0.0
    assert not res.reject


def test_conditional_coverage_df_and_accept() -> None:
    exc = _bernoulli_exceptions(5_000, rate=0.05, seed=2)
    res = christoffersen_conditional_coverage(exc, alpha=0.95)
    assert res.df == 2
    assert not res.reject


def test_conditional_coverage_rejects_bad_model() -> None:
    exc = _bernoulli_exceptions(5_000, rate=0.10, seed=3)
    res = christoffersen_conditional_coverage(exc, alpha=0.99)
    assert res.reject


def test_independence_requires_two_obs() -> None:
    with pytest.raises(ValueError):
        christoffersen_independence(np.array([True]))
