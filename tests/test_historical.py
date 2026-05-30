"""Tests for historical-simulation VaR and Expected Shortfall."""

from __future__ import annotations

import numpy as np
import pytest

from mibel_risk.var.historical import (
    expected_shortfall_historical,
    var_historical,
)


@pytest.fixture
def returns() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.normal(0.0, 0.02, size=10_000)


def test_var_es_ordering(returns: np.ndarray) -> None:
    """ES averages the tail beyond VaR, so ES >= VaR at the same alpha."""
    var = var_historical(returns, alpha=0.95)
    es = expected_shortfall_historical(returns, alpha=0.95)
    assert es >= var


def test_var_monotonic_in_alpha(returns: np.ndarray) -> None:
    """Higher confidence => more extreme quantile => larger VaR."""
    var_90 = var_historical(returns, alpha=0.90)
    var_95 = var_historical(returns, alpha=0.95)
    var_99 = var_historical(returns, alpha=0.99)
    assert var_90 < var_95 < var_99


def test_es_monotonic_in_alpha(returns: np.ndarray) -> None:
    es_95 = expected_shortfall_historical(returns, alpha=0.95)
    es_99 = expected_shortfall_historical(returns, alpha=0.99)
    assert es_95 < es_99


def test_reproducibility(returns: np.ndarray) -> None:
    """Pure function of its input: identical inputs give identical outputs."""
    assert var_historical(returns, 0.95) == var_historical(returns.copy(), 0.95)
    assert expected_shortfall_historical(returns, 0.95) == expected_shortfall_historical(
        returns.copy(), 0.95
    )


def test_recovers_known_quantile() -> None:
    """For a uniform grid the (1-alpha) empirical quantile is exact."""
    data = np.linspace(-1.0, 1.0, 1001)  # symmetric, step 0.002
    var = var_historical(data, alpha=0.95)
    assert var == pytest.approx(0.90, abs=1e-9)


def test_nan_handling() -> None:
    clean = np.array([-0.05, -0.02, 0.0, 0.03, 0.04])
    dirty = np.array([-0.05, np.nan, -0.02, 0.0, np.nan, 0.03, 0.04])
    assert var_historical(dirty, 0.8) == var_historical(clean, 0.8)


def test_invalid_alpha_raises(returns: np.ndarray) -> None:
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            var_historical(returns, alpha=bad)


def test_empty_returns_raises() -> None:
    with pytest.raises(ValueError):
        var_historical(np.array([]), 0.95)
    with pytest.raises(ValueError):
        var_historical(np.array([np.nan, np.nan]), 0.95)
