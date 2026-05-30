"""Tests for parametric (Gaussian / Student-t) VaR and ES."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from mibel_risk.var.parametric import (
    expected_shortfall_gaussian,
    expected_shortfall_t,
    var_parametric_gaussian,
    var_parametric_t,
)


@pytest.mark.parametrize("alpha", [0.90, 0.95, 0.99])
def test_gaussian_matches_scipy(alpha: float) -> None:
    """Closed form must equal sigma * norm.ppf(alpha) - mean."""
    mean, std = 0.001, 0.02
    expected = std * stats.norm.ppf(alpha) - mean
    assert var_parametric_gaussian(mean, std, alpha) == pytest.approx(expected, rel=1e-12)


def test_gaussian_zero_mean_zero_std() -> None:
    assert var_parametric_gaussian(0.0, 0.0, 0.95) == pytest.approx(0.0)


@pytest.mark.parametrize("alpha", [0.90, 0.95, 0.99])
def test_t_matches_scipy(alpha: float) -> None:
    """t-VaR equals the variance-rescaled t quantile minus the mean."""
    mean, std, df = 0.0, 0.02, 6.0
    scale = std * np.sqrt((df - 2.0) / df)
    expected = scale * stats.t.ppf(alpha, df) - mean
    assert var_parametric_t(mean, std, df, alpha) == pytest.approx(expected, rel=1e-12)


def test_t_heavier_tail_than_gaussian() -> None:
    """At high confidence the Student-t tail dominates the Gaussian one."""
    mean, std = 0.0, 0.02
    g = var_parametric_gaussian(mean, std, 0.99)
    t = var_parametric_t(mean, std, df=4.0, alpha=0.99)
    assert t > g


def test_t_converges_to_gaussian_large_df() -> None:
    """As df -> infinity the Student-t VaR approaches the Gaussian VaR."""
    mean, std = 0.0, 0.02
    g = var_parametric_gaussian(mean, std, 0.95)
    t = var_parametric_t(mean, std, df=500.0, alpha=0.95)
    assert t == pytest.approx(g, rel=2e-3)


def test_es_ge_var_gaussian() -> None:
    mean, std = 0.0, 0.02
    var = var_parametric_gaussian(mean, std, 0.95)
    es = expected_shortfall_gaussian(mean, std, 0.95)
    assert es > var


def test_es_ge_var_t() -> None:
    mean, std, df = 0.0, 0.02, 6.0
    var = var_parametric_t(mean, std, df, 0.95)
    es = expected_shortfall_t(mean, std, df, 0.95)
    assert es > var


def test_es_t_converges_to_es_gaussian_large_df() -> None:
    mean, std = 0.0, 0.02
    es_g = expected_shortfall_gaussian(mean, std, 0.975)
    es_t = expected_shortfall_t(mean, std, df=1000.0, alpha=0.975)
    assert es_t == pytest.approx(es_g, rel=5e-3)


def test_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        var_parametric_gaussian(0.0, -1.0, 0.95)  # negative std
    with pytest.raises(ValueError):
        var_parametric_gaussian(0.0, 0.02, 1.5)  # alpha out of range
    with pytest.raises(ValueError):
        var_parametric_t(0.0, 0.02, df=2.0, alpha=0.95)  # df not > 2
