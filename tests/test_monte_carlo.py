"""Tests for Monte Carlo VaR."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from mibel_risk.var.monte_carlo import var_monte_carlo
from mibel_risk.var.parametric import var_parametric_gaussian


def _gaussian_sim(std: float):
    def sim(n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.normal(0.0, std, size=n)

    return sim


def test_reproducible_with_seed() -> None:
    sim = _gaussian_sim(0.02)
    a = var_monte_carlo(sim, n_paths=50_000, alpha=0.95, seed=7)
    b = var_monte_carlo(sim, n_paths=50_000, alpha=0.95, seed=7)
    assert a == b


def test_different_seeds_differ() -> None:
    sim = _gaussian_sim(0.02)
    a = var_monte_carlo(sim, n_paths=50_000, alpha=0.95, seed=1)
    b = var_monte_carlo(sim, n_paths=50_000, alpha=0.95, seed=2)
    assert a != b


@pytest.mark.monte_carlo
def test_converges_to_gaussian_var() -> None:
    """With enough paths, MC VaR matches the analytic Gaussian VaR."""
    std = 0.02
    analytic = var_parametric_gaussian(0.0, std, 0.99)
    mc = var_monte_carlo(_gaussian_sim(std), n_paths=2_000_000, alpha=0.99, seed=0)
    assert mc == pytest.approx(analytic, rel=0.02)


@pytest.mark.monte_carlo
def test_error_shrinks_with_more_paths() -> None:
    """Monte Carlo error should fall as paths increase (O(n^-1/2))."""
    std = 0.02
    analytic = var_parametric_gaussian(0.0, std, 0.95)
    sim = _gaussian_sim(std)
    err_small = abs(var_monte_carlo(sim, n_paths=2_000, alpha=0.95, seed=0) - analytic)
    err_large = abs(var_monte_carlo(sim, n_paths=500_000, alpha=0.95, seed=0) - analytic)
    assert err_large < err_small


def test_return_es_tuple() -> None:
    sim = _gaussian_sim(0.02)
    out = var_monte_carlo(sim, n_paths=100_000, alpha=0.95, seed=3, return_es=True)
    assert isinstance(out, tuple)
    var, es = out
    assert es >= var


def test_path_agnostic_student_t() -> None:
    """Any generator plugs in: a Student-t sampler gives a fatter-tail VaR."""
    def t_sim(n: int, rng: np.random.Generator) -> np.ndarray:
        return stats.t.rvs(df=4, scale=0.02, size=n, random_state=rng)

    def g_sim(n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.normal(0.0, 0.02, size=n)

    v_t = var_monte_carlo(t_sim, n_paths=500_000, alpha=0.99, seed=0)
    v_g = var_monte_carlo(g_sim, n_paths=500_000, alpha=0.99, seed=0)
    assert v_t > v_g


def test_invalid_n_paths() -> None:
    with pytest.raises(ValueError):
        var_monte_carlo(_gaussian_sim(0.02), n_paths=0, alpha=0.95)


def test_simulate_fn_wrong_length() -> None:
    def bad_sim(n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.normal(0.0, 0.02, size=n - 1)

    with pytest.raises(ValueError):
        var_monte_carlo(bad_sim, n_paths=100, alpha=0.95, seed=0)
