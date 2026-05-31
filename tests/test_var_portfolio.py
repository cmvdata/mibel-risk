"""Tests for portfolio VaR and its component/marginal decomposition.

Scenarios here are simple 2-factor draws: each scenario is a length-2 array of
market factors, and a position's ``valuation_fn`` reads one factor as its
per-unit P&L. This keeps the correlation structure explicit and analytically
tractable (jointly Gaussian factors -> VaR is sub-additive and the covariance
split is exact), which is exactly what the properties below assert.
"""

from __future__ import annotations

import numpy as np
import pytest

from mibel_risk.portfolio import (
    Position,
    component_var,
    marginal_var,
    portfolio_pnl,
    portfolio_var_historical,
    portfolio_var_monte_carlo,
    risk_decomposition,
)


def _correlated_factors(
    n: int, rho: float, sigmas: tuple[float, float], seed: int
) -> np.ndarray:
    """Draw ``n`` two-factor scenarios with given correlation and vols."""
    rng = np.random.default_rng(seed)
    cov = np.array(
        [
            [sigmas[0] ** 2, rho * sigmas[0] * sigmas[1]],
            [rho * sigmas[0] * sigmas[1], sigmas[1] ** 2],
        ]
    )
    return rng.multivariate_normal(mean=[0.0, 0.0], cov=cov, size=n)


def _factor0(scenario: np.ndarray) -> float:
    return float(scenario[0])


def _factor1(scenario: np.ndarray) -> float:
    return float(scenario[1])


@pytest.fixture
def book() -> list[Position]:
    """A two-position book: a long forward and a long swing on two factors."""
    return [
        Position("forward", notional=100.0, direction="long", valuation_fn=_factor0),
        Position("swing_call", notional=250.0, direction="long", valuation_fn=_factor1),
    ]


@pytest.fixture
def scenarios() -> np.ndarray:
    """50k correlated (rho=0.3) two-factor scenarios."""
    return _correlated_factors(50_000, rho=0.3, sigmas=(1.0, 1.0), seed=7)


# --------------------------------------------------------------------------- #
# P&L aggregation                                                             #
# --------------------------------------------------------------------------- #
def test_portfolio_pnl_sums_positions(book: list[Position]) -> None:
    """Portfolio P&L is the signed-notional-weighted sum of the legs."""
    scenario = np.array([2.0, -1.0])
    expected = 100.0 * 2.0 + 250.0 * (-1.0)
    assert portfolio_pnl(book, scenario) == pytest.approx(expected)


def test_direction_short_flips_sign() -> None:
    """A short position negates its leg's contribution."""
    long_pos = Position("forward", 100.0, "long", _factor0)
    short_pos = Position("forward", 100.0, "short", _factor0)
    scenario = np.array([3.0, 0.0])
    assert long_pos.pnl(scenario) == pytest.approx(-short_pos.pnl(scenario))


# --------------------------------------------------------------------------- #
# Sub-additivity                                                              #
# --------------------------------------------------------------------------- #
def test_subadditivity(book: list[Position], scenarios: np.ndarray) -> None:
    """Portfolio VaR <= sum of standalone VaRs (diversification)."""
    total = portfolio_var_historical(book, scenarios, alpha=0.95)
    standalone = [
        portfolio_var_historical([pos], scenarios, alpha=0.95) for pos in book
    ]
    assert total <= sum(standalone)
    # rho=0.3 with comparable legs is a genuine diversification gap, not noise.
    assert total < 0.99 * sum(standalone)


def test_perfect_correlation_is_additive() -> None:
    """With identical factors (rho=1) VaR is (near) additive -- no diversification."""
    factors = _correlated_factors(50_000, rho=1.0, sigmas=(1.0, 1.0), seed=11)
    book = [
        Position("forward", 100.0, "long", _factor0),
        Position("forward", 100.0, "long", _factor1),
    ]
    total = portfolio_var_historical(book, factors, alpha=0.95)
    standalone = sum(
        portfolio_var_historical([pos], factors, alpha=0.95) for pos in book
    )
    assert total == pytest.approx(standalone, rel=1e-6)


# --------------------------------------------------------------------------- #
# Linearity in notional                                                       #
# --------------------------------------------------------------------------- #
def test_var_linear_in_notional(scenarios: np.ndarray) -> None:
    """Scaling a single position's notional scales its VaR by the same factor."""
    base = Position("forward", 100.0, "long", _factor0)
    doubled = Position("forward", 200.0, "long", _factor0)
    var_base = portfolio_var_historical([base], scenarios, alpha=0.95)
    var_doubled = portfolio_var_historical([doubled], scenarios, alpha=0.95)
    assert var_doubled == pytest.approx(2.0 * var_base, rel=1e-12)


def test_portfolio_var_scales_when_all_notionals_scale(
    book: list[Position], scenarios: np.ndarray
) -> None:
    """Doubling every notional doubles the portfolio VaR."""
    scaled = [
        Position(p.asset_type, 2.0 * p.notional, p.direction, p.valuation_fn)
        for p in book
    ]
    base = portfolio_var_historical(book, scenarios, alpha=0.95)
    twice = portfolio_var_historical(scaled, scenarios, alpha=0.95)
    assert twice == pytest.approx(2.0 * base, rel=1e-12)


# --------------------------------------------------------------------------- #
# Component / marginal / total consistency                                    #
# --------------------------------------------------------------------------- #
def test_component_sums_to_total(book: list[Position], scenarios: np.ndarray) -> None:
    """Component VaRs add back to the portfolio VaR (Euler additivity)."""
    total = portfolio_var_historical(book, scenarios, alpha=0.95)
    comp = component_var(book, scenarios, alpha=0.95)
    assert comp.sum() == pytest.approx(total, rel=1e-10)


def test_component_equals_notional_times_marginal(
    book: list[Position], scenarios: np.ndarray
) -> None:
    """The homogeneity identity component_i = notional_i * marginal_i holds."""
    comp = component_var(book, scenarios, alpha=0.95)
    marg = marginal_var(book, scenarios, alpha=0.95)
    notionals = np.array([p.notional for p in book])
    np.testing.assert_allclose(comp, notionals * marg, rtol=1e-12)


def test_risk_decomposition_internally_consistent(
    book: list[Position], scenarios: np.ndarray
) -> None:
    """The bundled decomposition agrees with the standalone helpers and itself."""
    rd = risk_decomposition(book, scenarios, alpha=0.95)
    assert rd.component_var.sum() == pytest.approx(rd.total_var, rel=1e-10)
    assert rd.pct_contribution.sum() == pytest.approx(1.0, rel=1e-10)
    np.testing.assert_allclose(
        rd.component_var, component_var(book, scenarios, alpha=0.95), rtol=1e-12
    )
    # Diversification benefit equals sum(standalone) - total and is positive.
    assert rd.diversification_benefit == pytest.approx(
        rd.standalone_var.sum() - rd.total_var, rel=1e-10
    )
    assert rd.diversification_benefit > 0.0


def test_hedge_position_has_negative_component() -> None:
    """An anti-correlated leg reduces book risk -> negative component VaR."""
    factors = _correlated_factors(50_000, rho=0.9, sigmas=(1.0, 1.0), seed=3)
    book = [
        Position("forward", 100.0, "long", _factor0),
        # A small short on the co-mover: notional < wA*rho so the hedge cross
        # term dominates its own variance and the net contribution is negative.
        Position("forward", 50.0, "short", _factor1),
    ]
    comp = component_var(book, factors, alpha=0.95)
    assert comp[1] < 0.0  # the hedge carries negative risk contribution
    assert comp.sum() == pytest.approx(
        portfolio_var_historical(book, factors, alpha=0.95), rel=1e-10
    )


# --------------------------------------------------------------------------- #
# Historical vs Monte Carlo                                                    #
# --------------------------------------------------------------------------- #
def test_historical_vs_monte_carlo_agree(book: list[Position]) -> None:
    """Same law, two estimators: historical and MC portfolio VaR converge."""
    rho, sigmas = 0.3, (1.0, 1.0)
    hist_paths = _correlated_factors(200_000, rho=rho, sigmas=sigmas, seed=21)

    def simulate_fn(n: int, rng: np.random.Generator) -> np.ndarray:
        cov = np.array(
            [
                [sigmas[0] ** 2, rho * sigmas[0] * sigmas[1]],
                [rho * sigmas[0] * sigmas[1], sigmas[1] ** 2],
            ]
        )
        return rng.multivariate_normal([0.0, 0.0], cov, size=n)

    var_hist = portfolio_var_historical(book, hist_paths, alpha=0.95)
    var_mc = portfolio_var_monte_carlo(book, simulate_fn, 200_000, alpha=0.95, seed=99)
    assert var_mc == pytest.approx(var_hist, rel=0.03)


def test_monte_carlo_matches_closed_form_gaussian(book: list[Position]) -> None:
    """MC portfolio VaR matches the analytic Gaussian VaR z*sigma_p."""
    rho, sA, sB = 0.3, 1.0, 1.0
    wA, wB = book[0].signed_notional, book[1].signed_notional
    sigma_p = np.sqrt((wA * sA) ** 2 + (wB * sB) ** 2 + 2 * rho * wA * sB * wB * sA)
    z = 1.6448536269514722  # Phi^{-1}(0.95)

    def simulate_fn(n: int, rng: np.random.Generator) -> np.ndarray:
        cov = np.array([[sA**2, rho * sA * sB], [rho * sA * sB, sB**2]])
        return rng.multivariate_normal([0.0, 0.0], cov, size=n)

    var_mc = portfolio_var_monte_carlo(book, simulate_fn, 400_000, alpha=0.95, seed=5)
    assert var_mc == pytest.approx(z * sigma_p, rel=0.02)


# --------------------------------------------------------------------------- #
# Validation & edge cases                                                     #
# --------------------------------------------------------------------------- #
def test_invalid_asset_type_raises() -> None:
    with pytest.raises(ValueError):
        Position("future", 1.0, "long", _factor0)


def test_negative_notional_raises() -> None:
    with pytest.raises(ValueError):
        Position("forward", -1.0, "long", _factor0)


def test_invalid_direction_raises() -> None:
    with pytest.raises(ValueError):
        Position("forward", 1.0, "sideways", _factor0)
    with pytest.raises(ValueError):
        Position("forward", 1.0, 0.0, _factor0)


def test_numeric_direction_allowed() -> None:
    """A numeric direction acts as a signed (possibly levered) multiplier."""
    pos = Position("forward", 100.0, 1.5, _factor0)
    assert pos.pnl(np.array([2.0, 0.0])) == pytest.approx(100.0 * 1.5 * 2.0)


def test_empty_book_raises_on_decomposition(scenarios: np.ndarray) -> None:
    with pytest.raises(ValueError):
        component_var([], scenarios, alpha=0.95)


def test_mc_bad_n_paths_raises(book: list[Position]) -> None:
    with pytest.raises(ValueError):
        portfolio_var_monte_carlo(book, lambda n, rng: np.zeros((n, 2)), 0)


def test_mc_wrong_scenario_count_raises(book: list[Position]) -> None:
    """A simulator that yields the wrong count is rejected."""
    with pytest.raises(ValueError):
        portfolio_var_monte_carlo(
            book, lambda n, rng: np.zeros((n - 1, 2)), 100, seed=0
        )


def test_riskless_book_has_zero_components() -> None:
    """A constant-P&L book has zero variance -> components attribute nothing."""
    const_fn = lambda s: 1.0  # noqa: E731 - tiny constant valuation
    book = [Position("forward", 100.0, "long", const_fn)]
    scen = np.zeros((1000, 2))
    comp = component_var(book, scen, alpha=0.95)
    np.testing.assert_allclose(comp, 0.0)
