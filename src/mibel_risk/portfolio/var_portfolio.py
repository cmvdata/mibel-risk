r"""Portfolio Value-at-Risk and its decomposition onto positions.

The portfolio P&L distribution is obtained by revaluing every
:class:`~mibel_risk.portfolio.positions.Position` on a common set of price
scenarios (historical paths or Monte Carlo draws) and summing. VaR is then the
single-position historical-simulation estimator applied to that aggregate P&L.

Risk decomposition
------------------
Portfolio VaR is positively homogeneous of degree one in the position sizes, so
by Euler's theorem it splits additively into *component VaRs* that sum back to
the total. We use the covariance (delta-normal) split, which is exact-additive
for **any** total-VaR figure: writing the portfolio P&L as :math:`P=\sum_i p_i`
with :math:`p_i` the P&L of position *i*,

.. math::

    \mathrm{CVaR}_i = \frac{\operatorname{Cov}(p_i, P)}{\operatorname{Var}(P)}
                      \, \mathrm{VaR}_P ,
    \qquad \sum_i \mathrm{CVaR}_i = \mathrm{VaR}_P ,

because :math:`\sum_i \operatorname{Cov}(p_i, P)=\operatorname{Var}(P)`. The
*marginal VaR* of a position is the sensitivity of portfolio VaR to its
notional; by homogeneity ``component = notional * marginal``, so
:math:`\mathrm{MVaR}_i = \mathrm{CVaR}_i / \text{notional}_i`.

References
----------
Jorion, P. (2007). *Value at Risk* (3rd ed.), McGraw-Hill, Ch. 7
(component and marginal VaR). Hallerbach, W. (2003). "Decomposing Portfolio
Value-at-Risk: A General Analysis", *Journal of Risk* 5(2).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from mibel_risk.portfolio.positions import (
    Position,
    Scenario,
    portfolio_pnl_vector,
)
from mibel_risk.var.historical import var_historical

__all__ = [
    "RiskDecomposition",
    "component_var",
    "marginal_var",
    "portfolio_var_historical",
    "portfolio_var_monte_carlo",
    "risk_decomposition",
]

#: A scenario generator: ``simulate_fn(n_paths, rng) -> sequence of scenarios``.
SimulateScenariosFn = Callable[[int, np.random.Generator], Sequence[Scenario]]


def _pnl_matrix(
    positions: Sequence[Position], scenarios: Sequence[Scenario]
) -> npt.NDArray[np.float64]:
    """Per-position P&L matrix of shape ``(n_positions, n_scenarios)``."""
    if len(positions) == 0:
        raise ValueError("`positions` must contain at least one position.")
    return np.array(
        [[pos.pnl(scenario) for scenario in scenarios] for pos in positions],
        dtype=np.float64,
    )


def portfolio_var_historical(
    positions: Sequence[Position],
    historical_paths: Sequence[Scenario],
    alpha: float = 0.95,
) -> float:
    """Historical-simulation VaR of the book.

    Revalues every position on each historical scenario, sums to a portfolio
    P&L per scenario, and returns the empirical ``(1 - alpha)`` quantile loss.

    Parameters
    ----------
    positions : sequence of Position
        The book.
    historical_paths : sequence of scenario objects
        Realised (or bootstrapped) price scenarios. Each is passed to every
        position's ``valuation_fn``.
    alpha : float, default 0.95
        Confidence level in ``(0, 1)``.

    Returns
    -------
    float
        Portfolio VaR as a positive loss magnitude (EUR).
    """
    pnl = portfolio_pnl_vector(positions, historical_paths)
    return var_historical(pnl, alpha=alpha)


def portfolio_var_monte_carlo(
    positions: Sequence[Position],
    simulate_fn: SimulateScenariosFn,
    n_paths: int,
    alpha: float = 0.95,
    *,
    seed: int | None = None,
) -> float:
    """Monte Carlo VaR of the book via a pluggable scenario generator.

    ``simulate_fn`` produces price *scenarios* (not P&L); the positions revalue
    them. This is what lets the same machinery consume a Schwartz-Smith forward
    simulator, a spot-model path generator, or a toy Gaussian sampler unchanged.

    Parameters
    ----------
    positions : sequence of Position
        The book.
    simulate_fn : callable
        ``simulate_fn(n_paths, rng) -> sequence`` of ``n_paths`` scenarios,
        receiving a seeded :class:`numpy.random.Generator`.
    n_paths : int
        Number of scenarios to draw; must be positive.
    alpha : float, default 0.95
        Confidence level in ``(0, 1)``.
    seed : int, optional
        Seed for the generator handed to ``simulate_fn`` (reproducibility).

    Returns
    -------
    float
        Portfolio VaR as a positive loss magnitude (EUR).

    Raises
    ------
    ValueError
        If ``n_paths`` is not positive or ``simulate_fn`` yields a different
        number of scenarios than requested.
    """
    if n_paths <= 0:
        raise ValueError(f"`n_paths` must be positive; got {n_paths}.")
    rng = np.random.default_rng(seed)
    scenarios = simulate_fn(n_paths, rng)
    pnl = portfolio_pnl_vector(positions, scenarios)
    if pnl.size != n_paths:
        raise ValueError(
            f"`simulate_fn` produced {pnl.size} scenarios but `n_paths`={n_paths}."
        )
    return var_historical(pnl, alpha=alpha)


def _component_from_matrix(
    pnl_matrix: npt.NDArray[np.float64], alpha: float, total_var: float | None
) -> tuple[npt.NDArray[np.float64], float]:
    """Covariance split of total VaR; returns ``(component_var, total_var)``."""
    portfolio = pnl_matrix.sum(axis=0)
    var_p = var_historical(portfolio, alpha=alpha) if total_var is None else float(total_var)
    var_total = float(np.var(portfolio))  # population variance (bias=True)
    n_pos = pnl_matrix.shape[0]
    if var_total <= 0.0:
        # Degenerate: a riskless book has no meaningful split. Attribute nothing.
        return np.zeros(n_pos, dtype=np.float64), var_p
    # Cov(p_i, P) with the population convention so the covariances sum to
    # Var(P) exactly, guaranteeing sum(component) == var_p.
    centred = pnl_matrix - pnl_matrix.mean(axis=1, keepdims=True)
    port_centred = portfolio - portfolio.mean()
    cov = (centred * port_centred).mean(axis=1)
    betas = cov / var_total
    return betas * var_p, var_p


def component_var(
    positions: Sequence[Position],
    scenarios: Sequence[Scenario],
    alpha: float = 0.95,
    *,
    total_var: float | None = None,
) -> npt.NDArray[np.float64]:
    """Component VaR: additive risk contribution of each position.

    The contributions sum to the portfolio VaR (within floating-point error).
    A negative component flags a position that *hedges* the book — its P&L is
    anti-correlated with the aggregate, so it reduces total risk.

    Parameters
    ----------
    positions : sequence of Position
        The book, length ``n``.
    scenarios : sequence of scenario objects
        Price scenarios to revalue on (historical or simulated).
    alpha : float, default 0.95
        Confidence level in ``(0, 1)``.
    total_var : float, optional
        Portfolio VaR to allocate. Defaults to the historical VaR of the book
        on these scenarios; pass an explicit figure (e.g. a parametric or
        regulatory VaR) to decompose that instead.

    Returns
    -------
    numpy.ndarray
        Component VaR per position (EUR), aligned with ``positions`` and
        summing to the total.
    """
    matrix = _pnl_matrix(positions, scenarios)
    comp, _ = _component_from_matrix(matrix, alpha, total_var)
    return comp


def marginal_var(
    positions: Sequence[Position],
    scenarios: Sequence[Scenario],
    alpha: float = 0.95,
    *,
    total_var: float | None = None,
) -> npt.NDArray[np.float64]:
    """Marginal VaR: sensitivity of portfolio VaR to each position's notional.

    By the homogeneity identity ``component = notional * marginal``, the
    marginal VaR is the component VaR per unit of notional — the EUR change in
    portfolio VaR from adding one more unit (MWh, MW, ...) of the position,
    holding the rest of the book fixed.

    Parameters
    ----------
    positions : sequence of Position
        The book.
    scenarios : sequence of scenario objects
        Price scenarios to revalue on.
    alpha : float, default 0.95
        Confidence level in ``(0, 1)``.
    total_var : float, optional
        Portfolio VaR to allocate; see :func:`component_var`.

    Returns
    -------
    numpy.ndarray
        Marginal VaR per position (EUR per unit of notional). Positions with
        zero notional carry zero marginal (the contribution is degenerate).
    """
    comp = component_var(positions, scenarios, alpha, total_var=total_var)
    notionals = np.array([float(pos.notional) for pos in positions], dtype=np.float64)
    marg = np.zeros_like(comp)
    nz = notionals > 0.0
    marg[nz] = comp[nz] / notionals[nz]
    return marg


@dataclass(frozen=True)
class RiskDecomposition:
    """Bundled portfolio-VaR decomposition.

    Attributes
    ----------
    total_var : float
        Portfolio VaR (EUR), positive loss magnitude.
    component_var : numpy.ndarray
        Additive contribution per position; sums to ``total_var``.
    marginal_var : numpy.ndarray
        Per-unit-of-notional sensitivity per position.
    pct_contribution : numpy.ndarray
        ``component_var / total_var`` — the share of risk each position carries
        (sums to 1 when ``total_var`` is non-zero).
    standalone_var : numpy.ndarray
        Each position's VaR valued in isolation. ``total_var`` is at most the
        sum of these (diversification), with equality only under perfect
        co-movement.
    diversification_benefit : float
        ``sum(standalone_var) - total_var`` — risk netted away by holding the
        positions jointly; non-negative whenever VaR is sub-additive here.
    """

    total_var: float
    component_var: npt.NDArray[np.float64]
    marginal_var: npt.NDArray[np.float64]
    pct_contribution: npt.NDArray[np.float64]
    standalone_var: npt.NDArray[np.float64]
    diversification_benefit: float


def risk_decomposition(
    positions: Sequence[Position],
    scenarios: Sequence[Scenario],
    alpha: float = 0.95,
) -> RiskDecomposition:
    """Full portfolio-VaR decomposition in one pass.

    Computes the portfolio VaR, its component and marginal splits, each
    position's standalone VaR, and the diversification benefit, sharing the
    single per-position P&L matrix across all of them.

    Parameters
    ----------
    positions : sequence of Position
        The book.
    scenarios : sequence of scenario objects
        Price scenarios to revalue on.
    alpha : float, default 0.95
        Confidence level in ``(0, 1)``.

    Returns
    -------
    RiskDecomposition
    """
    matrix = _pnl_matrix(positions, scenarios)
    comp, total = _component_from_matrix(matrix, alpha, None)
    notionals = np.array([float(pos.notional) for pos in positions], dtype=np.float64)
    marg = np.zeros_like(comp)
    nz = notionals > 0.0
    marg[nz] = comp[nz] / notionals[nz]
    pct = comp / total if total != 0.0 else np.zeros_like(comp)
    standalone = np.array(
        [var_historical(matrix[i], alpha=alpha) for i in range(matrix.shape[0])],
        dtype=np.float64,
    )
    return RiskDecomposition(
        total_var=total,
        component_var=comp,
        marginal_var=marg,
        pct_contribution=pct,
        standalone_var=standalone,
        diversification_benefit=float(standalone.sum() - total),
    )
