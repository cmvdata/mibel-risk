"""Portfolio-level risk for a MIBEL energy book.

This subpackage lifts the single-position VaR/ES estimators in
:mod:`mibel_risk.var` to a *book* of heterogeneous contracts — spot, forwards,
swing options, CCGT tolling agreements and solar PPAs. The unifying idea is
**scenario revaluation**: every contract is expressed as a
:class:`~mibel_risk.portfolio.positions.Position` that maps one simulated price
scenario to a profit-and-loss number, so all contracts are valued on the *same*
scenario and their P&Ls add up coherently. Aggregating across many scenarios
yields the portfolio P&L distribution from which VaR is read.

On top of the headline number the package decomposes risk back onto the
individual positions:

* :func:`~mibel_risk.portfolio.var_portfolio.component_var` — additive
  contributions that sum to the portfolio VaR (Euler / covariance split).
* :func:`~mibel_risk.portfolio.var_portfolio.marginal_var` — the sensitivity of
  portfolio VaR to a one-unit change in each position's notional.

Sign convention
---------------
P&L is positive for a gain, negative for a loss; VaR is returned as a positive
loss magnitude, matching :mod:`mibel_risk.var`.
"""

from mibel_risk.portfolio.positions import (
    ASSET_TYPES,
    Position,
    portfolio_pnl,
    portfolio_pnl_vector,
)
from mibel_risk.portfolio.var_portfolio import (
    RiskDecomposition,
    component_var,
    marginal_var,
    portfolio_var_historical,
    portfolio_var_monte_carlo,
    risk_decomposition,
)

__all__ = [
    "ASSET_TYPES",
    "Position",
    "RiskDecomposition",
    "component_var",
    "marginal_var",
    "portfolio_pnl",
    "portfolio_pnl_vector",
    "portfolio_var_historical",
    "portfolio_var_monte_carlo",
    "risk_decomposition",
]
