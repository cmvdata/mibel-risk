"""mibel-risk: market risk management for MIBEL energy portfolios.

Module 4 of the MIBEL analytics stack. Provides Value-at-Risk (VaR),
Expected Shortfall (ES), and backtesting tooling for power and gas
trading books denominated in EUR/MWh.

Submodules
----------
var          : historical, parametric (Gaussian / Student-t) and Monte Carlo VaR.
backtesting  : Kupiec POF and Christoffersen independence/conditional coverage tests.
"""

__version__ = "0.1.0"
