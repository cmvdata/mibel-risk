"""Value-at-Risk and Expected Shortfall estimators.

Three complementary families:

* :mod:`historical`   - non-parametric, empirical-quantile VaR/ES.
* :mod:`parametric`   - closed-form Gaussian and Student-t VaR/ES.
* :mod:`monte_carlo`  - simulation-based VaR/ES, path-generator agnostic.

Sign convention
---------------
All VaR / ES values are returned as **positive loss magnitudes** expressed in
the same units as the input ``returns`` (e.g. fractional P&L or EUR/MWh). A VaR
of ``0.08`` at ``alpha=0.95`` means: with 95% confidence the loss over the
horizon will not exceed 8% of the position.
"""

from mibel_risk.var.historical import (
    expected_shortfall_historical,
    var_historical,
)
from mibel_risk.var.monte_carlo import var_monte_carlo
from mibel_risk.var.parametric import (
    var_parametric_gaussian,
    var_parametric_t,
)

__all__ = [
    "expected_shortfall_historical",
    "var_historical",
    "var_monte_carlo",
    "var_parametric_gaussian",
    "var_parametric_t",
]
