"""VaR backtesting: coverage and independence tests.

A VaR model is validated out-of-sample by comparing realised losses against the
forecast VaR and counting *exceptions* (days where the loss exceeded VaR). Two
classic likelihood-ratio tests are provided:

* :mod:`kupiec`        - the POF (proportion-of-failures) test: does the
  *number* of exceptions match the model's nominal rate ``1 - alpha``?
* :mod:`christoffersen` - the independence and conditional-coverage tests: are
  exceptions also *independent* in time, or do they cluster (a sign the model
  misses volatility dynamics)?

References
----------
Kupiec (1995); Christoffersen (1998); Jorion (2007), Ch. 6.
"""

from mibel_risk.backtesting.christoffersen import (
    christoffersen_conditional_coverage,
    christoffersen_independence,
)
from mibel_risk.backtesting.kupiec import kupiec_pof

__all__ = [
    "christoffersen_conditional_coverage",
    "christoffersen_independence",
    "kupiec_pof",
]
