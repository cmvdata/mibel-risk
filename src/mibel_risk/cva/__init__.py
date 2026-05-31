"""Credit Valuation Adjustment (CVA) for an exposure profile.

CVA is the market price of counterparty credit risk: the expected loss from a
counterparty defaulting while it owes you a positive mark-to-market. Given an
**expected positive exposure** curve ``EE(t)`` (e.g. the one produced by
:mod:`mibel_risk.pfe`), a **default curve** (hazard rates, constant or
piecewise), a **recovery** rate and a **discount curve**, the discrete CVA is

.. math::

    \\mathrm{CVA} = (1 - R) \\sum_{i} \\mathrm{EE}(t_i)\\, \\mathrm{DF}(t_i)\\,
                    \\big[S(t_{i-1}) - S(t_i)\\big],

where ``S(t) = exp(-Λ(t))`` is the survival probability and
``S(t_{i-1}) - S(t_i)`` the marginal probability of default in interval *i*.

This subpackage is standalone (numpy/scipy only) and consumes the exposure
curve as a plain array, so it composes directly with the PFE estimators.
"""

from mibel_risk.cva.cva import (
    CVAResult,
    DefaultCurve,
    cva,
    discount_from_rate,
    survival_from_hazard,
)

__all__ = [
    "CVAResult",
    "DefaultCurve",
    "cva",
    "discount_from_rate",
    "survival_from_hazard",
]
