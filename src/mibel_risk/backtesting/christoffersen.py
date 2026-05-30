"""Christoffersen independence and conditional-coverage tests.

Kupiec's POF test checks the *number* of VaR exceptions but is blind to their
*timing*. Christoffersen (1998) adds two likelihood-ratio tests:

* **Independence** - tests whether an exception today is independent of an
  exception yesterday (a first-order Markov chain). Clustering signals a model
  that ignores volatility dynamics.
* **Conditional coverage** - the joint test, ``LR_cc = LR_pof + LR_ind``,
  combining correct unconditional coverage *and* independence.

References
----------
Christoffersen, P. (1998). "Evaluating interval forecasts." *International
Economic Review* 39(4), 841-862. See also Jorion (2007), Ch. 6.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import stats

from mibel_risk.backtesting.kupiec import kupiec_pof

__all__ = [
    "ChristoffersenResult",
    "christoffersen_conditional_coverage",
    "christoffersen_independence",
]


@dataclass(frozen=True)
class ChristoffersenResult:
    """Outcome of a Christoffersen likelihood-ratio test.

    Attributes
    ----------
    lr_stat : float
        Likelihood-ratio statistic.
    p_value : float
        Right-tail p-value under the chi-squared null.
    df : int
        Degrees of freedom (1 for independence, 2 for conditional coverage).
    reject : bool
        ``True`` if the null is rejected at the chosen significance level.
    """

    lr_stat: float
    p_value: float
    df: int
    reject: bool


def _transition_counts(arr: npt.NDArray[np.bool_]) -> tuple[int, int, int, int]:
    """Count Markov transitions (n00, n01, n10, n11) in the exception series."""
    prev = arr[:-1]
    cur = arr[1:]
    n00 = int(np.sum(~prev & ~cur))
    n01 = int(np.sum(~prev & cur))
    n10 = int(np.sum(prev & ~cur))
    n11 = int(np.sum(prev & cur))
    return n00, n01, n10, n11


def christoffersen_independence(
    exceptions: npt.ArrayLike,
    significance: float = 0.05,
) -> ChristoffersenResult:
    r"""Christoffersen independence likelihood-ratio test.

    Models the exception series as a first-order Markov chain and tests the null
    that the probability of an exception does not depend on whether yesterday
    was an exception (:math:`\pi_{01} = \pi_{11}`). The statistic is
    :math:`\chi^2_1` under the null.

    Parameters
    ----------
    exceptions : array-like of bool or {0, 1}
        Exception series ordered in time.
    significance : float, default 0.05
        Test size used to set the ``reject`` flag.

    Returns
    -------
    ChristoffersenResult
        Statistic, p-value (1 d.o.f.) and decision. When no exception is
        followed by a transition (e.g. zero or one exception), the test is
        degenerate and returns ``lr_stat = 0`` (cannot reject independence).
    """
    arr = np.asarray(exceptions).astype(bool).ravel()
    if arr.size < 2:
        raise ValueError("`exceptions` must contain at least two observations.")

    n00, n01, n10, n11 = _transition_counts(arr)

    # Transition probabilities of an exception given prior state.
    denom0 = n00 + n01
    denom1 = n10 + n11
    pi01 = n01 / denom0 if denom0 else 0.0
    pi11 = n11 / denom1 if denom1 else 0.0
    pi = (n01 + n11) / (denom0 + denom1) if (denom0 + denom1) else 0.0

    def _ll(p: float, k: int, n: int) -> float:
        if p <= 0.0 or p >= 1.0:
            return 0.0
        return k * np.log(p) + (n - k) * np.log(1.0 - p)

    # If there are no exceptions at all, independence cannot be assessed.
    if pi in (0.0, 1.0):
        return ChristoffersenResult(lr_stat=0.0, p_value=1.0, df=1, reject=False)

    ll_null = _ll(pi, n01 + n11, denom0 + denom1)
    ll_alt = _ll(pi01, n01, denom0) + _ll(pi11, n11, denom1)
    lr = float(max(-2.0 * (ll_null - ll_alt), 0.0))
    p_value = float(stats.chi2.sf(lr, df=1))
    return ChristoffersenResult(
        lr_stat=lr, p_value=p_value, df=1, reject=p_value < significance
    )


def christoffersen_conditional_coverage(
    exceptions: npt.ArrayLike,
    alpha: float = 0.95,
    significance: float = 0.05,
) -> ChristoffersenResult:
    r"""Christoffersen conditional-coverage test.

    The joint test of correct coverage *and* independence,

    .. math:: LR_{cc} = LR_{POF} + LR_{ind},

    distributed :math:`\chi^2_2` under the null. Rejecting it means the VaR
    model fails on the exception *count*, the exception *clustering*, or both.

    Parameters
    ----------
    exceptions : array-like of bool or {0, 1}
        Exception series ordered in time.
    alpha : float, default 0.95
        VaR confidence level feeding the POF component.
    significance : float, default 0.05
        Test size used to set the ``reject`` flag.

    Returns
    -------
    ChristoffersenResult
        Combined statistic, p-value (2 d.o.f.) and decision.
    """
    pof = kupiec_pof(exceptions, alpha=alpha, significance=significance)
    ind = christoffersen_independence(exceptions, significance=significance)
    lr = float(pof.lr_stat + ind.lr_stat)
    p_value = float(stats.chi2.sf(lr, df=2))
    return ChristoffersenResult(
        lr_stat=lr, p_value=p_value, df=2, reject=p_value < significance
    )
