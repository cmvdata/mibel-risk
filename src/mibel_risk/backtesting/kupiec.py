"""Kupiec proportion-of-failures (POF) unconditional-coverage test.

Given a series of VaR exceptions, Kupiec's POF test asks whether the observed
failure rate is statistically consistent with the model's nominal rate
``p = 1 - alpha``. It is a likelihood-ratio test with one degree of freedom.

References
----------
Kupiec, P. (1995). "Techniques for verifying the accuracy of risk measurement
models." *Journal of Derivatives* 3(2), 73-84. See also Jorion (2007), Ch. 6.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import stats

__all__ = ["KupiecResult", "kupiec_pof"]


@dataclass(frozen=True)
class KupiecResult:
    """Outcome of a Kupiec POF test.

    Attributes
    ----------
    lr_stat : float
        Likelihood-ratio statistic, asymptotically chi-squared with 1 d.o.f.
    p_value : float
        Right-tail p-value. Small values reject the null of correct coverage.
    n_obs : int
        Number of observations in the backtest window.
    n_exceptions : int
        Observed number of VaR exceptions.
    expected_exceptions : float
        Exceptions expected under the model, ``n_obs * (1 - alpha)``.
    reject : bool
        ``True`` if the null is rejected at the chosen significance level.
    """

    lr_stat: float
    p_value: float
    n_obs: int
    n_exceptions: int
    expected_exceptions: float
    reject: bool


def kupiec_pof(
    exceptions: npt.ArrayLike,
    alpha: float = 0.95,
    significance: float = 0.05,
) -> KupiecResult:
    r"""Kupiec proportion-of-failures (unconditional coverage) test.

    Under the null the exception indicator is i.i.d. Bernoulli with failure
    probability ``p = 1 - alpha``. With ``n`` observations and ``x`` exceptions
    the LR statistic is

    .. math:: LR_{POF} = -2\ln\!\frac{p^{x}(1-p)^{n-x}}
            {\hat{p}^{x}(1-\hat{p})^{n-x}},\quad \hat{p}=x/n,

    which is :math:`\chi^2_1` under the null.

    Parameters
    ----------
    exceptions : array-like of bool or {0, 1}
        Exception series: ``True``/``1`` on days the realised loss exceeded VaR.
    alpha : float, default 0.95
        VaR confidence level; the nominal failure rate is ``1 - alpha``.
    significance : float, default 0.05
        Test size used to set the ``reject`` flag.

    Returns
    -------
    KupiecResult
        Test statistic, p-value and decision.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"`alpha` must lie strictly in (0, 1); got {alpha}.")
    arr = np.asarray(exceptions).astype(bool).ravel()
    n = arr.size
    if n == 0:
        raise ValueError("`exceptions` must contain at least one observation.")

    x = int(arr.sum())
    p = 1.0 - alpha
    p_hat = x / n

    # Log-likelihood under the null (rate p) and the alternative (rate p_hat).
    def _loglik(rate: float) -> float:
        # Guard the log against degenerate 0/1 rates.
        rate = min(max(rate, 1e-12), 1.0 - 1e-12)
        return x * np.log(rate) + (n - x) * np.log(1.0 - rate)

    # All-clear sample (x == 0): closed-form simplification avoids 0*log(0).
    raw_lr = (
        -2.0 * n * np.log(1.0 - p)
        if x == 0
        else -2.0 * (_loglik(p) - _loglik(p_hat))
    )
    lr = float(max(raw_lr, 0.0))

    p_value = float(stats.chi2.sf(lr, df=1))
    return KupiecResult(
        lr_stat=lr,
        p_value=p_value,
        n_obs=n,
        n_exceptions=x,
        expected_exceptions=n * p,
        reject=p_value < significance,
    )
