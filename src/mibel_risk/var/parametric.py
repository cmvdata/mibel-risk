"""Parametric (analytic) Value-at-Risk and Expected Shortfall.

Parametric VaR assumes the return distribution belongs to a known family and
reads VaR straight off its inverse CDF. Two laws are provided:

* **Gaussian** - the textbook delta-normal estimator. Cheap and transparent,
  but it systematically *under*-states tail risk for energy returns, whose
  kurtosis is well above 3.
* **Student-t** - a fat-tailed alternative parametrised by degrees of freedom
  ``df``. As ``df -> infinity`` it collapses to the Gaussian; small ``df``
  (3-8 is typical for power) inflates the tail and gives a more honest VaR.

References
----------
Jorion, P. (2007). *Value at Risk* (3rd ed.), McGraw-Hill, Ch. 5
(delta-normal method). McNeil, Frey & Embrechts (2015), *Quantitative Risk
Management*, Ch. 2 & 8 (Student-t VaR and ES closed forms).
"""

from __future__ import annotations

import numpy as np
from scipy import stats

__all__ = [
    "expected_shortfall_gaussian",
    "expected_shortfall_t",
    "var_parametric_gaussian",
    "var_parametric_t",
]


def _check_alpha(alpha: float) -> None:
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"`alpha` must lie strictly in (0, 1); got {alpha}.")


def _check_std(std: float) -> None:
    if std < 0.0:
        raise ValueError(f"`std` must be non-negative; got {std}.")


def var_parametric_gaussian(mean: float, std: float, alpha: float = 0.95) -> float:
    r"""Gaussian (delta-normal) Value-at-Risk.

    For returns :math:`R \sim \mathcal{N}(\mu, \sigma^2)` the VaR at confidence
    ``alpha`` is

    .. math:: \mathrm{VaR}_\alpha = -(\mu + \sigma\,\Phi^{-1}(1-\alpha))
            = \sigma\,\Phi^{-1}(\alpha) - \mu,

    where :math:`\Phi^{-1}` is the standard-normal quantile function.

    Parameters
    ----------
    mean : float
        Mean of the return distribution (per horizon).
    std : float
        Standard deviation (volatility) of returns; must be non-negative.
    alpha : float, default 0.95
        Confidence level in ``(0, 1)``.

    Returns
    -------
    float
        VaR as a loss magnitude. Positive for the usual case of a small mean
        and material volatility.
    """
    _check_alpha(alpha)
    _check_std(std)
    z = stats.norm.ppf(alpha)
    return float(std * z - mean)


def var_parametric_t(mean: float, std: float, df: float, alpha: float = 0.95) -> float:
    r"""Student-t Value-at-Risk for fat-tailed returns.

    The Student-t quantile is rescaled so that ``std`` is the actual standard
    deviation of the distribution (not the raw scale of the t density). For
    ``df > 2`` the t-distribution with scale ``s`` has variance
    :math:`s^2\,\mathrm{df}/(\mathrm{df}-2)`, so we set
    :math:`s = \sigma\sqrt{(\mathrm{df}-2)/\mathrm{df}}` and

    .. math:: \mathrm{VaR}_\alpha = s\,t^{-1}_{\mathrm{df}}(\alpha) - \mu.

    Parameters
    ----------
    mean : float
        Mean of the return distribution.
    std : float
        Standard deviation of returns; must be non-negative.
    df : float
        Degrees of freedom, ``df > 2`` (variance is undefined otherwise).
        Smaller ``df`` -> heavier tails -> larger VaR.
    alpha : float, default 0.95
        Confidence level in ``(0, 1)``.

    Returns
    -------
    float
        VaR as a loss magnitude, always ``>=`` the Gaussian VaR for the same
        ``mean``/``std`` at usual confidence levels (heavier tail).
    """
    _check_alpha(alpha)
    _check_std(std)
    if df <= 2.0:
        raise ValueError(f"`df` must be > 2 for a finite variance; got {df}.")
    scale = std * np.sqrt((df - 2.0) / df)
    q = stats.t.ppf(alpha, df)
    return float(scale * q - mean)


def expected_shortfall_gaussian(mean: float, std: float, alpha: float = 0.95) -> float:
    r"""Gaussian Expected Shortfall (closed form).

    .. math:: \mathrm{ES}_\alpha = \sigma\,\frac{\phi(\Phi^{-1}(\alpha))}{1-\alpha} - \mu,

    where :math:`\phi` is the standard-normal pdf. Always ``>= VaR``.

    Parameters
    ----------
    mean, std : float
        Mean and standard deviation of returns (``std >= 0``).
    alpha : float, default 0.95
        Confidence level in ``(0, 1)``.

    Returns
    -------
    float
        Expected Shortfall as a loss magnitude.
    """
    _check_alpha(alpha)
    _check_std(std)
    z = stats.norm.ppf(alpha)
    es_scaled = stats.norm.pdf(z) / (1.0 - alpha)
    return float(std * es_scaled - mean)


def expected_shortfall_t(mean: float, std: float, df: float, alpha: float = 0.95) -> float:
    r"""Student-t Expected Shortfall (closed form).

    With scale ``s`` chosen so ``std`` is the true standard deviation,

    .. math:: \mathrm{ES}_\alpha = s\,\frac{g(t^{-1}_{\mathrm{df}}(\alpha))}{1-\alpha}
            \cdot \frac{\mathrm{df} + (t^{-1}_{\mathrm{df}}(\alpha))^2}{\mathrm{df}-1} - \mu,

    where :math:`g` is the t pdf with ``df`` degrees of freedom. See McNeil,
    Frey & Embrechts (2015), Example 2.15.

    Parameters
    ----------
    mean, std : float
        Mean and standard deviation of returns (``std >= 0``).
    df : float
        Degrees of freedom, ``df > 2``.
    alpha : float, default 0.95
        Confidence level in ``(0, 1)``.

    Returns
    -------
    float
        Expected Shortfall as a loss magnitude.
    """
    _check_alpha(alpha)
    _check_std(std)
    if df <= 2.0:
        raise ValueError(f"`df` must be > 2 for a finite variance; got {df}.")
    scale = std * np.sqrt((df - 2.0) / df)
    q = stats.t.ppf(alpha, df)
    pdf_q = stats.t.pdf(q, df)
    es_scaled = (pdf_q / (1.0 - alpha)) * ((df + q**2) / (df - 1.0))
    return float(scale * es_scaled - mean)
