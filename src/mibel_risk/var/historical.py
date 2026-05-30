"""Historical (non-parametric) Value-at-Risk and Expected Shortfall.

The historical-simulation method makes no distributional assumption: it reads
the risk of a position straight off the empirical distribution of realised
returns. This is the workhorse estimator for energy books because power and gas
return distributions are sharply non-Gaussian (heavy tails, spikes, regime
switches) and a closed-form law rarely fits across the whole sample.

References
----------
Jorion, P. (2007). *Value at Risk: The New Benchmark for Managing Financial
Risk* (3rd ed.), McGraw-Hill. Ch. 5 (historical simulation), Ch. 7 (Expected
Shortfall / conditional VaR).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

__all__ = ["expected_shortfall_historical", "var_historical"]


def _as_loss_array(returns: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Validate input and return a 1-D float array of returns (NaNs dropped)."""
    arr = np.asarray(returns, dtype=np.float64).ravel()
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        raise ValueError("`returns` must contain at least one non-NaN observation.")
    return arr


def _check_alpha(alpha: float) -> None:
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"`alpha` must lie strictly in (0, 1); got {alpha}.")


def var_historical(returns: npt.ArrayLike, alpha: float = 0.95) -> float:
    """Historical-simulation Value-at-Risk.

    VaR is the empirical ``(1 - alpha)`` quantile of the return distribution,
    reported as a positive loss. With ``alpha = 0.95`` the result is the loss
    that is exceeded on only 5% of days.

    Parameters
    ----------
    returns : array-like
        Realised period returns (or P&L) of the position. Sign convention:
        positive = gain, negative = loss. NaNs are ignored.
    alpha : float, default 0.95
        Confidence level in ``(0, 1)``. Higher ``alpha`` -> more extreme,
        larger VaR.

    Returns
    -------
    float
        VaR as a non-negative loss magnitude in the units of ``returns``. A
        return distribution whose ``(1 - alpha)`` quantile is positive (rare)
        yields a negative number, meaning even the tail outcome is a gain.

    Notes
    -----
    The quantile uses linear interpolation between order statistics
    (NumPy's default), which is the standard estimator in Jorion (2007, Ch. 5).
    """
    _check_alpha(alpha)
    arr = _as_loss_array(returns)
    quantile = np.quantile(arr, 1.0 - alpha)
    return float(-quantile)


def expected_shortfall_historical(returns: npt.ArrayLike, alpha: float = 0.95) -> float:
    """Historical Expected Shortfall (a.k.a. Conditional VaR / CVaR).

    ES is the average loss *conditional* on the loss breaching VaR — the mean of
    the worst ``(1 - alpha)`` fraction of outcomes. Unlike VaR it is a coherent
    risk measure (sub-additive) and captures tail severity beyond the quantile,
    which matters for the fat-tailed P&L of power positions.

    Parameters
    ----------
    returns : array-like
        Realised period returns (or P&L). NaNs are ignored.
    alpha : float, default 0.95
        Confidence level in ``(0, 1)``.

    Returns
    -------
    float
        Expected Shortfall as a non-negative loss magnitude. By construction
        ``ES >= VaR`` at the same ``alpha``.

    Notes
    -----
    Computed as the mean of all returns at or below the empirical
    ``(1 - alpha)`` quantile (the VaR threshold). If no observation lies in the
    tail (tiny sample), the single worst observation is used as a fallback so
    the estimator stays well-defined.

    References
    ----------
    Jorion (2007), Ch. 7; Acerbi & Tasche (2002), "On the coherence of
    expected shortfall", *Journal of Banking & Finance* 26(7).
    """
    _check_alpha(alpha)
    arr = _as_loss_array(returns)
    threshold = np.quantile(arr, 1.0 - alpha)
    tail = arr[arr <= threshold]
    if tail.size == 0:
        # Degenerate sample: fall back to the worst single outcome.
        tail = np.array([arr.min()])
    return float(-tail.mean())
