r"""Discrete unilateral Credit Valuation Adjustment.

CVA prices the expected loss from a counterparty defaulting while it owes a
positive mark-to-market. Over a time grid :math:`t_0 < t_1 < \dots < t_n` with
expected positive exposure :math:`\mathrm{EE}(t_i)`, recovery :math:`R`, discount
factors :math:`\mathrm{DF}(t_i)` and survival probabilities :math:`S(t_i)`:

.. math::

    \mathrm{CVA} = (1 - R) \sum_{i=1}^{n}
        \mathrm{EE}(t_i)\, \mathrm{DF}(t_i)\, \big[S(t_{i-1}) - S(t_i)\big].

The bracket is the marginal probability of default in interval :math:`i`; the
exposure is taken at the interval end (a standard, slightly conservative
convention). Survival comes from a hazard-rate default curve, constant or
piecewise-constant per interval, via :math:`S(t) = \exp(-\int_0^t \lambda)`.

References
----------
Gregory, J. (2020). *The xVA Challenge* (3rd ed.), Wiley — Ch. 12 (CVA). Brigo,
Morini & Pallavicini (2013), *Counterparty Credit Risk, Collateral and Funding*.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

__all__ = [
    "CVAResult",
    "DefaultCurve",
    "cva",
    "discount_from_rate",
    "survival_from_hazard",
]


def survival_from_hazard(
    time_grid: npt.ArrayLike, hazard: float | npt.ArrayLike
) -> npt.NDArray[np.float64]:
    r"""Survival probabilities ``S(t_i)`` from a hazard-rate curve.

    ``S(t_i) = exp(-Λ(t_i))`` with the cumulative hazard
    :math:`Λ(t_i) = \sum_{j<i} \lambda_j (t_{j+1} - t_j)`.

    Parameters
    ----------
    time_grid : array-like, shape (n,)
        Strictly increasing dates from ``t_0`` (today, where ``S = 1``).
    hazard : float or array-like
        The hazard rate(s). A scalar gives a **constant** hazard; an array gives
        a **piecewise-constant** hazard, of length ``n - 1`` (one rate per
        interval) or ``n`` (the value at each interval's end is used). Must be
        non-negative.

    Returns
    -------
    numpy.ndarray
        Survival curve of shape ``(n,)`` with ``S[0] = 1``, non-increasing.
    """
    grid = np.asarray(time_grid, dtype=np.float64).ravel()
    n = grid.size
    if n < 2:
        raise ValueError("`time_grid` must have at least two points.")
    if np.any(np.diff(grid) <= 0.0):
        raise ValueError("`time_grid` must be strictly increasing.")
    dt = np.diff(grid)  # length n - 1

    haz = np.asarray(hazard, dtype=np.float64).ravel()
    if haz.size == 1:
        per_interval = np.full(n - 1, haz.item())
    elif haz.size == n - 1:
        per_interval = haz
    elif haz.size == n:
        per_interval = haz[1:]  # value at each interval end
    else:
        raise ValueError(
            f"`hazard` must be scalar or have length {n - 1} or {n}; got {haz.size}."
        )
    if np.any(per_interval < 0.0):
        raise ValueError("hazard rates must be non-negative.")

    cumulative = np.concatenate([[0.0], np.cumsum(per_interval * dt)])
    return np.exp(-cumulative)


def discount_from_rate(
    time_grid: npt.ArrayLike, rate: float | npt.ArrayLike
) -> npt.NDArray[np.float64]:
    """Discount factors ``DF(t_i) = exp(-r t_i)`` from a flat (or per-point) rate.

    Parameters
    ----------
    time_grid : array-like, shape (n,)
        Dates from today.
    rate : float or array-like
        Continuously-compounded discount rate; scalar (flat) or one per grid
        point.

    Returns
    -------
    numpy.ndarray
        Discount factors of shape ``(n,)`` in ``(0, 1]``.
    """
    grid = np.asarray(time_grid, dtype=np.float64).ravel()
    r = np.asarray(rate, dtype=np.float64)
    return np.exp(-r * grid)


@dataclass(frozen=True)
class DefaultCurve:
    """A hazard-rate default curve.

    Attributes
    ----------
    hazard : float or numpy.ndarray
        Constant hazard (scalar) or piecewise-constant per-interval hazards.
    """

    hazard: float | npt.NDArray[np.float64]

    def survival(self, time_grid: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Survival curve on ``time_grid`` (see :func:`survival_from_hazard`)."""
        return survival_from_hazard(time_grid, self.hazard)

    def marginal_pd(self, time_grid: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Marginal default probabilities ``S(t_{i-1}) - S(t_i)``, shape (n-1,)."""
        s = self.survival(time_grid)
        return -np.diff(s)


@dataclass(frozen=True)
class CVAResult:
    """Outcome of a CVA computation.

    Attributes
    ----------
    cva : float
        The credit valuation adjustment (a non-negative loss, EUR).
    recovery : float
        Recovery rate used.
    total_default_prob : float
        Cumulative default probability over the horizon, ``1 - S(t_n)``.
    contributions : numpy.ndarray
        Per-interval CVA contributions (shape ``n - 1``); sum to ``cva``.
    discounted_ee : numpy.ndarray
        ``EE(t_i) * DF(t_i)`` at each interval end (shape ``n - 1``).
    """

    cva: float
    recovery: float
    total_default_prob: float
    contributions: npt.NDArray[np.float64]
    discounted_ee: npt.NDArray[np.float64]


def _coerce_default_curve(default_curve: DefaultCurve | float | npt.ArrayLike) -> DefaultCurve:
    if isinstance(default_curve, DefaultCurve):
        return default_curve
    return DefaultCurve(hazard=default_curve)  # float or array of hazards


def cva(
    time_grid: npt.ArrayLike,
    positive_exposure_curve: npt.ArrayLike,
    default_curve: DefaultCurve | float | npt.ArrayLike,
    recovery: float = 0.4,
    discount_curve: npt.ArrayLike | None = None,
) -> CVAResult:
    r"""Discrete unilateral CVA of an exposure profile.

    Parameters
    ----------
    time_grid : array-like, shape (n,)
        Strictly increasing dates from today (``t_0``).
    positive_exposure_curve : array-like, shape (n,)
        Expected **positive** exposure ``EE(t_i)`` (e.g.
        :attr:`mibel_risk.pfe.PFEResult.expected_exposure`). Must be
        non-negative.
    default_curve : DefaultCurve, float, or array-like
        The counterparty's hazard-rate curve. A scalar is a constant hazard; an
        array is piecewise-constant per interval; or pass a :class:`DefaultCurve`.
    recovery : float, default 0.4
        Recovery rate ``R`` in ``[0, 1]``; the loss given default is ``1 - R``.
    discount_curve : array-like, shape (n,), optional
        Discount factors ``DF(t_i)``. Default is all ones (undiscounted).

    Returns
    -------
    CVAResult

    Notes
    -----
    ``CVA = (1 - R) Σ_i EE(t_i) DF(t_i) [S(t_{i-1}) - S(t_i)]`` with exposure at
    the interval end. Non-negative by construction.
    """
    grid = np.asarray(time_grid, dtype=np.float64).ravel()
    ee = np.asarray(positive_exposure_curve, dtype=np.float64).ravel()
    n = grid.size
    if n < 2:
        raise ValueError("`time_grid` must have at least two points.")
    if ee.size != n:
        raise ValueError("`positive_exposure_curve` must match `time_grid` length.")
    if np.any(ee < -1e-12):
        raise ValueError("`positive_exposure_curve` must be non-negative.")
    if not 0.0 <= recovery <= 1.0:
        raise ValueError(f"`recovery` must lie in [0, 1]; got {recovery}.")

    if discount_curve is None:
        df = np.ones(n, dtype=np.float64)
    else:
        df = np.asarray(discount_curve, dtype=np.float64).ravel()
        if df.size != n:
            raise ValueError("`discount_curve` must match `time_grid` length.")
        if np.any(df < 0.0):
            raise ValueError("discount factors must be non-negative.")

    curve = _coerce_default_curve(default_curve)
    survival = curve.survival(grid)
    marginal_pd = -np.diff(survival)  # S[i-1] - S[i], shape (n-1,)

    lgd = 1.0 - recovery
    discounted_ee = ee[1:] * df[1:]  # interval-end exposure
    contributions = lgd * discounted_ee * marginal_pd
    return CVAResult(
        cva=float(contributions.sum()),
        recovery=float(recovery),
        total_default_prob=float(1.0 - survival[-1]),
        contributions=contributions,
        discounted_ee=discounted_ee,
    )
