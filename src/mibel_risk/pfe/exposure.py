r"""Monte Carlo Potential Future Exposure for swing, tolling and PPA contracts.

Each public estimator builds a ``(n_paths, n_steps)`` matrix of contract values
along the simulated market paths, converts it to a positive-exposure matrix
relative to inception, and reduces it to a PFE curve. The common engine is
:func:`pfe_from_value_paths`; the three wrappers differ only in the per-step
*payoff leg* they plug in:

* **swing call** — ``max(price - strike, 0)`` (a strip of daily calls);
* **tolling** — ``max(spark_spread, 0)`` with
  ``spark = power - HR*(gas + co2*eua)`` (a strip of spark-spread options);
* **PPA** — ``spot_pct * (price - strike)`` (a two-sided spot-vs-fixed swap, so
  exposure can come from either side of the strike).

Exposure is reported **undiscounted** (in time-*t* EUR); discounting belongs to
the downstream CVA integral (:mod:`mibel_risk.cva`).

References
----------
Gregory, J. (2020). *The xVA Challenge* (3rd ed.), Wiley — Ch. 7 (exposure) and
Ch. 11 (PFE / expected exposure profiles). Jorion (2007), Ch. 14 (credit
exposure of derivatives).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

__all__ = [
    "PFEResult",
    "pfe_from_value_paths",
    "pfe_ppa",
    "pfe_swing",
    "pfe_tolling",
]

#: GJ per MWh-thermal — the unit bridge for heat-rate arithmetic, matching
#: ``mibel_derivatives.products.tolling.GJ_PER_MWH``.
GJ_PER_MWH = 3.6


@dataclass(frozen=True)
class PFEResult:
    """A potential-future-exposure profile.

    Attributes
    ----------
    time_grid : numpy.ndarray
        The grid the profile is evaluated on (``n_steps`` points, in the same
        units the caller used — typically years or days from today).
    pfe : numpy.ndarray
        ``PFE(t)`` — the ``alpha``-quantile of positive exposure at each grid
        point (EUR). Non-negative, ``pfe[0] == 0``.
    expected_exposure : numpy.ndarray
        ``EE(t)`` — the mean positive exposure at each grid point (EUR). This is
        the curve the CVA integral consumes.
    alpha : float
        Quantile confidence level used for ``pfe``.
    n_paths : int
        Number of Monte Carlo paths behind the estimate.
    peak_pfe : float
        Maximum of ``pfe`` over the grid (the headline PFE number).
    peak_time : float
        Grid point at which ``peak_pfe`` occurs.
    """

    time_grid: npt.NDArray[np.float64]
    pfe: npt.NDArray[np.float64]
    expected_exposure: npt.NDArray[np.float64]
    alpha: float
    n_paths: int
    peak_pfe: float
    peak_time: float


def _check_alpha(alpha: float) -> None:
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"`alpha` must lie strictly in (0, 1); got {alpha}.")


def _resolve_paths(
    source: Any,
    n_paths: int | None,
    n_steps: int,
    seed: int | None,
) -> npt.NDArray[np.float64]:
    """Coerce a path source (array or simulator callable) to a 2-D float array."""
    if callable(source):
        if n_paths is None or n_paths <= 0:
            raise ValueError("`n_paths` must be a positive int when a simulator is given.")
        rng = np.random.default_rng(seed)
        arr = np.asarray(source(n_paths, rng), dtype=np.float64)
    else:
        arr = np.asarray(source, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"price paths must be 2-D (n_paths, n_steps); got shape {arr.shape}.")
    if arr.shape[1] != n_steps:
        raise ValueError(
            f"price paths have {arr.shape[1]} steps but `time_grid` has {n_steps}."
        )
    if not np.isfinite(arr).all():
        raise ValueError("price paths contain non-finite values.")
    return arr


def _remaining_fraction(n_steps: int) -> npt.NDArray[np.float64]:
    """Linear run-off of remaining delivery volume: 1 at inception, 0 at maturity."""
    if n_steps < 2:
        raise ValueError("`time_grid` must have at least two points.")
    k = np.arange(n_steps)
    return (n_steps - 1 - k) / (n_steps - 1)


def pfe_from_value_paths(
    value_paths: npt.ArrayLike,
    time_grid: npt.ArrayLike,
    alpha: float = 0.95,
) -> PFEResult:
    r"""PFE / EE curves from a matrix of contract values along simulated paths.

    The mark-to-market relative to inception is ``value_t - value_0`` (with
    ``value_0`` the common first-column value); its positive part is the
    exposure. ``PFE(t)`` is the ``alpha``-quantile of exposure across paths and
    ``EE(t)`` its mean.

    Parameters
    ----------
    value_paths : array-like, shape (n_paths, n_steps)
        Contract value (replacement cost of remaining cashflows) along each
        simulated path, aligned with ``time_grid``.
    time_grid : array-like, shape (n_steps,)
        Exposure dates (strictly increasing; ``time_grid[0]`` is inception).
    alpha : float, default 0.95
        PFE quantile confidence level in ``(0, 1)``.

    Returns
    -------
    PFEResult
    """
    _check_alpha(alpha)
    grid = np.asarray(time_grid, dtype=np.float64).ravel()
    if grid.size < 2:
        raise ValueError("`time_grid` must have at least two points (inception + one date).")
    values = np.asarray(value_paths, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("`value_paths` must be 2-D (n_paths, n_steps).")
    if values.shape[1] != grid.size:
        raise ValueError("`value_paths` columns must match `time_grid` length.")
    if values.shape[0] < 1:
        raise ValueError("`value_paths` must contain at least one path.")

    mtm = values - values[:, [0]]  # relative to the (deterministic) inception value
    exposure = np.maximum(mtm, 0.0)
    pfe = np.quantile(exposure, alpha, axis=0)
    ee = exposure.mean(axis=0)
    peak_idx = int(np.argmax(pfe))
    return PFEResult(
        time_grid=grid,
        pfe=pfe,
        expected_exposure=ee,
        alpha=alpha,
        n_paths=int(values.shape[0]),
        peak_pfe=float(pfe[peak_idx]),
        peak_time=float(grid[peak_idx]),
    )


def _profile(
    payoff_level: npt.NDArray[np.float64],
    notional: float,
    time_grid: npt.NDArray[np.float64],
    alpha: float,
) -> PFEResult:
    """Shared tail: value = level * notional * remaining_fraction, then PFE."""
    remaining = _remaining_fraction(payoff_level.shape[1])
    value_paths = payoff_level * (notional * remaining)
    return pfe_from_value_paths(value_paths, time_grid, alpha=alpha)


def pfe_swing(
    swing_option: Any,
    price_paths: Any,
    time_grid: npt.ArrayLike,
    *,
    alpha: float = 0.95,
    n_paths: int | None = None,
    seed: int | None = None,
    notional: float | None = None,
) -> PFEResult:
    """PFE curve of a swing call over its life.

    Models the swing as a strip of daily calls struck at ``swing_option.strike``;
    the per-step payoff leg is ``max(price - strike, 0)`` scaled by the running
    remaining volume.

    Parameters
    ----------
    swing_option : object
        Duck-typed swing terms exposing ``strike`` and (for the default
        notional) ``volume_per_right`` and ``n_rights`` — matching
        ``mibel_derivatives.pricing.swing.SwingTerms``.
    price_paths : array (n_paths, n_steps) or callable
        Simulated spot-price paths aligned with ``time_grid`` (e.g. produced from
        the Pieza 2 Schwartz-Smith fit), or a ``simulate(n_paths, rng)`` callable.
    time_grid : array-like, shape (n_steps,)
        Exposure dates over the contract life.
    alpha : float, default 0.95
        PFE quantile.
    n_paths, seed : optional
        Used only when ``price_paths`` is a simulator callable.
    notional : float, optional
        Total contract volume (MWh). Defaults to
        ``volume_per_right * n_rights``.

    Returns
    -------
    PFEResult
    """
    _check_alpha(alpha)
    grid = np.asarray(time_grid, dtype=np.float64).ravel()
    paths = _resolve_paths(price_paths, n_paths, grid.size, seed)
    strike = float(swing_option.strike)
    if notional is None:
        notional = float(swing_option.volume_per_right) * float(swing_option.n_rights)
    level = np.maximum(paths - strike, 0.0)
    return _profile(level, float(notional), grid, alpha)


def pfe_tolling(
    tolling_agreement: Any,
    power_paths: Any,
    gas_curve: npt.ArrayLike,
    eua_curve: npt.ArrayLike,
    time_grid: npt.ArrayLike,
    *,
    alpha: float = 0.95,
    n_paths: int | None = None,
    seed: int | None = None,
    hours_per_step: float = 24.0,
    notional: float | None = None,
) -> PFEResult:
    """PFE curve of a CCGT tolling agreement over its life.

    Models tolling as a strip of spark-spread options; the per-step payoff leg is
    ``max(power - HR*(gas + co2*eua), 0)`` where the electrical heat rate
    ``HR = heat_rate_gj_per_mwh / GJ_PER_MWH`` converts fuel/carbon to EUR/MWh_e.

    Parameters
    ----------
    tolling_agreement : object
        Duck-typed agreement exposing ``effective_capacity_mw`` (or
        ``capacity_mw``) and an ``asset`` with ``heat_rate_full_gj_per_mwh`` and
        ``co2_intensity_t_per_mwh_th`` — matching
        ``mibel_derivatives.products.tolling.TollingAgreement``.
    power_paths : array (n_paths, n_steps) or callable
        Simulated power-price paths aligned with ``time_grid``.
    gas_curve, eua_curve : array-like, shape (n_steps,)
        Deterministic gas (EUR/MWh) and carbon (EUR/t) forward curves.
    time_grid : array-like, shape (n_steps,)
        Exposure dates.
    alpha : float, default 0.95
        PFE quantile.
    n_paths, seed : optional
        Used only when ``power_paths`` is a simulator callable.
    hours_per_step : float, default 24
        Running hours represented by one grid step (sets the volume scale).
    notional : float, optional
        Total dispatched volume (MWh). Defaults to
        ``capacity * hours_per_step * n_steps``.

    Returns
    -------
    PFEResult
    """
    _check_alpha(alpha)
    grid = np.asarray(time_grid, dtype=np.float64).ravel()
    paths = _resolve_paths(power_paths, n_paths, grid.size, seed)
    gas = np.asarray(gas_curve, dtype=np.float64).ravel()
    eua = np.asarray(eua_curve, dtype=np.float64).ravel()
    if gas.size != grid.size or eua.size != grid.size:
        raise ValueError("`gas_curve` and `eua_curve` must match `time_grid` length.")

    asset = tolling_agreement.asset
    heat_rate = float(asset.heat_rate_full_gj_per_mwh) / GJ_PER_MWH  # MWh_th / MWh_e
    co2 = float(asset.co2_intensity_t_per_mwh_th)
    capacity = float(
        getattr(tolling_agreement, "effective_capacity_mw", None)
        or tolling_agreement.capacity_mw
    )

    short_run_cost = heat_rate * (gas + co2 * eua)  # EUR/MWh_e, broadcast over steps
    spark = paths - short_run_cost  # (n_paths, n_steps)
    level = np.maximum(spark, 0.0)
    if notional is None:
        notional = capacity * hours_per_step * grid.size
    return _profile(level, float(notional), grid, alpha)


def pfe_ppa(
    ppa: Any,
    price_paths: Any,
    time_grid: npt.ArrayLike,
    *,
    alpha: float = 0.95,
    n_paths: int | None = None,
    seed: int | None = None,
    notional: float | None = None,
) -> PFEResult:
    """PFE curve of a solar PPA over its life.

    Models the spot-linked share of the PPA as a two-sided swap; the per-step
    payoff leg is ``spot_pct * (price - strike)`` (exposure can arise from either
    side of the strike, unlike the one-sided swing/tolling options).

    Parameters
    ----------
    ppa : object
        Duck-typed PPA exposing ``strike`` and ``spot_pct`` and (for the default
        notional) ``capacity_mw``, ``plant_factor`` and ``duration_years`` —
        matching ``mibel_derivatives.products.ppa.PPA``.
    price_paths : array (n_paths, n_steps) or callable
        Simulated spot-price paths aligned with ``time_grid``.
    time_grid : array-like, shape (n_steps,)
        Exposure dates over the contract life.
    alpha : float, default 0.95
        PFE quantile.
    n_paths, seed : optional
        Used only when ``price_paths`` is a simulator callable.
    notional : float, optional
        Total spot-linked generation (MWh). Defaults to
        ``capacity * plant_factor * 8760 * duration_years``.

    Returns
    -------
    PFEResult
    """
    _check_alpha(alpha)
    grid = np.asarray(time_grid, dtype=np.float64).ravel()
    paths = _resolve_paths(price_paths, n_paths, grid.size, seed)
    strike = float(ppa.strike)
    spot_pct = float(ppa.spot_pct)
    if notional is None:
        notional = (
            float(ppa.capacity_mw)
            * float(ppa.plant_factor)
            * 8760.0
            * float(ppa.duration_years)
        )
    level = spot_pct * (paths - strike)  # two-sided
    return _profile(level, float(notional), grid, alpha)
