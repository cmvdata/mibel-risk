"""Tests for potential-future-exposure profiles.

Contracts are duck-typed via :class:`types.SimpleNamespace` stand-ins whose
attribute names match the real ``mibel_derivatives`` product dataclasses; price
paths are synthetic GBM-style simulations (all starting from a common spot, so
the inception column is deterministic and ``PFE(0)`` is exactly zero).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from mibel_risk.pfe import pfe_from_value_paths, pfe_ppa, pfe_swing, pfe_tolling

N_STEPS = 60
S0 = 60.0


def _price_paths(n_paths: int, n_steps: int = N_STEPS, sigma: float = 0.6, seed: int = 0) -> np.ndarray:
    """Driftless GBM paths, all starting at ``S0`` (column 0 deterministic)."""
    rng = np.random.default_rng(seed)
    dt = 1.0 / (n_steps - 1)
    shocks = rng.normal(0.0, sigma * np.sqrt(dt), size=(n_paths, n_steps - 1))
    incr = shocks - 0.5 * sigma**2 * dt
    log_path = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(incr, axis=1)], axis=1)
    return S0 * np.exp(log_path)


TIME_GRID = np.linspace(0.0, 1.0, N_STEPS)

SWING = SimpleNamespace(strike=60.0, volume_per_right=1.0, n_rights=100)
PPA = SimpleNamespace(strike=55.0, spot_pct=0.5, capacity_mw=50.0, plant_factor=0.2, duration_years=10)
TOLLING = SimpleNamespace(
    effective_capacity_mw=400.0,
    capacity_mw=400.0,
    asset=SimpleNamespace(heat_rate_full_gj_per_mwh=6.5, co2_intensity_t_per_mwh_th=0.2),
)
GAS = np.full(N_STEPS, 30.0)
EUA = np.full(N_STEPS, 70.0)


def _all_profiles(n_paths: int, alpha: float = 0.95, seed: int = 0) -> list:
    paths = _price_paths(n_paths, seed=seed)
    return [
        pfe_swing(SWING, paths, TIME_GRID, alpha=alpha),
        pfe_tolling(TOLLING, paths, GAS, EUA, TIME_GRID, alpha=alpha),
        pfe_ppa(PPA, paths, TIME_GRID, alpha=alpha),
    ]


# --------------------------------------------------------------------------- #
# PFE(0) ~ 0                                                                   #
# --------------------------------------------------------------------------- #
def test_pfe_zero_at_inception() -> None:
    """Exposure relative to inception is zero at t=0 for every product."""
    for res in _all_profiles(5_000):
        assert res.pfe[0] == pytest.approx(0.0, abs=1e-9)
        assert res.expected_exposure[0] == pytest.approx(0.0, abs=1e-9)
        assert res.time_grid[0] == 0.0


def test_pfe_zero_at_maturity() -> None:
    """Remaining volume runs off to zero, so exposure vanishes at maturity."""
    for res in _all_profiles(5_000):
        assert res.pfe[-1] == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# PFE >= 0 always                                                             #
# --------------------------------------------------------------------------- #
def test_pfe_nonnegative() -> None:
    for res in _all_profiles(5_000):
        assert np.all(res.pfe >= 0.0)
        assert np.all(res.expected_exposure >= 0.0)
        assert res.peak_pfe >= 0.0


# --------------------------------------------------------------------------- #
# Monotonicity in alpha                                                       #
# --------------------------------------------------------------------------- #
def test_pfe_monotonic_in_alpha() -> None:
    """A higher confidence level gives a (weakly) higher PFE at every date."""
    paths = _price_paths(8_000, seed=1)
    for fn, args in [
        (pfe_swing, (SWING, paths, TIME_GRID)),
        (pfe_ppa, (PPA, paths, TIME_GRID)),
    ]:
        lo = fn(*args, alpha=0.95)
        hi = fn(*args, alpha=0.99)
        assert np.all(hi.pfe >= lo.pfe - 1e-9)
        assert hi.peak_pfe >= lo.peak_pfe - 1e-9

    lo_t = pfe_tolling(TOLLING, paths, GAS, EUA, TIME_GRID, alpha=0.95)
    hi_t = pfe_tolling(TOLLING, paths, GAS, EUA, TIME_GRID, alpha=0.99)
    assert np.all(hi_t.pfe >= lo_t.pfe - 1e-9)


# --------------------------------------------------------------------------- #
# Convergence with n_paths                                                    #
# --------------------------------------------------------------------------- #
def test_pfe_converges_in_n_paths() -> None:
    """Two independent large-sample estimates of peak PFE agree closely."""
    peak_a = pfe_swing(SWING, _price_paths(40_000, seed=11), TIME_GRID).peak_pfe
    peak_b = pfe_swing(SWING, _price_paths(40_000, seed=22), TIME_GRID).peak_pfe
    assert peak_a == pytest.approx(peak_b, rel=0.03)


def test_pfe_monte_carlo_error_shrinks() -> None:
    """Cross-seed dispersion of the PFE peak falls as n_paths grows."""
    def spread(n: int) -> float:
        peaks = [
            pfe_ppa(PPA, _price_paths(n, seed=s), TIME_GRID).peak_pfe
            for s in range(6)
        ]
        return float(np.std(peaks))

    assert spread(40_000) < spread(2_500)


# --------------------------------------------------------------------------- #
# Shape: PFE humps in the interior                                            #
# --------------------------------------------------------------------------- #
def test_pfe_peaks_in_interior() -> None:
    """The diffusion-vs-amortisation trade-off puts the peak strictly inside."""
    for res in _all_profiles(20_000, seed=3):
        peak_idx = int(np.argmax(res.pfe))
        assert 0 < peak_idx < N_STEPS - 1


def test_peak_exceeds_endpoints() -> None:
    """The interior peak strictly dominates the (zero) endpoints."""
    for res in _all_profiles(10_000):
        assert res.peak_pfe > res.pfe[0]
        assert res.peak_pfe > res.pfe[-1]
        # At the peak date the exposure is well-spread, so PFE >= EE there.
        peak_idx = int(np.argmax(res.pfe))
        assert res.pfe[peak_idx] >= res.expected_exposure[peak_idx] - 1e-9


# --------------------------------------------------------------------------- #
# Simulator-callable path source                                              #
# --------------------------------------------------------------------------- #
def test_pfe_accepts_simulator_callable() -> None:
    """A ``simulate(n_paths, rng)`` callable is an accepted path source."""
    def sim(n: int, rng: np.random.Generator) -> np.ndarray:
        dt = 1.0 / (N_STEPS - 1)
        shocks = rng.normal(0.0, 0.6 * np.sqrt(dt), size=(n, N_STEPS - 1))
        log_path = np.concatenate([np.zeros((n, 1)), np.cumsum(shocks, axis=1)], axis=1)
        return S0 * np.exp(log_path)

    res = pfe_swing(SWING, sim, TIME_GRID, n_paths=5_000, seed=7)
    assert res.n_paths == 5_000
    assert res.pfe[0] == pytest.approx(0.0, abs=1e-9)
    assert res.peak_pfe > 0.0


def test_notional_scales_pfe_linearly() -> None:
    """PFE scales linearly with the contract notional."""
    paths = _price_paths(6_000, seed=4)
    base = pfe_swing(SWING, paths, TIME_GRID, notional=100.0)
    twice = pfe_swing(SWING, paths, TIME_GRID, notional=200.0)
    np.testing.assert_allclose(twice.pfe, 2.0 * base.pfe, rtol=1e-12)


# --------------------------------------------------------------------------- #
# Validation                                                                   #
# --------------------------------------------------------------------------- #
def test_invalid_alpha_raises() -> None:
    paths = _price_paths(100)
    for bad in (0.0, 1.0, -0.1, 1.2):
        with pytest.raises(ValueError):
            pfe_swing(SWING, paths, TIME_GRID, alpha=bad)


def test_mismatched_grid_raises() -> None:
    paths = _price_paths(100)
    with pytest.raises(ValueError):
        pfe_swing(SWING, paths, TIME_GRID[:-1])  # paths have N_STEPS columns


def test_mismatched_fuel_curve_raises() -> None:
    paths = _price_paths(100)
    with pytest.raises(ValueError):
        pfe_tolling(TOLLING, paths, GAS[:-1], EUA, TIME_GRID)


def test_simulator_without_n_paths_raises() -> None:
    with pytest.raises(ValueError):
        pfe_swing(SWING, lambda n, rng: _price_paths(n), TIME_GRID)


def test_non_finite_paths_raise() -> None:
    paths = _price_paths(100)
    paths[0, 5] = np.nan
    with pytest.raises(ValueError):
        pfe_swing(SWING, paths, TIME_GRID)


def test_single_point_grid_raises() -> None:
    with pytest.raises(ValueError):
        pfe_from_value_paths(np.ones((10, 1)), np.array([0.0]))
