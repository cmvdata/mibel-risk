"""Tests for the discrete CVA engine."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from mibel_risk.cva import (
    DefaultCurve,
    cva,
    discount_from_rate,
    survival_from_hazard,
)

N = 41
TIME_GRID = np.linspace(0.0, 5.0, N)  # 5-year horizon


def _ee_hump() -> np.ndarray:
    """A PFE-style expected-exposure hump: zero at the ends, peak mid-life."""
    x = TIME_GRID / TIME_GRID[-1]
    ee = 1_000_000.0 * np.sin(np.pi * x)  # 0 at t0 and tN, peak at mid
    return ee


EE = _ee_hump()


# --------------------------------------------------------------------------- #
# CVA >= 0                                                                     #
# --------------------------------------------------------------------------- #
def test_cva_nonnegative() -> None:
    res = cva(TIME_GRID, EE, default_curve=0.02, recovery=0.4)
    assert res.cva >= 0.0
    assert np.all(res.contributions >= 0.0)


def test_contributions_sum_to_cva() -> None:
    res = cva(TIME_GRID, EE, default_curve=0.03, recovery=0.4)
    assert res.contributions.sum() == pytest.approx(res.cva, rel=1e-12)


# --------------------------------------------------------------------------- #
# Linearity in (1 - recovery)                                                  #
# --------------------------------------------------------------------------- #
def test_linear_in_loss_given_default() -> None:
    """CVA is exactly proportional to LGD = 1 - recovery."""
    base = cva(TIME_GRID, EE, default_curve=0.02, recovery=0.0).cva  # LGD = 1
    for recovery in (0.2, 0.4, 0.6, 0.9):
        res = cva(TIME_GRID, EE, default_curve=0.02, recovery=recovery).cva
        assert res == pytest.approx((1.0 - recovery) * base, rel=1e-12)


def test_recovery_one_gives_zero() -> None:
    """Full recovery -> no loss -> CVA = 0."""
    assert cva(TIME_GRID, EE, default_curve=0.05, recovery=1.0).cva == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# Monotonicity in hazard rate                                                  #
# --------------------------------------------------------------------------- #
def test_monotonic_in_hazard() -> None:
    """Higher hazard -> more default probability -> larger CVA."""
    hazards = [0.005, 0.01, 0.02, 0.05, 0.1]
    cvas = [cva(TIME_GRID, EE, default_curve=h, recovery=0.4).cva for h in hazards]
    assert all(b > a for a, b in pairwise(cvas))


# --------------------------------------------------------------------------- #
# Limit: hazard = 0 -> CVA = 0                                                  #
# --------------------------------------------------------------------------- #
def test_zero_hazard_gives_zero_cva() -> None:
    res = cva(TIME_GRID, EE, default_curve=0.0, recovery=0.4)
    assert res.cva == pytest.approx(0.0)
    assert res.total_default_prob == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# Discounting                                                                  #
# --------------------------------------------------------------------------- #
def test_discounting_reduces_cva() -> None:
    """A positive discount rate lowers CVA versus undiscounted."""
    undiscounted = cva(TIME_GRID, EE, default_curve=0.03, recovery=0.4).cva
    df = discount_from_rate(TIME_GRID, 0.05)
    discounted = cva(TIME_GRID, EE, default_curve=0.03, recovery=0.4, discount_curve=df).cva
    assert discounted < undiscounted
    assert discounted > 0.0


# --------------------------------------------------------------------------- #
# Piecewise hazard                                                             #
# --------------------------------------------------------------------------- #
def test_piecewise_hazard_between_flat_bounds() -> None:
    """A piecewise hazard rising from 0.01 to 0.05 yields a CVA between the two
    flat-hazard CVAs."""
    low = cva(TIME_GRID, EE, default_curve=0.01, recovery=0.4).cva
    high = cva(TIME_GRID, EE, default_curve=0.05, recovery=0.4).cva
    ramp = np.linspace(0.01, 0.05, N - 1)
    mid = cva(TIME_GRID, EE, default_curve=ramp, recovery=0.4).cva
    assert low < mid < high


def test_default_curve_object_equivalent_to_scalar() -> None:
    a = cva(TIME_GRID, EE, default_curve=0.02, recovery=0.4).cva
    b = cva(TIME_GRID, EE, default_curve=DefaultCurve(0.02), recovery=0.4).cva
    assert a == pytest.approx(b, rel=1e-12)


# --------------------------------------------------------------------------- #
# Survival / discount helpers                                                  #
# --------------------------------------------------------------------------- #
def test_survival_constant_hazard_closed_form() -> None:
    """Constant-hazard survival matches exp(-lambda t)."""
    s = survival_from_hazard(TIME_GRID, 0.03)
    np.testing.assert_allclose(s, np.exp(-0.03 * TIME_GRID), rtol=1e-12)
    assert s[0] == 1.0


def test_survival_monotone_non_increasing() -> None:
    s = survival_from_hazard(TIME_GRID, np.linspace(0.0, 0.1, N - 1))
    assert np.all(np.diff(s) <= 1e-15)


def test_total_default_prob_matches_survival() -> None:
    res = cva(TIME_GRID, EE, default_curve=0.04, recovery=0.4)
    s = survival_from_hazard(TIME_GRID, 0.04)
    assert res.total_default_prob == pytest.approx(1.0 - s[-1], rel=1e-12)


# --------------------------------------------------------------------------- #
# Validation                                                                   #
# --------------------------------------------------------------------------- #
def test_bad_recovery_raises() -> None:
    for bad in (-0.1, 1.5):
        with pytest.raises(ValueError):
            cva(TIME_GRID, EE, default_curve=0.02, recovery=bad)


def test_negative_exposure_raises() -> None:
    bad_ee = EE.copy()
    bad_ee[5] = -1.0
    with pytest.raises(ValueError):
        cva(TIME_GRID, bad_ee, default_curve=0.02, recovery=0.4)


def test_negative_hazard_raises() -> None:
    with pytest.raises(ValueError):
        cva(TIME_GRID, EE, default_curve=-0.01, recovery=0.4)


def test_mismatched_exposure_length_raises() -> None:
    with pytest.raises(ValueError):
        cva(TIME_GRID, EE[:-1], default_curve=0.02, recovery=0.4)


def test_mismatched_discount_length_raises() -> None:
    with pytest.raises(ValueError):
        cva(TIME_GRID, EE, default_curve=0.02, recovery=0.4, discount_curve=np.ones(N - 1))


def test_single_point_grid_raises() -> None:
    with pytest.raises(ValueError):
        cva(np.array([0.0]), np.array([1.0]), default_curve=0.02)
