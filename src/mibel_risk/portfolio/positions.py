"""Positions and P&L aggregation for an energy portfolio.

A :class:`Position` wraps a single contract as a mapping from a *price
scenario* to a profit-and-loss number. The scenario is whatever object the
position's ``valuation_fn`` understands — a daily spot path, an hourly
power-price array, a ``(power, gas, carbon)`` tuple of curves — so that
heterogeneous contracts (spot, forwards, swing options, tolling agreements,
PPAs) can be revalued **on the same simulated scenario** and their P&Ls added.
That shared-scenario revaluation is what preserves the cross-contract
correlation a portfolio VaR must capture: a single market draw moves every
position at once.

Per-unit valuation
------------------
``valuation_fn(scenario) -> float`` returns the P&L **per unit of notional**
(e.g. EUR per MWh of contracted volume, or EUR per MW of capacity). The
position's signed notional then scales it::

    position P&L = direction_sign * notional * valuation_fn(scenario)

Keeping ``notional`` a separate multiplicative factor (rather than baking it
into ``valuation_fn``) is what makes the book's P&L — and therefore its VaR —
exactly linear in each position's size, which the risk decomposition relies on.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

__all__ = [
    "ASSET_TYPES",
    "Position",
    "portfolio_pnl",
    "portfolio_pnl_vector",
]

#: The contract families understood by the portfolio layer. The label is
#: descriptive metadata (it does not change the maths — the payoff lives
#: entirely in ``valuation_fn``) but is validated so a book cannot silently
#: carry a typo'd or unsupported instrument type.
ASSET_TYPES: frozenset[str] = frozenset(
    {"spot_long", "forward", "swing_call", "tolling", "ppa"}
)

#: A scenario is opaque to this layer: each ``valuation_fn`` interprets it.
Scenario = Any
ValuationFn = Callable[[Scenario], float]

_DIRECTION_SIGN: dict[str, float] = {"long": 1.0, "short": -1.0}


def _direction_to_sign(direction: str | float | int) -> float:
    """Normalise ``direction`` to a signed multiplier.

    Accepts the strings ``"long"`` / ``"short"`` or any non-zero number (whose
    sign — and magnitude, allowing fractional/levered books — is used directly).
    """
    if isinstance(direction, str):
        key = direction.strip().lower()
        if key not in _DIRECTION_SIGN:
            raise ValueError(
                f"`direction` string must be 'long' or 'short'; got {direction!r}."
            )
        return _DIRECTION_SIGN[key]
    if isinstance(direction, bool):  # guard: bool is an int subclass
        raise TypeError("`direction` must be 'long'/'short' or a number, not bool.")
    value = float(direction)
    if value == 0.0:
        raise ValueError("`direction` number must be non-zero.")
    return value


@dataclass(frozen=True)
class Position:
    """A single contract in the book.

    Attributes
    ----------
    asset_type : str
        One of :data:`ASSET_TYPES` (``'spot_long'``, ``'forward'``,
        ``'swing_call'``, ``'tolling'``, ``'ppa'``).
    notional : float
        Position size in the contract's natural unit (MWh of volume, MW of
        capacity, number of options, ...). Must be non-negative; the long/short
        sense lives in ``direction``.
    direction : str or float
        ``'long'`` / ``'short'`` (mapped to ``+1`` / ``-1``) or a numeric
        signed multiplier for levered/fractional books.
    valuation_fn : callable
        ``valuation_fn(scenario) -> float`` returning the **per-unit** P&L of
        the contract under that price scenario.
    """

    asset_type: str
    notional: float
    direction: str | float | int
    valuation_fn: ValuationFn

    def __post_init__(self) -> None:
        if self.asset_type not in ASSET_TYPES:
            raise ValueError(
                f"`asset_type` must be one of {sorted(ASSET_TYPES)}; got "
                f"{self.asset_type!r}."
            )
        if not np.isfinite(self.notional) or self.notional < 0.0:
            raise ValueError(
                f"`notional` must be a finite, non-negative number; got "
                f"{self.notional!r}."
            )
        if not callable(self.valuation_fn):
            raise TypeError("`valuation_fn` must be callable.")
        # Validate `direction` eagerly so construction fails fast.
        _direction_to_sign(self.direction)

    @property
    def sign(self) -> float:
        """Signed notional multiplier (``+1`` long, ``-1`` short)."""
        return _direction_to_sign(self.direction)

    @property
    def signed_notional(self) -> float:
        """``sign * notional`` — the size that scales the per-unit P&L."""
        return self.sign * float(self.notional)

    def pnl(self, scenario: Scenario) -> float:
        """P&L of this position under one price ``scenario`` (EUR)."""
        return float(self.signed_notional * self.valuation_fn(scenario))


def portfolio_pnl(positions: Sequence[Position], scenario: Scenario) -> float:
    """Total book P&L under a single price ``scenario``.

    Parameters
    ----------
    positions : sequence of Position
        The book. An empty book has zero P&L.
    scenario : object
        One price scenario, passed verbatim to every position's
        ``valuation_fn``. All positions must agree on the scenario's structure.

    Returns
    -------
    float
        Sum of the individual position P&Ls (EUR), gains positive.
    """
    return float(sum(pos.pnl(scenario) for pos in positions))


def portfolio_pnl_vector(
    positions: Sequence[Position], scenarios: Iterable[Scenario]
) -> npt.NDArray[np.float64]:
    """Book P&L across many scenarios.

    Parameters
    ----------
    positions : sequence of Position
        The book.
    scenarios : iterable of scenario objects
        Each yields one portfolio P&L. May be a sequence of arrays, a 2-D array
        iterated row-wise, or any iterable of scenario objects.

    Returns
    -------
    numpy.ndarray
        1-D array of portfolio P&Ls, one per scenario (gains positive).
    """
    return np.array(
        [portfolio_pnl(positions, scenario) for scenario in scenarios],
        dtype=np.float64,
    )
