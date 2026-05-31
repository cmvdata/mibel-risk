"""Potential Future Exposure (PFE) for structured energy derivatives.

PFE answers a counterparty-credit question: *if my counterparty defaults at some
future date, how much could the still-live contract be worth to me (a positive
replacement cost) at that date?* For each simulated market path the contract's
mark-to-market relative to inception is computed along a time grid; the positive
part is the **exposure**, and the ``alpha``-quantile of exposure across paths at
each grid date is the PFE curve ``PFE(t)``.

The estimators here are **standalone**: they consume simulated price paths (an
array, or a callable simulator) plus a duck-typed contract object, and never
import :mod:`mibel_derivatives`. The attribute names they read (``strike``,
``n_rights``, ``volume_per_right`` for swing; ``capacity_mw`` and ``asset`` for
tolling; ``strike``, ``spot_pct`` for PPA) match the corresponding
``mibel_derivatives`` product dataclasses, so a real product object passes
straight through while tests use lightweight stand-ins.

Exposure model
--------------
For a contract entered at fair value the inception mark-to-market is zero, so
exposure starts at zero, rises as the market diffuses, and amortises back to zero
as the remaining delivery volume runs off — the classic PFE *hump*. Concretely
``value_t = payoff_level(price_t) * remaining_volume(t)`` and the exposure is
``max(value_t - value_0, 0)`` (``value_0`` is the deterministic inception value,
common to all paths), guaranteeing ``PFE(0) = 0`` and ``PFE(T) = 0`` by
construction. See :mod:`mibel_risk.pfe.exposure` for the per-product payoff legs.
"""

from mibel_risk.pfe.exposure import (
    PFEResult,
    pfe_from_value_paths,
    pfe_ppa,
    pfe_swing,
    pfe_tolling,
)

__all__ = [
    "PFEResult",
    "pfe_from_value_paths",
    "pfe_ppa",
    "pfe_swing",
    "pfe_tolling",
]
