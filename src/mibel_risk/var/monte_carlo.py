"""Monte Carlo Value-at-Risk and Expected Shortfall.

Monte Carlo VaR draws a large sample of simulated period returns from a
user-supplied generator and applies the historical-simulation estimator to the
draws. The method is *path-agnostic*: any callable that returns an array of
simulated returns plugs in unchanged — a Gaussian/Student-t sampler, a fitted
Schwartz-Smith two-factor price model, a GARCH path generator, or a full
portfolio revaluation engine. This is what lets the same VaR machinery serve
both a single forward position and a structured book.

References
----------
Jorion, P. (2007). *Value at Risk* (3rd ed.), McGraw-Hill, Ch. 12
(Monte Carlo methods). Glasserman (2003), *Monte Carlo Methods in Financial
Engineering*, Ch. 1 & 9.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import numpy.typing as npt

from mibel_risk.var.historical import (
    expected_shortfall_historical,
    var_historical,
)

__all__ = ["var_monte_carlo"]

SimulateFn = Callable[[int, np.random.Generator], npt.ArrayLike]


def var_monte_carlo(
    simulate_fn: SimulateFn,
    n_paths: int,
    alpha: float = 0.95,
    *,
    seed: int | None = None,
    return_es: bool = False,
) -> float | tuple[float, float]:
    """Monte Carlo Value-at-Risk via a pluggable path generator.

    The simulator is called once to produce ``n_paths`` simulated returns; VaR
    (and optionally ES) is then the empirical tail statistic of that sample, so
    the estimate converges to the true VaR at the usual Monte Carlo rate
    :math:`O(n^{-1/2})`.

    Parameters
    ----------
    simulate_fn : callable
        Path generator with signature ``simulate_fn(n_paths, rng) -> array``,
        returning an array-like of ``n_paths`` simulated period returns (gains
        positive, losses negative). It receives a seeded
        :class:`numpy.random.Generator` so that, given ``seed``, results are
        fully reproducible. A generator that ignores the ``rng`` argument is
        allowed but then ``seed`` has no effect.
    n_paths : int
        Number of simulated returns to draw; must be positive. Larger values
        shrink the Monte Carlo error.
    alpha : float, default 0.95
        Confidence level in ``(0, 1)``.
    seed : int, optional
        Seed for the :class:`numpy.random.Generator` handed to ``simulate_fn``.
        Pass an int for reproducible VaR; leave ``None`` for fresh randomness.
    return_es : bool, default False
        If ``True``, return ``(var, es)`` where ``es`` is the Monte Carlo
        Expected Shortfall at the same ``alpha``.

    Returns
    -------
    float or tuple[float, float]
        The VaR loss magnitude, or ``(var, es)`` when ``return_es`` is set.

    Raises
    ------
    ValueError
        If ``n_paths`` is not positive, or if ``simulate_fn`` returns a sample
        whose length does not match ``n_paths``.

    Examples
    --------
    >>> import numpy as np
    >>> sim = lambda n, rng: rng.normal(0.0, 0.02, size=n)
    >>> v = var_monte_carlo(sim, n_paths=200_000, alpha=0.99, seed=0)
    >>> round(v, 3)  # ~ 0.02 * Phi^{-1}(0.99) = 0.0465
    0.047
    """
    if n_paths <= 0:
        raise ValueError(f"`n_paths` must be positive; got {n_paths}.")

    rng = np.random.default_rng(seed)
    draws = np.asarray(simulate_fn(n_paths, rng), dtype=np.float64).ravel()
    if draws.size != n_paths:
        raise ValueError(
            f"`simulate_fn` returned {draws.size} samples but `n_paths`={n_paths}."
        )

    var = var_historical(draws, alpha=alpha)
    if return_es:
        es = expected_shortfall_historical(draws, alpha=alpha)
        return var, es
    return var
