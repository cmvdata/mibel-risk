# mibel-risk

**Market risk management for MIBEL energy portfolios** — Value-at-Risk (VaR),
Expected Shortfall (ES) and VaR backtesting for power and gas trading books
denominated in EUR/MWh.

This is **Module 4** of the MIBEL analytics stack, a standalone companion to:

- **`mibel-derivatives`** — pricing & sensitivities (swing options, CCGT tolling, solar PPAs)
- **`mibel-forecasting`** — price/load forecasting
- **`mibel-congestion-monitor`** — interconnection & congestion analytics

`mibel-risk` consumes the P&L / return series those modules produce and answers
the risk-management question: *how much can this book lose, and is our risk model
actually any good?*

## Positioning

Energy returns are sharply non-Gaussian — heavy tails, price spikes, regime
switches — so the library deliberately ships **three complementary VaR
estimators** and the **backtests** that tell you which one to trust on a given
book:

| Estimator | Assumption | Strength | Use when |
|-----------|------------|----------|----------|
| Historical | none (empirical) | honest about realised tails | enough history, stable regime |
| Parametric Gaussian | normal returns | cheap, transparent | quick desk-level checks |
| Parametric Student-t | fat-tailed | captures kurtosis | spiky power/gas returns |
| Monte Carlo | any path generator | reuses pricing-model simulators | structured / path-dependent books |

## Install

```bash
uv sync --extra dev          # runtime + dev tooling
uv sync --extra notebooks    # + JupyterLab
```

## Quickstart

```python
import numpy as np
from mibel_risk.var import (
    var_historical, expected_shortfall_historical,
    var_parametric_gaussian, var_parametric_t, var_monte_carlo,
)
from mibel_risk.backtesting import kupiec_pof, christoffersen_conditional_coverage

returns = np.random.default_rng(0).normal(0.0, 0.02, size=2_000)

# Three VaRs at 99% confidence
var_historical(returns, alpha=0.99)
var_parametric_gaussian(returns.mean(), returns.std(), alpha=0.99)
var_parametric_t(returns.mean(), returns.std(), df=5, alpha=0.99)
var_monte_carlo(lambda n, rng: rng.normal(0, 0.02, n), n_paths=1_000_000, alpha=0.99, seed=0)

# Backtest: are exceptions correctly sized AND independent in time?
exceptions = returns < -var_historical(returns, alpha=0.99)
christoffersen_conditional_coverage(exceptions, alpha=0.99)
```

All VaR/ES values are returned as **positive loss magnitudes** in the units of
the input returns. See module docstrings for sign conventions and formulae.

## Layout

```
src/mibel_risk/
  var/
    historical.py     # empirical-quantile VaR + ES
    parametric.py     # Gaussian & Student-t VaR + ES (closed form)
    monte_carlo.py    # path-agnostic simulation VaR
  backtesting/
    kupiec.py         # POF unconditional-coverage test
    christoffersen.py # independence + conditional-coverage tests
tests/                # pytest suite (mark: monte_carlo for slow stochastic tests)
notebooks/            # exploratory analysis
reports/              # generated figures & write-ups
data/curated/         # curated return/P&L series (gitignored)
```

## Roadmap

- [x] **v0.1** — VaR (historical / parametric / Monte Carlo) + ES + Kupiec & Christoffersen backtests
- [ ] **v0.2** — portfolio VaR: covariance & component/marginal VaR, correlation across MIBEL nodes
- [ ] **v0.3** — EWMA & GARCH volatility scaling; filtered historical simulation
- [ ] **v0.4** — Extreme Value Theory (POT/GPD) tail estimation for spike risk
- [ ] **v0.5** — integration with `mibel-derivatives` P&L vectors; book-level dashboards
- [ ] **v0.6** — regulatory reporting helpers (Basel traffic-light, ES at 97.5%)

## References

- Jorion, P. (2007). *Value at Risk: The New Benchmark for Managing Financial Risk* (3rd ed.).
- Christoffersen, P. (1998). "Evaluating interval forecasts." *Int. Economic Review* 39(4).
- Kupiec, P. (1995). "Techniques for verifying the accuracy of risk measurement models." *J. Derivatives* 3(2).
- McNeil, Frey & Embrechts (2015). *Quantitative Risk Management* (2nd ed.).

## License

MIT
