# Portfolio VaR — MIBEL energy book

**Module:** `mibel_risk.portfolio` · **Notebook:** `notebooks/01_portfolio_var.ipynb`
**Confidence level:** 95% · **Sign convention:** VaR reported as a positive loss (EUR).

This deliverable lifts the single-position VaR estimators in `mibel_risk.var` to
a **book** of heterogeneous contracts drawn from the `mibel_derivatives` stack,
and decomposes the portfolio VaR back onto the individual positions.

---

## 1 · What was built

| File | Role |
|------|------|
| `src/mibel_risk/portfolio/positions.py` | `Position(asset_type, notional, direction, valuation_fn)` and `portfolio_pnl` / `portfolio_pnl_vector` — scenario revaluation. |
| `src/mibel_risk/portfolio/var_portfolio.py` | `portfolio_var_historical`, `portfolio_var_monte_carlo`, `component_var`, `marginal_var`, plus the bundled `risk_decomposition`. |
| `tests/test_var_portfolio.py` | 20 tests: sub-additivity, linearity in notional, component/marginal/total consistency, historical vs Monte Carlo agreement, validation. |
| `notebooks/01_portfolio_var.ipynb` | Applies the engine to a 3-leg mock book (forward + swing + PPA). |
| `scripts/_build_portfolio_var_notebook.py` | Reproducible notebook builder (outputs stripped on commit). |

### Design — scenario revaluation
Every contract is a `Position` mapping **one price scenario** to a P&L number,
with `P&L = direction · notional · valuation_fn(scenario)`. All legs are valued
on the *same* scenario, so cross-contract correlation is preserved exactly and
the per-leg P&Ls add coherently. Keeping `notional` a separate factor makes the
book's P&L — and therefore its VaR — exactly linear in each position's size,
which the Euler/covariance decomposition relies on.

### Risk decomposition
Portfolio VaR is positively homogeneous of degree one in the notionals, so by
Euler's theorem it splits additively. We use the covariance (delta-normal)
split, which is **exact-additive for any total-VaR figure**:

```
component_i = [Cov(p_i, P) / Var(P)] · VaR_P ,   Σ_i component_i = VaR_P
marginal_i  = component_i / notional_i           (sensitivity per unit)
```

A negative component flags a **hedging** leg (P&L anti-correlated with the book).

---

## 2 · The mock book

| Leg | Instrument | Source (Pieza) | Sense |
|-----|-----------|----------------|-------|
| 1 | Monthly baseload **forward** | OMIP / Schwartz–Smith (Pieza 2) | long |
| 2 | Annual **swing call** | Pieza 3 | long |
| 3 | Solar **PPA**, spot-linked share | Pieza 5 | short spot |

The PPA's spot leg is deliberately a *short* power exposure, so it hedges the
long forward + swing — a realistic offtaker position that makes the
decomposition interesting.

---

## 3 · Historical VaR + backtest (OMIE spot 2019–2024)

OMIE Spain spot collapsed to a **daily baseload** price:

* Window: **2019-01-01 → 2024-12-31** (2 192 days)
* Mean level **85.17 EUR/MWh**; daily change std **19.77 EUR/MWh**

Delta-normal book, net spot delta **1 140 MWh**:

| Metric | Value (EUR / day) |
|--------|-------------------|
| Historical VaR (95%) | **33 250** |
| Gaussian VaR (95%) | 37 037 |
| Historical ES (95%) | 55 296 |

### Rolling 250-day backtest

| | Value |
|---|---|
| Backtest window | 2019-09-09 → 2024-12-31 (1 941 days) |
| Exceptions | **131 / 1 941 = 6.75%** (nominal 5%) |
| Kupiec POF | LR = 11.32, p = 0.001 → **reject** |
| Christoffersen independence | LR = 8.72, p = 0.003 → **reject** |
| Christoffersen cond. coverage | LR = 20.05, p < 0.001 → **reject** |

**Reading.** A *static* historical VaR under-covers MIBEL's clustered
volatility: it breaches 6.75% of days (not 5%) and the breaches cluster in time.
Both diagnostics correctly reject the model — exactly what the backtesting tools
exist to catch. The honest conclusion is that a constant-window historical VaR is
mis-specified for power; a volatility-scaled or EWMA filter is the natural next
step. The point here is that the engine *flags* the mis-specification rather than
hiding it.

---

## 4 · Monte Carlo VaR from the Schwartz–Smith fit (Pieza 2)

Loaded `forward_fit_production.pkl` (fit @ 2024-12-31):
`kappa = 0.384`, `sigma_chi = 1.021`, `sigma_xi = 0.231`, `rho = -0.806`;
last state `chi = -0.711`, `xi = 4.911`. Simulated **4 000 × 365** daily price
paths `S_t = exp(chi_t + xi_t)` (mean 88.9, p5 33.7, p95 190.9 EUR/MWh) and
fully revalued each leg.

* Monte Carlo portfolio **VaR(95%) = EUR 15 932** (path-generator API)
* Cross-check on the fixed simulated set: **EUR 16 021** (decomposition total)
* Expected Shortfall (95%): **EUR 19 003**; P&L std **EUR 11 196**

### Risk decomposition

| Leg | Notional | Standalone VaR | Component VaR | Marginal VaR | % contribution |
|-----|---------:|---------------:|--------------:|-------------:|---------------:|
| forward (long) | 1 200 | 16 791 | **14 508** | 12.09 | 91% |
| swing_call (long) | 3 | 17 222 | **10 231** | 3 410.31 | 64% |
| ppa (short) | 450 | 34 502 | **−8 717** | −19.37 | −54% |
| **PORTFOLIO** | — | 68 514 (gross) | **16 021** | — | 100% |

* `Σ component = EUR 16 021 ≡ portfolio VaR` (additivity holds to floating point).
* **Diversification benefit = EUR 52 493 (76.6% of gross standalone risk)** — the
  short PPA leg is the dominant hedge: although it carries the largest standalone
  VaR (34 502), its **negative** component (−8 717) pulls the book's risk below
  that of the forward leg alone.

---

## 5 · Decisions taken autonomously

1. **`src/` and `tests/` are standalone** — no import of `mibel_derivatives`, no
   raw data — so CI (which has neither) stays green. The sibling repo is consumed
   only inside the notebook via `sys.path`, with a portable upward search for the
   checkout (`MIBEL_DERIVATIVES` env var → sibling dir → fallback).
2. **`valuation_fn` returns per-unit P&L**, with `notional` a separate scalar, to
   make linearity-in-notional (and therefore the Euler decomposition) exact.
3. **Covariance (delta-normal) component split** chosen over a single-quantile
   conditional expectation: it is stable and *exactly* additive to any total-VaR
   figure, where the quantile estimator is noisy.
4. **Notebook valuations are analytic** (forward MtM; perfect-foresight swing
   intrinsic; spot-linked PPA) rather than the full LSM/DP pricers — fast, fully
   reproducible, and sufficient for a VaR demo. The production pricers are
   exercised in the PFE notebook (Deliverable 2).
5. **The backtest rejection is reported, not engineered away** — a faithful
   empirical result that validates the backtesting machinery.
6. **`pyarrow` added to the `notebooks` extra** (not the core/`dev` deps) so the
   notebook can read the curated parquet without touching CI's dependency set.
