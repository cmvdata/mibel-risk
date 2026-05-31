"""Builder for notebooks/01_portfolio_var.ipynb (run once, then strip outputs)."""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

cells: list = []
md = lambda s: cells.append(new_markdown_cell(s))  # noqa: E731
co = lambda s: cells.append(new_code_cell(s))  # noqa: E731

md(
    """# 01 · Portfolio VaR — MIBEL energy book

This notebook applies `mibel_risk` to a concrete three-instrument book drawn
from the `mibel_derivatives` stack:

| Leg | Instrument | Pieza | Sense |
|-----|-----------|-------|-------|
| 1 | Monthly baseload **forward** | OMIP / Pieza 2 | long |
| 2 | Annual **swing call** | Pieza 3 | long |
| 3 | Solar **PPA** (offtaker, spot-linked) | Pieza 5 | short spot |

We produce two complementary VaR views and a backtest:

1. **Historical VaR + backtest** on OMIE spot 2019–2024 — a delta-normal book,
   rolling 250-day historical VaR, validated with Kupiec (coverage) and
   Christoffersen (independence / conditional coverage).
2. **Monte Carlo VaR** from the **Schwartz–Smith** production fit (Pieza 2,
   `forward_fit_production.pkl`): simulate daily price paths, fully revalue each
   leg, and **decompose** the portfolio VaR into component and marginal VaR.

> The book is a *mock* portfolio chosen to be economically sensible and to
> exercise every part of the risk engine; it is not a live position.
"""
)

co(
    '''import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _find_derivatives() -> Path:
    """Locate the sibling mibel_derivatives checkout (data + price models)."""
    env = os.environ.get("MIBEL_DERIVATIVES")
    if env and Path(env).exists():
        return Path(env)
    here = Path.cwd().resolve()
    for base in [here, *here.parents]:
        for cand in (base / "Mibel_derivatives", base.parent / "Mibel_derivatives"):
            if cand.exists():
                return cand
    return Path(r"C:\\Users\\Carlo\\Desktop\\Projects\\Mibel_derivatives")


# mibel_risk is the installed package; mibel_derivatives is the sibling repo we
# read the production fit and price model from (added to the path, not imported
# as a dependency — mibel_risk stays standalone).
DERIV = _find_derivatives()
if str(DERIV / "src") not in sys.path:
    sys.path.insert(0, str(DERIV / "src"))

from mibel_derivatives.models import forward  # noqa: E402

from mibel_risk.portfolio import (  # noqa: E402
    Position,
    portfolio_pnl_vector,
    portfolio_var_historical,
    portfolio_var_monte_carlo,
    risk_decomposition,
)
from mibel_risk.var import (  # noqa: E402
    expected_shortfall_historical,
    var_parametric_gaussian,
)
from mibel_risk.backtesting import (  # noqa: E402
    christoffersen_conditional_coverage,
    christoffersen_independence,
    kupiec_pof,
)

ALPHA = 0.95
CURATED = DERIV / "data" / "curated"
pd.set_option("display.float_format", lambda v: f"{v:,.2f}")
np.set_printoptions(suppress=True)
print("imports OK")'''
)

md(
    """## 1 · Historical VaR + backtest on OMIE 2019–2024

We load the curated OMIE Spain spot series, collapse it to a **daily baseload**
price, and form daily price changes `dS_t`. The book is represented in
delta-normal form: each leg carries a signed MWh exposure to `dS`, so the daily
P&L is `sum_i sign_i * delta_i * dS_t`. This is exactly the
`Position` / `portfolio_*` machinery with a scalar scenario `dS`.
"""
)

co(
    '''omie = pd.read_parquet(CURATED / "omie_spot_es_2019_2024.parquet")
omie["datetime_utc"] = pd.to_datetime(omie["datetime_utc"], utc=True)
daily = (
    omie.set_index("datetime_utc")["price_eur_mwh"]
    .resample("1D")
    .mean()
    .dropna()
)
dS = daily.diff().dropna()
print(f"OMIE daily baseload: {daily.index.min().date()} -> {daily.index.max().date()} "
      f"({len(daily)} days)")
print(f"mean = {daily.mean():,.2f} EUR/MWh | daily change std = {dS.std():,.2f} EUR/MWh")
daily.plot(figsize=(10, 3), title="OMIE daily baseload spot (EUR/MWh)")
plt.tight_layout(); plt.show()'''
)

co(
    '''# Delta-normal book: signed MWh exposure of each leg to the daily spot move.
# (forward = full baseload delta; swing = delta-adjusted; PPA = short spot hedge.)
def _identity(ds: float) -> float:
    return float(ds)  # per-MWh P&L = the price change itself

book_delta = [
    Position("forward", notional=1200.0, direction="long", valuation_fn=_identity),
    Position("swing_call", notional=540.0, direction="long", valuation_fn=_identity),
    Position("ppa", notional=600.0, direction="short", valuation_fn=_identity),
]
labels = ["forward", "swing_call", "ppa(short)"]
net_delta = sum(p.signed_notional for p in book_delta)
print(f"net spot delta = {net_delta:,.0f} MWh")

scenarios = dS.to_numpy()  # one scenario per historical day
pnl_hist = portfolio_pnl_vector(book_delta, scenarios)

var_h = portfolio_var_historical(book_delta, scenarios, alpha=ALPHA)
es_h = expected_shortfall_historical(pnl_hist, alpha=ALPHA)
var_g = var_parametric_gaussian(pnl_hist.mean(), pnl_hist.std(ddof=1), alpha=ALPHA)
print(f"Historical VaR(95%)   = EUR {var_h:,.0f} / day")
print(f"Gaussian   VaR(95%)   = EUR {var_g:,.0f} / day")
print(f"Historical ES(95%)    = EUR {es_h:,.0f} / day")'''
)

md(
    """### 1.1 · Rolling backtest

We forecast a one-day VaR each day from a trailing 250-day window of historical
P&L, flag an **exception** when the realised loss breaches it, and feed the
exception series to Kupiec and Christoffersen. A well-specified 95% model breaches
on ~5% of days, with the breaches spread out in time (independent).
"""
)

co(
    '''WINDOW = 250
pnl_series = pd.Series(pnl_hist, index=dS.index)

var_fc, realised = [], []
for t in range(WINDOW, len(pnl_series)):
    window = pnl_series.iloc[t - WINDOW:t].to_numpy()
    v = -np.quantile(window, 1.0 - ALPHA)  # one-day historical VaR (positive loss)
    var_fc.append(v)
    realised.append(pnl_series.iloc[t])

var_fc = np.asarray(var_fc)
realised = np.asarray(realised)
exceptions = realised < -var_fc
bt_index = pnl_series.index[WINDOW:]

n = exceptions.size
rate = exceptions.mean()
print(f"backtest window: {bt_index.min().date()} -> {bt_index.max().date()} ({n} days)")
print(f"exceptions: {int(exceptions.sum())} / {n}  ({rate:.2%}; nominal {1 - ALPHA:.0%})")

kp = kupiec_pof(exceptions, alpha=ALPHA)
ci = christoffersen_independence(exceptions)
cc = christoffersen_conditional_coverage(exceptions, alpha=ALPHA)
print(f"Kupiec POF          LR={kp.lr_stat:6.3f}  p={kp.p_value:.3f}  reject={kp.reject}")
print(f"Christoffersen ind. LR={ci.lr_stat:6.3f}  p={ci.p_value:.3f}  reject={ci.reject}")
print(f"Christoffersen cc   LR={cc.lr_stat:6.3f}  p={cc.p_value:.3f}  reject={cc.reject}")'''
)

co(
    '''fig, ax = plt.subplots(figsize=(11, 3.5))
ax.plot(bt_index, realised, lw=0.6, color="0.5", label="daily P&L")
ax.plot(bt_index, -var_fc, lw=1.0, color="tab:blue", label="-VaR(95%) forecast")
ax.scatter(bt_index[exceptions], realised[exceptions], s=14, color="tab:red",
           zorder=3, label="exception")
ax.set_title("Rolling 250-day historical VaR backtest — OMIE book")
ax.set_ylabel("EUR / day"); ax.legend(loc="lower left", ncol=3, fontsize=8)
plt.tight_layout(); plt.show()'''
)

md(
    """## 2 · Monte Carlo VaR from the Schwartz–Smith fit (Pieza 2)

We load the production two-factor fit and simulate daily price paths
`S_t = exp(chi_t + xi_t)` from its last filtered state. Each leg is then **fully
revalued** on every path (non-linear payoffs), giving the portfolio P&L
distribution and a full risk decomposition.
"""
)

co(
    '''import pickle  # noqa: E402

with open(CURATED / "forward_fit_production.pkl", "rb") as fh:
    ss = pickle.load(fh)
p = ss.params
chi0 = float(ss.state_chi.iloc[-1])
xi0 = float(ss.state_xi.iloc[-1])
start = pd.Timestamp(ss.trade_dates[-1])
print(f"SS fit @ {start.date()}: kappa={p.kappa:.3f} sigma_chi={p.sigma_chi:.3f} "
      f"sigma_xi={p.sigma_xi:.3f} rho={p.rho:.3f}")
print(f"last state: chi={chi0:.3f}  xi={xi0:.3f}")

H = 365            # daily horizon (one delivery year)
N_PATHS = 4000     # Monte Carlo paths

def simulate_spot_paths(n_paths: int, seed: int) -> np.ndarray:
    chi, xi = forward.simulate(p, chi0, xi0, start, H, n_paths, seed=seed)
    return np.exp(chi + xi)  # (n_paths, H) daily spot proxy, EUR/MWh

S = simulate_spot_paths(N_PATHS, seed=2024)
print(f"simulated spot paths: shape={S.shape}  "
      f"mean={S.mean():,.1f}  p5={np.quantile(S, 0.05):,.1f}  p95={np.quantile(S, 0.95):,.1f}")'''
)

co(
    '''# --- Leg valuation functions, each mapping ONE daily spot path -> per-unit P&L ---
FWD_DAYS = 30          # monthly baseload forward
SWING_K = 65.0         # swing strike (EUR/MWh)
SWING_Q = 1.0          # MWh per exercised right
SWING_R = 90           # number of rights over the year
PPA_SPOT_PCT = 0.5     # spot-linked share of the PPA

# Calibrate the fair levels (strikes / premium) on the simulated set so each leg
# is marked at par at inception -> P&L is the mark-to-market deviation.
F0 = float(S[:, :FWD_DAYS].mean())          # forward fair price
PPA_REF = float(S.mean())                   # PPA spot reference

def swing_intrinsic(path: np.ndarray) -> float:
    """Perfect-foresight intrinsic value: exercise the R most in-the-money days.
    (An upper bound on the LSM value; the full pricer is used in notebook 02.)"""
    payoff = np.maximum(path - SWING_K, 0.0) * SWING_Q
    if SWING_R >= payoff.size:
        return float(payoff.sum())
    top = np.argpartition(payoff, -SWING_R)[-SWING_R:]
    return float(payoff[top].sum())

SWING_PREMIUM = float(np.mean([swing_intrinsic(path) for path in S]))

def fwd_unit_pnl(path: np.ndarray) -> float:
    return float(path[:FWD_DAYS].mean() - F0)          # per MWh

def swing_unit_pnl(path: np.ndarray) -> float:
    return swing_intrinsic(path) - SWING_PREMIUM       # per contract

def ppa_unit_pnl(path: np.ndarray) -> float:
    return float(path.mean() - PPA_REF)                # per MWh of spot exposure

# Notionals are chosen so the three legs carry *comparable* standalone risk
# (the swing P&L is per-contract = 90 rights, hence a small contract count).
book_mc = [
    Position("forward", notional=1200.0, direction="long", valuation_fn=fwd_unit_pnl),
    Position("swing_call", notional=3.0, direction="long", valuation_fn=swing_unit_pnl),
    Position("ppa", notional=900.0 * PPA_SPOT_PCT, direction="short", valuation_fn=ppa_unit_pnl),
]
print(f"forward fair F0 = {F0:,.2f} | swing premium = {SWING_PREMIUM:,.2f} | PPA ref = {PPA_REF:,.2f}")'''
)

co(
    '''# Monte Carlo VaR via the path-generator API (fresh draws each call)...
def simulate_scenarios(n_paths: int, rng: np.random.Generator):
    seed = int(rng.integers(0, 2**32 - 1))
    return simulate_spot_paths(n_paths, seed=seed)

var_mc = portfolio_var_monte_carlo(book_mc, simulate_scenarios, N_PATHS, alpha=ALPHA, seed=7)

# ...and the full decomposition on the fixed simulated set S.
scen_mc = list(S)
pnl_mc = portfolio_pnl_vector(book_mc, scen_mc)
rd = risk_decomposition(book_mc, scen_mc, alpha=ALPHA)

print(f"Monte Carlo portfolio VaR(95%) = EUR {var_mc:,.0f}")
print(f"Expected Shortfall ES(95%)     = EUR {expected_shortfall_historical(pnl_mc, ALPHA):,.0f}")
print(f"mean P&L = EUR {pnl_mc.mean():,.0f}  |  std = EUR {pnl_mc.std():,.0f}")'''
)

md("### 2.1 · Risk decomposition by position")

co(
    '''labels_mc = ["forward (long)", "swing_call (long)", "ppa (short)"]
decomp = pd.DataFrame(
    {
        "notional": [p.notional for p in book_mc],
        "standalone_VaR": rd.standalone_var,
        "component_VaR": rd.component_var,
        "marginal_VaR": rd.marginal_var,
        "pct_contribution": rd.pct_contribution,
    },
    index=labels_mc,
)
decomp.loc["PORTFOLIO"] = [
    np.nan, rd.standalone_var.sum(), rd.total_var, np.nan, 1.0
]
print(decomp.to_string())
print(f"\\nsum(component) = EUR {rd.component_var.sum():,.0f}  ==  total VaR EUR {rd.total_var:,.0f}")
print(f"diversification benefit = EUR {rd.diversification_benefit:,.0f} "
      f"({rd.diversification_benefit / rd.standalone_var.sum():.1%} of gross)")'''
)

co(
    '''fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
colors = ["tab:blue" if c >= 0 else "tab:red" for c in rd.component_var]
axes[0].bar(labels_mc, rd.component_var, color=colors)
axes[0].axhline(0, color="0.3", lw=0.8)
axes[0].set_title("Component VaR (sums to portfolio VaR)")
axes[0].set_ylabel("EUR"); axes[0].tick_params(axis="x", rotation=20)

axes[1].bar(labels_mc, rd.standalone_var, color="0.6", label="standalone")
axes[1].axhline(rd.total_var, color="tab:green", lw=1.5, ls="--",
                label=f"portfolio VaR = {rd.total_var:,.0f}")
axes[1].set_title("Standalone VaR vs diversified portfolio VaR")
axes[1].set_ylabel("EUR"); axes[1].tick_params(axis="x", rotation=20); axes[1].legend(fontsize=8)
plt.tight_layout(); plt.show()'''
)

md(
    """## 3 · Summary

* **Historical VaR** on the OMIE-driven delta-normal book gives a daily 95% loss
  figure; the rolling backtest's Kupiec and Christoffersen statistics validate
  the model's coverage and the time-independence of its breaches.
* **Monte Carlo VaR** from the Schwartz–Smith production fit fully revalues the
  three legs over simulated price paths. The decomposition shows where the risk
  sits (component VaR, summing exactly to the total) and how much the short PPA
  leg hedges the long forward + swing (diversification benefit).

Key figures are collected in `reports/diagnostics/portfolio_var.md`.
"""
)

nb = new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.metadata["language_info"] = {"name": "python", "version": "3.11"}
out = Path("notebooks/01_portfolio_var.ipynb")
out.parent.mkdir(exist_ok=True)
nbf.write(nb, str(out))
print(f"wrote {out} with {len(cells)} cells")
