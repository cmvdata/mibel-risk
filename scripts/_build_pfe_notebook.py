"""Builder for notebooks/02_pfe_derivados.ipynb (run once, then strip outputs)."""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

cells: list = []
md = lambda s: cells.append(new_markdown_cell(s))  # noqa: E731
co = lambda s: cells.append(new_code_cell(s))  # noqa: E731

md(
    """# 02 · PFE on structured derivatives — MIBEL book

**Potential Future Exposure** (PFE) for the three structured contracts of the
`mibel_derivatives` stack:

| Contract | Pieza | Exposure leg |
|----------|-------|--------------|
| Annual **swing call** | Pieza 3 | strip of daily calls `max(S − K, 0)` |
| CCGT **tolling** (Castejón) | Pieza 4 | strip of spark-spread options `max(spark, 0)` |
| Solar **PPA** | Pieza 5 | two-sided spot-vs-fixed swap `spot_pct·(S − K)` |

We load the **Schwartz–Smith production fit** (`forward_fit_production.pkl`,
Pieza 2), simulate daily price paths, and compute each contract's PFE curve
`PFE(t)` over its life with `mibel_risk.pfe`. The real `mibel_derivatives`
product objects are passed straight to the PFE estimators (duck-typed — no
import dependency).

> Exposure is the positive mark-to-market relative to inception, so every curve
> starts at zero, humps as the market diffuses, and amortises back to zero as the
> remaining delivery volume runs off.
"""
)

co(
    '''import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _find_derivatives() -> Path:
    env = os.environ.get("MIBEL_DERIVATIVES")
    if env and Path(env).exists():
        return Path(env)
    here = Path.cwd().resolve()
    for base in [here, *here.parents]:
        for cand in (base / "Mibel_derivatives", base.parent / "Mibel_derivatives"):
            if cand.exists():
                return cand
    return Path(r"C:\\Users\\Carlo\\Desktop\\Projects\\Mibel_derivatives")


DERIV = _find_derivatives()
if str(DERIV / "src") not in sys.path:
    sys.path.insert(0, str(DERIV / "src"))

from mibel_derivatives.models import forward  # noqa: E402
from mibel_derivatives.pricing.swing import SwingTerms  # noqa: E402
from mibel_derivatives.products.ppa import PPA  # noqa: E402
from mibel_derivatives.products.tolling import TollingAgreement  # noqa: E402

from mibel_risk.pfe import pfe_ppa, pfe_swing, pfe_tolling  # noqa: E402

CURATED = DERIV / "data" / "curated"
ALPHA = 0.95
pd.set_option("display.float_format", lambda v: f"{v:,.2f}")
print("imports OK")'''
)

md(
    """## 1 · Simulate price paths from the Schwartz–Smith fit (Pieza 2)

`S_t = exp(chi_t + xi_t)` from the last filtered state, over a one-year daily
grid (the common contract life used for all three exposures)."""
)

co(
    '''with open(CURATED / "forward_fit_production.pkl", "rb") as fh:
    ss = pickle.load(fh)
p = ss.params
chi0, xi0 = float(ss.state_chi.iloc[-1]), float(ss.state_xi.iloc[-1])
start = pd.Timestamp(ss.trade_dates[-1])
print(f"SS fit @ {start.date()}: kappa={p.kappa:.3f} rho={p.rho:.3f} "
      f"sigma_chi={p.sigma_chi:.3f} sigma_xi={p.sigma_xi:.3f}")

H = 365
N_PATHS = 4000
chi, xi = forward.simulate(p, chi0, xi0, start, H, N_PATHS, seed=2024)
S = np.exp(chi + xi)                       # (N_PATHS, H) daily spot/power, EUR/MWh
time_grid = np.linspace(0.0, 1.0, H)       # years from today
print(f"paths: {S.shape}  mean={S.mean():,.1f}  p5={np.quantile(S, 0.05):,.1f}  "
      f"p95={np.quantile(S, 0.95):,.1f} EUR/MWh")'''
)

md("## 2 · Define the three contracts (real `mibel_derivatives` objects)")

co(
    '''swing = SwingTerms(strike=65.0, volume_per_right=20.0, n_rights=300)
tolling = TollingAgreement()  # Castejón I CCGT defaults (Pmax 386 MW)
ppa = PPA(
    capacity_mw=50.0, plant_factor=0.20, fixed_pct=0.5, spot_pct=0.5,
    strike=55.0, duration_years=1,
)

# Flat fuel/carbon curves for the tolling spark spread (representative MIBEL levels).
gas_curve = np.full(H, 32.0)   # EUR/MWh
eua_curve = np.full(H, 75.0)   # EUR/t

print(f"swing  : K={swing.strike}  vol/right={swing.volume_per_right}  rights={swing.n_rights}")
print(f"tolling: Pmax={tolling.asset.pmax_mw} MW  HR={tolling.asset.heat_rate_full_gj_per_mwh} GJ/MWh  "
      f"co2={tolling.asset.co2_intensity_t_per_mwh_th} t/MWh_th")
print(f"ppa    : cap={ppa.capacity_mw} MW  CF={ppa.plant_factor}  spot_pct={ppa.spot_pct}  K={ppa.strike}")'''
)

md("## 3 · PFE curves over the contract life")

co(
    '''res_swing = pfe_swing(swing, S, time_grid, alpha=ALPHA)
res_tolling = pfe_tolling(tolling, S, gas_curve, eua_curve, time_grid, alpha=ALPHA,
                          hours_per_step=24.0)
res_ppa = pfe_ppa(ppa, S, time_grid, alpha=ALPHA)

profiles = {"swing": res_swing, "tolling": res_tolling, "ppa": res_ppa}
for name, r in profiles.items():
    print(f"{name:8s}: peak PFE(95%) = EUR {r.peak_pfe:14,.0f}  at t={r.peak_time:.2f}y  "
          f"| EE peak = EUR {r.expected_exposure.max():14,.0f}")'''
)

co(
    '''fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharex=True)
for ax, (name, r) in zip(axes, profiles.items()):
    ax.plot(r.time_grid, r.pfe, color="tab:blue", lw=1.6, label="PFE 95%")
    ax.plot(r.time_grid, r.expected_exposure, color="tab:orange", lw=1.2, label="EE (mean)")
    ax.fill_between(r.time_grid, 0, r.pfe, color="tab:blue", alpha=0.08)
    ax.axvline(r.peak_time, color="0.6", ls=":", lw=0.8)
    ax.set_title(f"{name} — peak EUR {r.peak_pfe:,.0f}")
    ax.set_xlabel("years"); ax.set_ylabel("exposure (EUR)")
    ax.legend(fontsize=8)
fig.suptitle("Potential Future Exposure profiles (alpha = 95%)", y=1.03)
plt.tight_layout(); plt.show()'''
)

md(
    """## 4 · PFE vs notional

The peak PFE is the headline counterparty limit a credit desk would set. We
express it against each contract's gross notional value (total volume × current
spot) to see how much of the notional is actually *at risk*."""
)

co(
    '''S0 = float(S[:, 0].mean())

swing_notional = swing.volume_per_right * swing.n_rights
tolling_notional = tolling.effective_capacity_mw * 24.0 * H
ppa_notional = ppa.capacity_mw * ppa.plant_factor * 8760.0 * ppa.duration_years

rows = [
    ("swing", swing_notional, res_swing.peak_pfe),
    ("tolling", tolling_notional, res_tolling.peak_pfe),
    ("ppa", ppa_notional, res_ppa.peak_pfe),
]
table = pd.DataFrame(
    [
        {
            "notional_MWh": vol,
            "notional_value_EUR": vol * S0,
            "peak_PFE_EUR": pfe,
            "PFE_pct_of_notional_value": 100.0 * pfe / (vol * S0),
        }
        for _, vol, pfe in rows
    ],
    index=[r[0] for r in rows],
)
print(f"reference spot S0 = {S0:,.2f} EUR/MWh\\n")
print(table.to_string())'''
)

md(
    """## 5 · Summary

* All three PFE profiles share the textbook **hump**: zero at inception, peaking
  mid-life where price diffusion is largest relative to the remaining volume, and
  decaying to zero at maturity.
* The **tolling** exposure is the largest in absolute EUR (it tolls the full
  Castejón capacity over a year), while the **PPA** — a two-sided swap on a small
  solar plant — is the smallest.
* Expressed as a fraction of notional value, the peak PFE shows how a 95%
  adverse market move translates into a counterparty replacement cost; these
  peak figures feed directly into the CVA integral in notebook 03.

Key figures are collected in `reports/diagnostics/pfe.md`.
"""
)

nb = new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.metadata["language_info"] = {"name": "python", "version": "3.11"}
out = Path("notebooks/02_pfe_derivados.ipynb")
out.parent.mkdir(exist_ok=True)
nbf.write(nb, str(out))
print(f"wrote {out} with {len(cells)} cells")
