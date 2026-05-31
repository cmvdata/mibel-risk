"""Builder for notebooks/03_cva.ipynb (run once, then strip outputs)."""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

cells: list = []
md = lambda s: cells.append(new_markdown_cell(s))  # noqa: E731
co = lambda s: cells.append(new_code_cell(s))  # noqa: E731

md(
    """# 03 · CVA — tolling agreement vs Iberdrola

**Credit Valuation Adjustment** of the CCGT **tolling** agreement (Pieza 4,
Castejón I) against counterparty **Iberdrola**.

CVA is the market price of the counterparty's credit risk — the discounted
expected loss from Iberdrola defaulting while it owes a positive mark-to-market
on the toll:

```
CVA = (1 − R) · Σ_i  EE(t_i) · DF(t_i) · [S(t_{i−1}) − S(t_i)]
```

We take the tolling **expected-exposure** curve from the PFE engine
(`mibel_risk.pfe`), a constant hazard rate for Iberdrola, recovery `R = 40%`, and
a flat discount curve, then stress CVA against hazard and recovery.

> *Iberdrola* is a strong investment-grade utility (≈ BBB+/A−); a 5y CDS of
> ~70 bps implies a hazard `λ ≈ CDS/(1−R) ≈ 0.012`. We use that as the base case.
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
from mibel_derivatives.products.tolling import TollingAgreement  # noqa: E402

from mibel_risk.cva import cva, discount_from_rate, survival_from_hazard  # noqa: E402
from mibel_risk.pfe import pfe_tolling  # noqa: E402

CURATED = DERIV / "data" / "curated"
print("imports OK")'''
)

md(
    """## 1 · Tolling expected-exposure curve (from Pieza 2 + PFE)

Simulate daily power-price paths from the Schwartz–Smith fit, then take the
tolling **EE(t)** profile from `pfe_tolling`."""
)

co(
    '''with open(CURATED / "forward_fit_production.pkl", "rb") as fh:
    ss = pickle.load(fh)
p = ss.params
chi0, xi0 = float(ss.state_chi.iloc[-1]), float(ss.state_xi.iloc[-1])
start = pd.Timestamp(ss.trade_dates[-1])

H = 365
N_PATHS = 4000
chi, xi = forward.simulate(p, chi0, xi0, start, H, N_PATHS, seed=2024)
power = np.exp(chi + xi)
time_grid = np.linspace(0.0, 1.0, H)  # years

tolling = TollingAgreement()
gas_curve = np.full(H, 32.0)
eua_curve = np.full(H, 75.0)

pfe_res = pfe_tolling(tolling, power, gas_curve, eua_curve, time_grid, alpha=0.95,
                      hours_per_step=24.0)
EE = pfe_res.expected_exposure
print(f"tolling EE: peak EUR {EE.max():,.0f} at t={time_grid[int(np.argmax(EE))]:.2f}y "
      f"| PFE95 peak EUR {pfe_res.peak_pfe:,.0f}")'''
)

md("## 2 · Base-case CVA")

co(
    '''LAMBDA = 0.012     # Iberdrola hazard (~70 bps CDS at R=0.4)
RECOVERY = 0.40
DISCOUNT_RATE = 0.03

df = discount_from_rate(time_grid, DISCOUNT_RATE)
res = cva(time_grid, EE, default_curve=LAMBDA, recovery=RECOVERY, discount_curve=df)

surv = survival_from_hazard(time_grid, LAMBDA)
print(f"hazard lambda      = {LAMBDA:.4f} /yr")
print(f"1y default prob    = {1 - surv[-1]:.3%}")
print(f"recovery R         = {RECOVERY:.0%}")
print(f"discount rate      = {DISCOUNT_RATE:.1%}")
print(f"-> CVA             = EUR {res.cva:,.0f}")
print(f"   (= {1e4 * res.cva / EE.max():.1f} bps of peak EE)")'''
)

co(
    '''fig, ax1 = plt.subplots(figsize=(10, 3.8))
ax1.plot(time_grid, EE, color="tab:blue", lw=1.6, label="EE(t)  (EUR)")
ax1.fill_between(time_grid, 0, EE, color="tab:blue", alpha=0.08)
ax1.set_xlabel("years"); ax1.set_ylabel("expected exposure (EUR)", color="tab:blue")
ax1.tick_params(axis="y", labelcolor="tab:blue")
ax2 = ax1.twinx()
ax2.plot(time_grid, survival_from_hazard(time_grid, LAMBDA), color="tab:red", lw=1.2,
         label="survival S(t)")
ax2.set_ylabel("survival probability", color="tab:red")
ax2.tick_params(axis="y", labelcolor="tab:red"); ax2.set_ylim(0.97, 1.001)
ax1.set_title(f"Tolling EE vs Iberdrola survival — CVA = EUR {res.cva:,.0f}")
fig.tight_layout(); plt.show()'''
)

md("## 3 · Sensitivity to hazard rate and recovery")

co(
    '''hazards = np.linspace(0.0, 0.06, 31)
recoveries = np.linspace(0.0, 0.8, 31)

cva_vs_hazard = np.array([cva(time_grid, EE, h, RECOVERY, df).cva for h in hazards])
cva_vs_recovery = np.array([cva(time_grid, EE, LAMBDA, r, df).cva for r in recoveries])

fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
axes[0].plot(1e4 * hazards, cva_vs_hazard / 1e6, color="tab:blue")
axes[0].axvline(1e4 * LAMBDA, color="0.6", ls=":")
axes[0].set_xlabel("hazard rate (bps/yr)"); axes[0].set_ylabel("CVA (EUR m)")
axes[0].set_title("CVA vs hazard  (R = 40%)")
axes[1].plot(100 * recoveries, cva_vs_recovery / 1e6, color="tab:green")
axes[1].axvline(100 * RECOVERY, color="0.6", ls=":")
axes[1].set_xlabel("recovery (%)"); axes[1].set_ylabel("CVA (EUR m)")
axes[1].set_title("CVA vs recovery  (lambda = 120 bps)  — linear in (1-R)")
plt.tight_layout(); plt.show()

print(f"CVA at lambda=120bps, R=40%   = EUR {res.cva:,.0f}")
print(f"CVA at lambda=300bps, R=40%   = EUR {cva(time_grid, EE, 0.03, 0.40, df).cva:,.0f}")
print(f"CVA at lambda=120bps, R=20%   = EUR {cva(time_grid, EE, LAMBDA, 0.20, df).cva:,.0f}")'''
)

co(
    '''# 2-D CVA surface
grid_cva = np.array([[cva(time_grid, EE, h, r, df).cva for r in recoveries] for h in hazards])
fig, ax = plt.subplots(figsize=(7, 4.2))
im = ax.pcolormesh(100 * recoveries, 1e4 * hazards, grid_cva / 1e6, shading="auto", cmap="viridis")
cs = ax.contour(100 * recoveries, 1e4 * hazards, grid_cva / 1e6, colors="white", linewidths=0.5)
ax.clabel(cs, inline=True, fontsize=7, fmt="%.1f")
ax.scatter([100 * RECOVERY], [1e4 * LAMBDA], color="red", s=40, zorder=3, label="base case")
ax.set_xlabel("recovery (%)"); ax.set_ylabel("hazard rate (bps/yr)")
ax.set_title("CVA surface (EUR m)"); ax.legend(loc="upper right", fontsize=8)
fig.colorbar(im, ax=ax, label="CVA (EUR m)")
plt.tight_layout(); plt.show()'''
)

md(
    """## 4 · Summary

* The tolling agreement against Iberdrola carries a base-case CVA driven by the
  large mid-life expected exposure of a full-capacity CCGT toll.
* CVA is **monotonic increasing** in the hazard rate (more default risk) and
  **linear in the loss-given-default** `(1 − R)` — both visible in the
  sensitivity panels and the surface contours.
* Because Iberdrola is strongly investment-grade, the base-case default
  probability over the year is small, but the exposure is large, so the credit
  charge is non-trivial — exactly the trade-off a CVA desk prices.

Key figures are collected in `reports/diagnostics/cva.md`.
"""
)

nb = new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.metadata["language_info"] = {"name": "python", "version": "3.11"}
out = Path("notebooks/03_cva.ipynb")
out.parent.mkdir(exist_ok=True)
nbf.write(nb, str(out))
print(f"wrote {out} with {len(cells)} cells")
