# Credit Valuation Adjustment — tolling vs Iberdrola

**Module:** `mibel_risk.cva` · **Notebook:** `notebooks/03_cva.ipynb`

CVA is the market price of counterparty credit risk: the discounted expected
loss from the counterparty defaulting while it owes a positive mark-to-market.

```
CVA = (1 − R) · Σ_i  EE(t_i) · DF(t_i) · [S(t_{i−1}) − S(t_i)]
```

with expected positive exposure `EE`, recovery `R`, discount factors `DF`, and
survival `S(t) = exp(−∫ λ)` from a hazard-rate default curve.

---

## 1 · What was built

| File | Role |
|------|------|
| `src/mibel_risk/cva/cva.py` | `cva(time_grid, positive_exposure_curve, default_curve, recovery, discount_curve)` + helpers `survival_from_hazard`, `discount_from_rate`, and the `DefaultCurve` / `CVAResult` dataclasses. |
| `tests/test_cva.py` | 18 tests: `CVA ≥ 0`, linearity in `(1 − R)`, monotonicity in hazard, `hazard = 0 ⇒ CVA = 0`, discounting, piecewise hazard, helper closed-forms, validation. |
| `notebooks/03_cva.ipynb` | CVA of the Castejón tolling toll vs Iberdrola, with hazard/recovery sensitivities and a 2-D CVA surface. |
| `scripts/_build_cva_notebook.py` | Reproducible notebook builder. |

### Design
* **Standalone** (numpy only); consumes the exposure as a plain `EE(t)` array, so
  it composes directly with `mibel_risk.pfe` (`PFEResult.expected_exposure`).
* **Default curve** accepts a constant hazard (scalar), a piecewise-constant
  per-interval hazard (array), or a `DefaultCurve` object.
* Exposure is taken at the **interval end** (a standard, slightly conservative
  convention); CVA is non-negative by construction.

---

## 2 · Base-case result

Tolling EE from the Pieza 2 simulation (4 000 paths × 365 days): **peak EE
EUR 33 921 209 at t = 0.55 y**, PFE(95%) peak EUR 167 891 328.

| Input | Value |
|-------|-------|
| Counterparty | Iberdrola (≈ BBB+/A−) |
| Hazard `λ` | 0.012 /yr (~70 bps CDS at R = 40%) |
| 1-year default probability | **1.193%** |
| Recovery `R` | 40% |
| Discount rate | 3.0% (flat) |
| **CVA** | **EUR 153 738** |

Sensitivities (same EE and discount curve):

| Scenario | CVA |
|----------|-----|
| `λ = 120 bps`, `R = 40%` (base) | EUR 153 738 |
| `λ = 300 bps`, `R = 40%` | EUR 380 834 |
| `λ = 120 bps`, `R = 20%` | EUR 204 984 |

The `R = 20%` figure is exactly `(0.80 / 0.60) × 153 738` — confirming the
**linearity in `(1 − R)`** that the engine guarantees. CVA rises monotonically
with the hazard rate. The 2-D surface in the notebook shows the joint
dependence: contours of constant CVA in the `(recovery, hazard)` plane.

---

## 3 · Decisions taken autonomously

1. **Standalone numpy engine** — no `mibel_derivatives` import in `src/`; the
   notebook supplies the EE curve from the PFE module and the real tolling
   product.
2. **Iberdrola hazard calibrated from a CDS proxy** — `λ ≈ CDS / (1 − R)` with a
   ~70 bps 5-year CDS for a strong investment-grade utility, giving `λ ≈ 0.012`.
   This is a representative assumption, not a market quote.
3. **`time_grid` added as an explicit first argument** to the spec's
   `cva(positive_exposure_curve, default_curve, recovery, discount_curve)` —
   the brief calls for a *discrete implementation over a time grid*, which the
   grid makes explicit and unambiguous.
4. **Exposure-at-interval-end** convention (rather than midpoint averaging) —
   simple and marginally conservative; documented in the module.
5. **Unilateral CVA only** — no DVA/bilateral or wrong-way-risk terms; matches
   the brief's "CVA simple" scope. Flat discounting (3%) and flat fuel/carbon
   curves keep the credit dimension in focus.
