# Potential Future Exposure — structured derivatives

**Module:** `mibel_risk.pfe` · **Notebook:** `notebooks/02_pfe_derivados.ipynb`
**Confidence level:** 95% · **Exposure:** undiscounted positive mark-to-market (EUR).

PFE answers a counterparty-credit question: *if the counterparty defaults at a
future date, what is the live contract's positive replacement cost?* For each
simulated price path the contract's mark-to-market relative to inception is
computed along a time grid; the positive part is the exposure, and its
`alpha`-quantile across paths is the PFE curve `PFE(t)`.

---

## 1 · What was built

| File | Role |
|------|------|
| `src/mibel_risk/pfe/exposure.py` | `pfe_swing`, `pfe_tolling`, `pfe_ppa` (one PFE curve each over the contract life) + the shared engine `pfe_from_value_paths` and `PFEResult`. |
| `tests/test_pfe.py` | 16 tests: convergence in `n_paths`, `PFE(0) ≈ 0`, monotonicity in `alpha`, `PFE ≥ 0`, interior hump, notional linearity, validation. |
| `notebooks/02_pfe_derivados.ipynb` | Loads the Pieza 2 fit, computes PFE for the three real product objects, plots the profiles, compares to notional. |
| `scripts/_build_pfe_notebook.py` | Reproducible notebook builder. |

### Design — standalone, duck-typed
The estimators **never import `mibel_derivatives`**. They consume simulated price
paths (an array, or a `simulate(n_paths, rng)` callable) plus a duck-typed
contract object whose attribute names match the real product dataclasses
(`SwingTerms.strike/volume_per_right/n_rights`, `TollingAgreement.asset.*`,
`PPA.strike/spot_pct/...`). A real product passes straight through; tests use
`SimpleNamespace` stand-ins. `src/` stays CI-safe.

### Exposure model
```
value_t      = payoff_level(price_t) · notional · remaining_fraction(t)
exposure_t   = max(value_t − value_0, 0)          # value_0 = deterministic inception value
PFE(t)       = quantile_alpha( exposure_t )        EE(t) = mean( exposure_t )
```
Because `value_0` is common to all paths, **`PFE(0) = 0` exactly**; because
`remaining_fraction` runs off linearly to zero, **`PFE(T) = 0`**. The result is
the textbook hump. Per-product payoff legs:

| Contract | `payoff_level` | Shape |
|----------|----------------|-------|
| swing call | `max(price − K, 0)` | one-sided (strip of calls) |
| tolling | `max(power − HR·(gas + co2·eua), 0)` | one-sided (spark-spread options) |
| PPA | `spot_pct · (price − K)` | two-sided (spot-vs-fixed swap) |

`HR = heat_rate_gj_per_mwh / 3.6` (MWh-thermal per MWh-electric).

---

## 2 · Results (Pieza 2 simulation, 4 000 paths × 365 days)

Schwartz–Smith fit @ 2024-12-31 (`kappa = 0.384`, `sigma_chi = 1.021`,
`sigma_xi = 0.231`, `rho = -0.806`); daily spot `S_t = exp(chi_t + xi_t)`,
mean 88.9, p5 33.7, p95 190.9 EUR/MWh; reference spot `S0 = 66.62 EUR/MWh`.

| Contract | Peak PFE (95%) | Peak date | Peak EE | Notional | PFE / notional value |
|----------|---------------:|:---------:|--------:|---------:|---------------------:|
| swing | **EUR 362 089** | 0.38 y | 88 894 | 6 000 MWh | **90.6%** |
| tolling | **EUR 167 891 328** | 0.43 y | 33 921 209 | 3 382 236 MWh | **74.5%** |
| ppa | **EUR 2 478 397** | 0.38 y | 561 166 | 87 600 MWh | **42.5%** |

* All three peak in the **interior** (0.38–0.43 y), confirming the
  diffusion-vs-amortisation hump.
* **Tolling** dominates in absolute EUR — it tolls the full Castejón I capacity
  (386 MW) over a year. **PPA**, a two-sided swap on a 50 MW solar plant, is the
  smallest and the least exposed as a share of notional (42.5%) because its swap
  payoff nets rather than accumulating one-sided optionality.
* The peaks are a large fraction of notional value, which reflects the
  **substantial short-term volatility** of the calibrated SS fit
  (`sigma_chi ≈ 1.0 /√yr`): a 95% adverse move over half a year is big.

---

## 3 · Decisions taken autonomously

1. **Standalone, duck-typed estimators** (as for Deliverable 1) — `src/` and
   `tests/` never import `mibel_derivatives`, so CI stays green; the notebook
   supplies real product objects and simulated paths.
2. **MtM-relative-to-inception convention** (`value_t − value_0`) chosen so
   `PFE(0) = 0` holds *exactly and for any moneyness*, rather than relying on the
   contract being struck at-the-money.
3. **Linear volume run-off** as the amortisation schedule — simple, monotone, and
   guarantees `PFE(T) = 0`; documented as a modelling choice.
4. **Undiscounted exposure** — discounting is deferred to the CVA integral
   (Deliverable 3), keeping the PFE curve a pure replacement-cost profile.
5. **Forward-level value proxy** (`payoff_level(price_t)`) rather than a nested
   re-pricing of remaining cashflows — avoids nested Monte Carlo while preserving
   the correct exposure shape; sufficient for a PFE / counterparty-limit view.
6. **Flat fuel/carbon curves** (gas 32 EUR/MWh, EUA 75 EUR/t) for the tolling
   spark spread — representative MIBEL levels; the power leg is the stochastic
   driver of exposure.
7. **`EE ≤ PFE` is *not* asserted** as a universal property: where early-life
   exposure has >95% mass at zero, the mean (EE) can exceed the 95% quantile
   (PFE = 0). The test instead checks the relationship at the peak date, where it
   holds.
