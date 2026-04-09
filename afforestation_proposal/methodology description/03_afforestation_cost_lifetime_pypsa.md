# Afforestation Cost & Lifetime for PyPSA-Eur: How to Annualise

## 1. The Problem You're Facing

In PyPSA-Eur, every technology has:
- **investment** (overnight capital cost, €/unit)
- **lifetime** (years) — used to compute the annuity
- **FOM** (fixed O&M, %/year of investment)

The annuity factor spreads the investment over the lifetime:

```
annuity = r / (1 - (1+r)^(-n))
annualised_cost = investment × annuity + investment × FOM
```

For a wind turbine this is intuitive: you spend X €/kW upfront, it lasts 25 years, you pay Y% FOM.

For afforestation, it's conceptually different: you spend ~€3,000–8,000/ha **once** to plant, then €50–200/ha/yr to maintain, and the "asset" (the forest) doesn't depreciate — it *grows*. The question is: **what lifetime do you set?**

---

## 2. The Correct Way to Think About It

### The key insight: match the lifetime to the sequestration accounting period

In an energy system model, afforestation is modelled as a **store** or **generator** of negative CO₂ emissions. The "capacity" is the annual sequestration rate (tCO₂/ha/yr), and the "investment" is the cost to create one hectare of forest.

The **lifetime should equal the rotation length** (or the planning horizon, whichever is shorter) because:

1. **The rotation period is when you get the sequestration benefit.** After one rotation (60–100 years for European forests), the stand is either harvested (and CO₂ is partly released) or reaches maturity (and net sequestration drops to near zero). Either way, the *annual removal service* effectively ends or resets.

2. **This makes the annualised cost comparable to other CDR options** like DAC, which also have investment + lifetime + FOM.

3. **It's consistent with what Fernandes et al. (2026) and other IAM literature do** — they treat the establishment cost as a one-time investment amortised over the period of active carbon uptake.

### Recommended parameter values

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Investment** | €3,000–8,000 /ha | Establishment cost. Your €7,790/ha is within range for European broadleaf-conifer mix. Includes site prep, seedlings, planting labour, fencing. |
| **Lifetime** | **30 years** (central), sensitivity 20–50 | See reasoning below |
| **FOM** | 2–5% of investment/year | Covers annual maintenance, fire protection, fence repair. Your 5% ≈ €390/yr at €7,790 investment, which is reasonable for European conditions. |

---

## 3. Why 30 Years (Not the Full Rotation)

This is the crux of the issue. A full rotation is 60–100 years, but **30 years is the better choice** for an energy system model, for three reasons:

### Reason 1: The high-sequestration phase is ~30 years
Looking at the Pilli et al. yield tables, the **current annual increment (CAI)** peaks at age 20–40 for most European species, and the **mean annual increment (MAI)** peaks at 40–60 years. After age 30, the per-year CO₂ removal rate has already passed its maximum and is declining. If you set lifetime = 80 years, you're averaging over decades of low/zero net uptake, which makes the annualised cost artificially low but misrepresents the actual service delivered.

### Reason 2: Planning horizon compatibility
PyPSA-Eur typically models out to 2050 (from ~2025). That's 25 years. Even in myopic pathway optimisation stepping to 2060 or 2070, you're looking at 30–45 year horizons. Setting lifetime = 80 years means the model "sees" costs and benefits far beyond its planning horizon, which creates unrealistic optimism about afforestation.

### Reason 3: Consistency with EU policy frameworks
The EU LULUCF regulation and the new Carbon Removal Certification Framework (CRCF) both use monitoring periods of **5–30 years for afforestation credits**. The CRCF specifically generates "temporary" carbon removal units that expire and need renewal. This suggests 30 years is the policy-relevant accounting period.

### What this looks like numerically

With your numbers (€7,790/ha, FOM 5%, discount rate 7%):

| Lifetime | Annuity factor | Annualised investment | + FOM (5%) | Total annual cost/ha |
|----------|---------------|----------------------|------------|---------------------|
| 20 yr | 0.0944 | €735 | €390 | **€1,125/ha/yr** |
| 30 yr | 0.0806 | €628 | €390 | **€1,018/ha/yr** |
| 50 yr | 0.0725 | €565 | €390 | **€955/ha/yr** |
| 80 yr | 0.0706 | €550 | €390 | **€940/ha/yr** |

Notice: beyond 30 years, the annualised cost barely changes (because discounting makes distant years nearly worthless at 7%). **The lifetime choice matters much less than you might think** at typical discount rates.

### Converting to €/tCO₂

If your sequestration rate is, say, 8 tCO₂/ha/yr (mid-range for temperate Europe):

| Lifetime | Annual cost/ha | Sequestration | **€/tCO₂** |
|----------|---------------|---------------|------------|
| 20 yr | €1,125 | 8 tCO₂ | **€141** |
| 30 yr | €1,018 | 8 tCO₂ | **€127** |
| 50 yr | €955 | 8 tCO₂ | **€119** |

These are in the right ballpark: the OECD reports afforestation costs of €16–54/tCO₂ at lower discount rates, and up to €100–200/tCO₂ at higher ones. Your costs are on the high end because of the 7% discount rate; at 3–4% they'd be lower.

---

## 4. Alternative Formulation: Marginal Cost Instead of Annuity

An arguably cleaner approach for PyPSA-Eur is to **not use the annuity at all** for the establishment cost, and instead fold everything into a **marginal cost per tCO₂**:

```python
# One-time establishment cost, annualised over active uptake period
establishment_annualised = 7790 * annuity(0.07, 30)  # €628/ha/yr
maintenance_annual = 7790 * 0.05                       # €390/ha/yr
total_annual_per_ha = 628 + 390                        # €1018/ha/yr

sequestration_rate = 8  # tCO2/ha/yr (region-specific from Pilli et al.)
marginal_cost = total_annual_per_ha / sequestration_rate  # €127/tCO2
```

Then model afforestation as a **store** with:
- `e_nom_max` = available land (ha) × sequestration rate (tCO₂/ha/yr)  →  max annual CO₂ removal
- `capital_cost` = 0 (or minimal)
- `marginal_cost` = €127/tCO₂ (region-specific)

This avoids the lifetime question entirely by pre-computing the annualised cost per tonne. The Fernandes et al. paper likely does something similar.

---

## 5. Cost Data Sources for European Afforestation

### Establishment costs (one-time, €/ha)

| Source | Conifers | Broadleaves | Notes |
|--------|----------|-------------|-------|
| Climate-ADAPT / EFI (2000) | — | — | Aid: €2,400–4,800/ha depending on species |
| Irish scheme (2023–2027) | €2,740–4,600 | €4,600–6,220 | Grant covers establishment + 4yr maintenance |
| ForestNavigator D4.4 (2024) | Spatially explicit EU maps | +40% vs conifers | Best available EU-wide spatial data |
| Finnish study (Salminen 2022) | €2,000+ threshold for profitability | | Afforestation on cropland |
| Italy (Reg. 2080/92 plantations) | — | — | EU subsidy-based plantations, 15–20yr minimum |
| **Your assumption** | **€7,790** | | On the high side but defensible for broadleaf-mix in W. Europe |

### Annual maintenance (€/ha/yr)

| Source | Range | Notes |
|--------|-------|-------|
| Climate-ADAPT / EFI | €100–300/ha/yr | Decreasing from year 1 to year 3 |
| Irish scheme premium | €520–1,170/ha/yr | Includes income compensation |
| Bodin et al. (your PDF) | $167–2,421/ha/yr | Tropics/subtropics, not directly applicable |
| **Your 5% FOM** | **€390/ha/yr** | Reasonable for European conditions |

### Opportunity cost (often the largest hidden cost)
Income foregone from agriculture: €180–725/ha/yr in the EU (EC Regulation 1054/94), highly region-dependent. This is often **not included** in energy system models but could be added as an additional marginal cost if you want to represent land competition.

---

## 6. Recommended Implementation Summary

```yaml
# In your technology-data or config:
afforestation:
  investment: 7790      # €/ha (establishment cost)
  lifetime: 30          # years (active high-sequestration period)
  FOM: 5                # %/year of investment
  # OR equivalently, pre-compute:
  # marginal_cost: 127  # €/tCO2 (at 8 tCO2/ha/yr, r=0.07)
  
  # Sequestration rate: region-specific from Pilli et al. (2024)
  # Range: 4–15 tCO2/ha/yr depending on NUTS-2 region
  
  # Potential: from Corine land cover (already implemented)
```

### Sensitivity scenarios

| Scenario | Investment | Lifetime | FOM | Rate | €/tCO₂ |
|----------|-----------|----------|-----|------|--------|
| Low cost (assisted natural regen.) | €1,500/ha | 30 yr | 3% | 5 tCO₂ | €34 |
| Central (your current) | €7,790/ha | 30 yr | 5% | 8 tCO₂ | €127 |
| High cost (broadleaf, high-income land) | €10,000/ha | 30 yr | 5% | 6 tCO₂ | €252 |

---

## 7. Key References

- **Bodin et al. (2021)** — TEER framework. Useful for structuring cost categories but cost data mostly tropical.
- **Pilli et al. (2024)** — Zenodo dataset for spatially-resolved sequestration rates AND BCEF.
- **ForestNavigator D4.4** — Best available EU-wide spatially explicit afforestation cost model (conifers + broadleaves). Available at: https://www.forestnavigator.eu/resources/
- **Kryszk et al. (2024)** — Referenced in ForestNavigator, on European afforestation cost competitiveness.
- **Salminen et al. (2022)** — Finnish study on cropland afforestation profitability vs agriculture, useful for opportunity costs.
- **Fernandes et al. (2026)** — arXiv:2603.25663, already implements afforestation in PyPSA-Eur. **Check their code and assumptions first.**
- **EC (2021)** — 3 Billion Trees Roadmap SWD(2021) 651, contains EU establishment cost ranges.
- **IPCC (2019)** — 2019 Refinement Guidelines, Vol. 4 AFOLU, for default sequestration factors.
