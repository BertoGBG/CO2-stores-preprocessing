# Afforestation in PyPSA-Eur: Policy-Aligned Methodology & Case Study

## 1. Why Align with EU Policy Frameworks?

Aligning your PyPSA-Eur afforestation module with EU policy frameworks serves three purposes:

1. **Publishability** — Reviewers will ask how your CDR accounting relates to the LULUCF Regulation and the CRCF. If you can show direct correspondence, your methodology is immediately defensible.

2. **Policy relevance** — PyPSA-Eur results are used to inform EU energy/climate policy. If your afforestation module mirrors the accounting rules member states actually use, your model outputs can be directly compared to national LULUCF targets and budgets.

3. **Consistency with other models** — IAMs like GLOBIOM, EU-CBM-HAT, and EFISCEN already use LULUCF-compliant accounting. Alignment makes cross-model comparison possible.

---

## 2. The Three Relevant EU Frameworks

### 2.1 LULUCF Regulation (EU 2018/841, amended by EU 2023/839)

**What it does for afforestation:**
- Emissions and removals from **afforested land** are accounted in full (gross-net accounting) — meaning every tonne of CO₂ removed by new forests counts directly.
- A **20-year transition period** applies: land remains in the "afforested land" category for 20 years after conversion, after which it moves to "managed forest land" (where different accounting rules apply, with comparison against a Forest Reference Level).
- An exception allows a **30-year transition period** if justified by IPCC Guidelines.
- The 2026–2030 period introduces binding national targets: EU-wide 310 MtCO₂eq net removals by 2030.
- From 2026, compliance is based on **reported GHG inventory data** (not modelled benchmarks), using advanced monitoring including remote sensing.

**Implication for your model:** Under LULUCF, afforestation removals in the first 20–30 years are counted at face value. This directly supports using a **20–30 year accounting window** as the "lifetime" in PyPSA-Eur.

### 2.2 Carbon Removal Certification Framework — CRCF (EU 2024/3012)

**What it does for afforestation:**
- Establishes a voluntary EU-wide certification for carbon removals, with afforestation falling under **"carbon farming"** — generating temporary certified units.
- Monitoring periods for carbon farming are typically **5–30 years**, with units expiring at the end of the monitoring period unless renewed.
- Operators must demonstrate **additionality**, **robust quantification**, **permanence** (with reversal liability during the monitoring period), and **environmental sustainability** (QU.A.L.ITY criteria).
- Afforestation methodology delegated act expected by **summer 2026** (currently in public consultation via DG CLIMA).
- Quantification must follow standardised baseline comparisons.

**Implication for your model:** The CRCF gives you a policy-grounded justification for using **temporary, time-limited removal credits** with a 20–30 year horizon. It also means you should account for **reversal risk** (see below).

### 2.3 EU Biodiversity Strategy / 3 Billion Trees Pledge

- Targets planting at least 3 billion additional trees by 2030.
- SWD(2021) 651 provides establishment cost ranges and emphasises that maintenance for the first 15–20 years is critical.
- Links to CAP funding for afforestation on agricultural land.

**Implication for your model:** This gives political legitimacy to the afforestation potentials you derive from Corine land cover.

---

## 3. Policy-Aligned Methodology

### 3.1 Sequestration Rate — Consistent with LULUCF Gross-Net Accounting

Under LULUCF, afforested land removals are reported as the **actual net carbon stock change** in biomass pools. This is exactly what you compute from the Pilli et al. yield tables:

```
CO₂_removal(t) = Δ_biomass_stock(t) × carbon_fraction × 3.667 × (1 + root:shoot)
```

For an energy system model, you need an **annualised constant** rather than a time-varying curve. The rotation-averaged MAI (Proposal A from the earlier document) is the right simplification, but you should document it as an approximation of the LULUCF gross-net accounting.

**Key alignment point:** Under LULUCF, both aboveground and belowground biomass, dead organic matter, and soil organic carbon are reported. The Pilli et al. BCEF data covers aboveground + bark + branches + foliage. You should add:
- Root biomass: IPCC root-to-shoot ratio (0.20–0.30 for temperate)
- Optionally, soil carbon: ~0.5–1.5 tCO₂/ha/yr for temperate afforestation (more uncertain)
- Dead organic matter: small, often neglected in energy models

This makes your rate directly comparable to what member states report in their national GHG inventories.

### 3.2 Accounting Period — The 20/30 Year LULUCF Transition

The LULUCF Regulation's 20-year (default) or 30-year (justified) transition period for afforested land provides the natural **lifetime parameter** for PyPSA-Eur:

| PyPSA-Eur parameter | LULUCF-aligned value | Justification |
|---------------------|---------------------|---------------|
| `lifetime` | **30 years** | Art. 5(3) of Reg. 2018/841 allows 30-year transition if justified by IPCC GL. Most EU member states use this. |
| Alternative | **20 years** | Default LULUCF transition period. More conservative. |
| Sensitivity | **50 years** | For scenarios exploring long-term forest maintenance without harvest |

After the transition period, the land moves to "managed forest land" where removals are compared against a Forest Reference Level — the net additional benefit may be small or zero. This further supports **not** using an 80-year lifetime.

### 3.3 Costs — CRCF-Compatible Structure

The CRCF requires that certified removals meet **additionality** — the removal would not have happened without the certification incentive. In an energy system model, this translates to: the cost of afforestation must exceed the cost of doing nothing (i.e., there must be a positive cost). Your €7,790/ha + 5% FOM already satisfies this.

Structure your costs to match CRCF/TEER expenditure categories:

| Cost component | Your value | CRCF/TEER category |
|----------------|-----------|---------------------|
| Site preparation, seedlings, planting | €5,000–6,000/ha | Establishment (consumables + paid labour) |
| Fencing, protection | €1,000–2,000/ha | Establishment (consumables) |
| Annual maintenance (yr 1–5) | €200–400/ha/yr | Monitoring phase (paid labour + consumables) |
| Annual maintenance (yr 6–30) | €50–150/ha/yr | Monitoring phase |
| **Weighted FOM over 30 years** | **~3–5% of investment** | **Compatible with your 5%** |
| Monitoring, reporting, verification (MRV) | €10–30/ha/yr | Required under CRCF |
| Reversal risk buffer | 10–20% discount on credits | CRCF permanence requirement |

### 3.4 Reversal Risk — A CRCF Requirement You Should Model

The CRCF requires operators to be liable for carbon reversals (fire, storm, pest, land-use change) during the monitoring period. This is typically handled by:
- A **buffer pool**: 10–20% of certified units are held in reserve
- Or equivalently: reduce the effective sequestration rate by 10–20%

**Implementation in PyPSA-Eur:**
```python
# Apply reversal risk discount
reversal_risk = 0.15  # 15% buffer, consistent with CRCF/VCS practices
effective_rate = MAI_CO2 * (1 - reversal_risk)
```

This is simple but significantly affects results — a 15% reversal buffer turns an 8 tCO₂/ha/yr rate into 6.8 tCO₂/ha/yr.

---

## 4. Case Study: Denmark (NUTS-2: DK01–DK05)

Denmark is a good test case because: (a) you're based at AAU, (b) Denmark has significant afforestation potential on agricultural land, (c) Denmark has well-documented NFI data in the Pilli et al. dataset, and (d) Denmark has specific LULUCF targets.

### 4.1 Denmark's LULUCF Context

- Denmark's 2030 LULUCF target: increase net removals relative to 2016–2018 average
- Forest covers ~14.5% of Danish land area (~630,000 ha)
- National target of doubling forest area to ~25% by ~2100 (long-standing policy since 1989)
- Main afforestation species: Norway spruce, Sitka spruce, beech, oak
- Afforestation rates: ~1,500–3,000 ha/yr in recent decades (below target)

### 4.2 Sequestration Rate for Danish NUTS-2 Regions

From the Pilli et al. (2024) dataset, Denmark's main forest types and their approximate MAI:

| Forest type | Rotation (yr) | Standing vol. at harvest (m³/ha) | BCEF (t/m³) | MAI_CO₂ (tCO₂/ha/yr) |
|-------------|:---:|:---:|:---:|:---:|
| Norway spruce (even-aged) | 60 | ~450 | ~0.55 | ~7.5 |
| Sitka spruce | 50 | ~500 | ~0.55 | ~10.1 |
| Beech (high forest) | 100 | ~350 | ~0.65 | ~4.2 |
| Oak (high forest) | 120 | ~250 | ~0.70 | ~3.4 |
| **Weighted mix (60% conifer, 40% broadleaf)** | — | — | — | **~6.5** |

*Note: These are illustrative. Actual values should be computed from the Zenodo dataset files.*

After applying 15% reversal risk buffer: **~5.5 tCO₂/ha/yr**

### 4.3 Available Land from Corine

Using Corine Land Cover classes eligible for afforestation (non-irrigated arable land with low productivity, pastures, etc.), PyPSA-Eur already computes available hectares per clustered node. For Denmark, typical estimates are 200,000–400,000 ha of potential afforestation land (consistent with the national doubling target).

### 4.4 Cost for Denmark

| Parameter | Value | Source |
|-----------|-------|--------|
| Establishment cost | €6,000–8,000/ha | Danish afforestation grants + ForestNavigator D4.4 |
| FOM | 4–5%/yr | Maintenance declining from €300/ha/yr to €100/ha/yr |
| Lifetime | 30 yr | LULUCF 30-year transition |
| Discount rate | 7% (PyPSA-Eur default) | |
| **Annualised cost** | **~€870–1,020/ha/yr** | |
| **Effective rate** | **5.5 tCO₂/ha/yr** | After reversal buffer |
| **Cost per tCO₂** | **€158–185/tCO₂** | |

### 4.5 Comparison with Denmark's LULUCF Targets

This is where the policy alignment pays off. You can now ask:

> *"If the PyPSA-Eur optimiser selects X hectares of afforestation in Denmark, how does that compare to Denmark's LULUCF commitment?"*

- Denmark's total LULUCF budget for 2026–2029 requires increasing net removals
- At 5.5 tCO₂/ha/yr, 10,000 ha of new afforestation ≈ 55 ktCO₂/yr additional removal
- At 50,000 ha (realistic 2025–2050 target) ≈ 275 ktCO₂/yr
- This can be directly compared to Denmark's national LULUCF target gap

### 4.6 Dashboard: PyPSA-Eur Parameters for Denmark

```yaml
# Denmark afforestation node parameters
afforestation_DK:
  # Sequestration
  sequestration_rate: 5.5  # tCO2/ha/yr (MAI × (1 - reversal_risk))
  
  # Costs (LULUCF/CRCF-aligned)
  investment: 7790          # €/ha
  lifetime: 30              # years (LULUCF Art. 5(3) transition period)
  FOM: 5                    # %/year
  # → annualised: ~€1,018/ha/yr → €185/tCO2
  
  # Potential (from Corine)
  available_land: 300000    # ha (illustrative for all DK nodes combined)
  max_annual_removal: 1650  # ktCO2/yr = 300,000 × 5.5
  
  # Policy reference
  accounting_method: "LULUCF gross-net (afforested land)"
  CRCF_unit_type: "Carbon Farming Sequestration Unit (temporary)"
  CRCF_monitoring_period: 30  # years
  reversal_risk_buffer: 0.15
```

---

## 5. How to Present This in a Paper

### Methodology section structure:

1. **Afforestation potential** — Derived from Corine land cover (existing PyPSA-Eur pipeline). Cite the land availability per NUTS-2 region.

2. **Sequestration rate** — Computed as rotation-averaged MAI from Pilli et al. (2024) yield tables, converted to tCO₂/ha/yr using the BCEF from the same dataset, IPCC carbon fractions, and root-to-shoot ratios. Note consistency with LULUCF gross-net accounting of afforested land (Art. 5, Reg. 2018/841). Apply a 15% reversal risk buffer consistent with CRCF permanence requirements.

3. **Cost** — Establishment cost from ForestNavigator D4.4 / national afforestation scheme data. Annual maintenance as FOM. Annualised over a 30-year lifetime corresponding to the LULUCF transition period (Art. 5(3)). Discount rate as per PyPSA-Eur standard (typically 7%).

4. **Case study** — Present Denmark as a validation case, comparing model-derived afforestation potential and cost with Denmark's actual LULUCF 2030 target and national afforestation programme.

5. **Sensitivity analysis** — Vary lifetime (20, 30, 50 yr), reversal risk (0%, 15%, 25%), and establishment cost (low: €3,000, central: €7,790, high: €10,000).

### One-sentence summary for the abstract:

> "Afforestation is modelled as a CDR technology with spatially-resolved sequestration rates derived from EU National Forest Inventory data and costs aligned with the LULUCF Regulation's 30-year accounting period for afforested land, providing policy-consistent estimates of negative emission potential across Europe."

---

## 6. Data Pipeline (Updated with Policy Alignment)

```
┌─────────────────────────────────────────────────────┐
│  SEQUESTRATION RATE                                  │
│                                                      │
│  Pilli et al. (2024) Zenodo → yield tables by        │
│  forest type, age class, NUTS-2                      │
│       ↓                                              │
│  Compute MAI = standing_vol_at_rotation / T          │
│       ↓                                              │
│  Convert: MAI × BCEF × 0.5 × 3.667 × 1.25          │
│       ↓                                              │
│  Apply reversal buffer: × (1 - 0.15)  ← CRCF        │
│       ↓                                              │
│  Result: tCO₂/ha/yr per NUTS-2                       │
├─────────────────────────────────────────────────────┤
│  COSTS                                               │
│                                                      │
│  ForestNavigator D4.4 or national scheme data        │
│       ↓                                              │
│  Establishment: €3,000–8,000/ha (one-time)           │
│  Maintenance: 3–5% FOM/yr                            │
│       ↓                                              │
│  Annualise over 30 yr  ← LULUCF transition period    │
│  with PyPSA-Eur discount rate                        │
│       ↓                                              │
│  Result: €/ha/yr or €/tCO₂ per NUTS-2               │
├─────────────────────────────────────────────────────┤
│  POTENTIAL                                           │
│                                                      │
│  Corine land cover (existing in PyPSA-Eur)           │
│       ↓                                              │
│  Available ha per node/bus                           │
│       ↓                                              │
│  × rate = max annual CDR potential (tCO₂/yr/node)   │
├─────────────────────────────────────────────────────┤
│  VALIDATION                                          │
│                                                      │
│  Compare model outputs with:                         │
│  • National LULUCF 2030 targets (Reg. 2023/839)     │
│  • National afforestation programme rates            │
│  • EU-CBM-HAT projections (Pilli et al. 2024b)      │
│  • Fernandes et al. (2026) PyPSA-Eur results        │
└─────────────────────────────────────────────────────┘
```

---

## 7. References (Additional to Earlier Document)

- **Regulation (EU) 2018/841** — LULUCF Regulation, Art. 5–6 on afforestation/deforestation accounting.
- **Regulation (EU) 2023/839** — Amended LULUCF, 310 MtCO₂eq 2030 target, national budgets 2026–2029.
- **Regulation (EU) 2024/3012** — CRCF, Art. 4 on quality criteria, Art. 8–9 on carbon farming certification.
- **SWD(2021) 651** — 3 Billion Trees Roadmap, establishment cost data.
- **ForestNavigator D4.4** — EU forest management costing module. https://www.forestnavigator.eu/resources/
- **Fernandes et al. (2026)** — arXiv:2603.25663, PyPSA-Eur with afforestation CDR.
- **Avitabile et al. (2024)** — Harmonised forest biomass & increment statistics, Scientific Data 11(1):274.
- **Pilli et al. (2024)** — Volume, increment, BCEF database. Zenodo: 10.5281/zenodo.11387301.
