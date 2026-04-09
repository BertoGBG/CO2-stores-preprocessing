# Afforestation CO₂ Sequestration Rates for PyPSA-Eur: Methodology Proposals

## 1. The Core Problem

We need a **single annualised sequestration rate** (tCO₂ ha⁻¹ yr⁻¹) per NUTS-2 region that is meaningful for energy system optimisation. The difficulty is that real forests follow a sigmoidal growth curve — fast growth when young, peaking around age 30–60, then levelling off — and are typically harvested on 60–100 year rotations. The optimiser doesn't model forest age classes; it sees afforestation as a CDR technology with a **cost per tCO₂** and a **potential per region** (hectares × rate).

The Corine land cover work already handles the *potential hectares*. What remains is the *rate* and how to make it spatially resolved and scientifically defensible.

---

## 2. Understanding our Sources

### Nabuurs et al. (2013/2018) — Nature Climate Change
This work documents the *existing* European forest carbon sink (~436 MtCO₂eq/yr for the EU) and its emerging saturation. It provides aggregate country/regional sink values for *established forests of mixed ages*, not specifically for newly afforested land. Not directly usable for our purpose without significant reinterpretation, because it blends old growth, managed forests, and new plantations.

### Avitabile et al. (2024) — Scientific Data (your third link)
This is the **harmonised biomass & increment dataset** for 38 European countries at NUTS-1 to NUTS-3 resolution used in the model.
It provides:
- **Net Annual Increment (NAI)** in m³ ha⁻¹ yr⁻¹ for existing forests (2010–2020)
- **Aboveground biomass stock** in t ha⁻¹ for 2020, also for young forstes (1-19y) but not growth rates directly.
- Sub-national resolution for 24 countries

This gives you the *average productivity of existing forests per region* but again reflects a mix of age classes, not specifically young afforestation as we need to assume and average age for young forests to xalulate the rates. Note: using a 10 years average age the average growth rate is about 5.5 t/ha y a bit lower than the EU general estimates at 8 t/ha y.

### Pilli et al. (2024) — Zenodo dataset
This is the **age-class-resolved** volume and increment library (222 forest types, 48 management types, 25 EU Member States). It provides standing volume and net increment **by age class** for each forest type and region (often NUTS-2). Combined with the BCEF data in the same dataset, you can convert volume increments to biomass and then to CO₂. **This is the most useful dataset for our purpose** because it lets you compute rotation-averaged values.

### EU publication (op.europa.eu link)
This is the JRC report documenting the methodology behind the EU-CBM-HAT carbon budget model, which uses the Pilli et al. data. It provides the full conversion chain from volume → biomass → carbon.

---

## 3. Three Proposals

### Proposal A: Rotation-Averaged Rate from Pilli et al. Yield Tables (Recommended)

**Concept**: Use the age-class-resolved increment data to compute a *rotation-averaged* annual CO₂ sequestration rate, which represents the steady-state annual removal you would get from a landscape of afforested plots planted in staggered years. With this method we take in cosideration the forest mix and weighted for age until harvesting.

**Why this is a better idea than averge growth rates from ABG density**: In an energy system model with a planning horizon of 2025–2050, afforestation decisions made in 2025 will produce forests of different ages by 2050. If we assume continuous planting (especially in transition studies), the aggregate removal rate converges to the rotation average. This is exactly the annualised value we need.

**Method**:

1. **Download** the Pilli et al. (2024) Zenodo dataset: `Volume_increment_database.zip`

2. **Select forest types per region**: For each NUTS-2 region, identify the 1–3 dominant forest types (e.g., "Norway spruce high forest", "Beech high forest") from the database. Alternatively, use a weighted mix matching the existing forest composition in each region.

3. **Extract the net annual increment (NAI) curve** by age class for each forest type:
   - The data gives NAI in m³ ha⁻¹ yr⁻¹ (merchantable volume, under bark) per 10- or 20-year age class.

4. **Convert to CO₂**:
   - Use the BCEF from the companion `Volume_biomass_bcef_database.zip` to convert m³ → tonnes dry biomass
   - Apply carbon fraction: ~0.5 tC per t dry biomass (IPCC default)
   - Convert to CO₂: multiply by 3.667 (44/12)
   - **Add belowground biomass**: use IPCC root-to-shoot ratio (~0.20–0.30 for temperate forests) to account for root carbon
   - **Optionally add soil carbon**: newly afforested land accumulates soil organic carbon, typically ~0.5–1.5 tCO₂ ha⁻¹ yr⁻¹ in temperate Europe for the first decades. This is more uncertain.

5. **Compute the rotation-averaged rate**:
   ```
   rate_avg = (1/T) × Σ_{age=0}^{T} NAI_CO2(age)
   ```
   Where T is the rotation length (e.g., 60–80 years for conifers, 80–120 years for broadleaves). This is equivalent to:
   ```
   rate_avg = Total_standing_CO2_at_harvest / T
   ```
   This gives you the *mean annual increment* (MAI) in CO₂ terms.

6. **Result**: One value per NUTS-2 region (tCO₂ ha⁻¹ yr⁻¹), accounting for the local species mix and growing conditions.

**Expected range**: 5–15 tCO₂ ha⁻¹ yr⁻¹ for most of temperate Europe, lower (3–7) for Scandinavia and southern Mediterranean, higher (10–18) for Central Europe.

**Advantages**:
- Uses the most granular, harmonised EU dataset available
- Physically consistent: represents what a landscape-scale afforestation programme would deliver over time
- Consistent with the JRC's own carbon accounting methodology (EU-CBM-HAT)
- Sub-national resolution matching NUTS-2

**Disadvantages**:
- Requires some data processing (but straightforward)
- Assumes typical managed rotation, not conservation forestry
- Does not capture climate change effects on future growth

---

### Proposal B: Direct Use of Avitabile et al. (2024) Net Annual Increment

**Concept**: Use the already-computed harmonised NAI per NUTS region as a proxy for what newly afforested land would produce.

**Method**:

1. **Download** the Avitabile et al. dataset from Scientific Data (supplementary files).

2. **Extract NAI** per NUTS region in m³ ha⁻¹ yr⁻¹ (this is for the *average existing forest*).

3. **Convert to CO₂** using country-level BCEF values:
   - NAI (m³/ha/yr) × BCEF (t biomass/m³) × 0.5 (carbon fraction) × 3.667 (CO₂/C) × 1.25 (root:shoot adjustment)
   - Or use the simpler approximation: NAI × 0.7–1.0 tCO₂ per m³ (this factor combines all conversions and is region-dependent).

4. **Apply a correction factor**: The existing forest NAI reflects a mixed-age landscape. For *new afforestation*, the first 10–15 years have near-zero merchantable increment, while years 20–60 have above-average increment. Over a rotation, the MAI is typically **70–90% of the current average NAI** of established forests. Apply a factor of ~0.8.

**Expected result**: Similar range to Proposal A but less precise.

**Advantages**:
- Very fast to implement
- Already at NUTS resolution, harmonised
- Avoids dealing with age-class data

**Disadvantages**:
- Less accurate: existing forest NAI ≠ afforestation MAI
- The 0.8 correction factor is crude
- Does not distinguish species/management types

---

### Proposal C: IPCC Tier-1 Default Values with Regional Scaling

**Concept**: Use IPCC default sequestration rates for afforestation, scaled by a regional productivity index derived from Avitabile et al.

**Method**:

1. **Start from IPCC AR6/2019 Refinement defaults**:
   - Temperate oceanic broadleaf afforestation: ~5–8 tCO₂ ha⁻¹ yr⁻¹ (first 20 years, above + belowground)
   - Temperate continental broadleaf: ~4–7 tCO₂ ha⁻¹ yr⁻¹
   - Temperate conifer afforestation: ~7–12 tCO₂ ha⁻¹ yr⁻¹
   - Boreal conifer: ~2–5 tCO₂ ha⁻¹ yr⁻¹

2. **Assign each NUTS-2 region** to an IPCC ecological zone (this mapping exists in the IPCC EFD).

3. **Create a spatial scaling factor**: Using the Avitabile et al. NAI, compute each NUTS-2 region's productivity relative to its ecological zone average. Use this to scale the IPCC defaults.

4. **Apply a long-term averaging**: The IPCC "first 20 years" rates are biased high because they capture the fastest-growing phase. For a rotation-averaged value, reduce by ~30%.

**Advantages**:
- Consistent with IPCC methodology (easy to defend in publications)
- Very simple implementation
- Good fallback for regions where Pilli et al. data are missing

**Disadvantages**:
- Low spatial resolution (ecological zones, not NUTS-2)
- The scaling factor helps but is still approximate
- Known to underestimate carbon uptake by ~32% globally (Cook-Patton et al., 2020)

---

## 4. Implementation plan

**Use Proposal A** as the primary approach. It is the most defensible and uses exactly the data infrastructure that the JRC built for EU LULUCF accounting. The Pilli et al. dataset already sits behind the EU-CBM-HAT model that EU member states use for their carbon sink projections.

**The key insight for PyPSA-Eur implementation**:

The rotation-averaged MAI (Mean Annual Increment) is the correct metric for an energy system model. It answers the question: *"If I convert 1 hectare per year to forest, what is the steady-state annual CO₂ removal?"* This sidesteps the entire growth-curve complexity.

Concretely, if in a NUTS-2 region:
- A spruce plantation reaches 400 m³/ha at harvest age 80 years
- BCEF converts this to ~200 t dry biomass/ha (above + belowground)
- That equals ~100 tC/ha = ~367 tCO₂/ha
- **MAI = 367/80 ≈ 4.6 tCO₂ ha⁻¹ yr⁻¹**

This is conservative (it ignores soil carbon and deadwood accumulation) but it is what you can robustly defend.

---

## 5. Practical Notes for PyPSA-Eur Implementation

### Integration with Corine Land Cover
You already have available land per region from Corine. The afforestation potential (ha) × MAI rate (tCO₂/ha/yr) gives you the maximum annual CDR potential per node/bus.

### Cost assumptions
Afforestation costs vary by region but typical European values are:
- Establishment cost: €2,000–8,000 per hectare (one-time)
- Annual maintenance: €50–200 per hectare per year
- Annualised over rotation: roughly €5–50 per tCO₂

### Consistency with Fernandes et al. (2026)
Note that a very recent paper (Fernandes et al., March 2026, arXiv:2603.25663) has already implemented afforestation in PyPSA-Eur. Their work uses a similar approach — regionalised sequestration potentials combined with Corine land availability. You should review their methodology and code (likely merged or being merged into PyPSA-Eur) to ensure consistency or build upon it.

### What about harvest and permanence?
For energy system modelling, the convention is:
- Assume the forest is maintained (not harvested during the planning horizon), OR
- If harvested, assume replanting — the rotation-averaged rate already accounts for this cycle
- The wood products from harvesting create a separate carbon pool (HWP), which you can optionally model

### Sensitivity
Consider running scenarios with:
- A "low" rate: MAI × 0.7 (accounting for establishment losses, poor soils)
- A "central" rate: MAI as computed
- A "high" rate: MAI × 1.3 (accounting for soil carbon gains)

---

## 6. Data Pipeline Summary

```
Pilli et al. (2024) Zenodo dataset
    │
    ├── Volume_increment_database.zip
    │       → NAI(age, forest_type, NUTS2) in m³/ha/yr
    │
    └── Volume_biomass_bcef_database.zip
            → BCEF(forest_type, age) in t_biomass/m³
                │
                ▼
    MAI_volume = Standing_volume_at_rotation / Rotation_length  [m³/ha/yr]
                │
                ▼
    MAI_biomass = MAI_volume × BCEF  [t_dry_biomass/ha/yr]
                │
                ▼
    MAI_CO2 = MAI_biomass × 0.5 × 3.667 × (1 + root:shoot)  [tCO₂/ha/yr]
                │
                ▼
    Per NUTS-2 region: weighted average across forest types
                │
                ▼
    Merge with Corine land availability → max CDR potential per node
```

---

## References

- Avitabile, V. et al. (2024). Harmonised statistics and maps of forest biomass and increment in Europe. *Scientific Data*, 11(1), 274.
- Pilli, R. et al. (2024). Volume, increment, and aboveground biomass data series and BCEF for the main forest types of EU Member States. *Annals of Forest Science*, 81, 35. Data: https://zenodo.org/records/11387301
- Nabuurs, G.J. et al. (2013). First signs of carbon sink saturation in European forest biomass. *Nature Climate Change*, 3, 792–796.
- Fernandes, R. et al. (2026). Exploring carbon dioxide removal strategies to help decarbonise Europe using high-resolution modelling. arXiv:2603.25663.
- IPCC (2019). 2019 Refinement to the 2006 IPCC Guidelines for National Greenhouse Gas Inventories. Volume 4: AFOLU.
- Cook-Patton, S.C. et al. (2020). Mapping carbon accumulation potential from global natural forest regrowth. *Nature*, 585, 545–550.
