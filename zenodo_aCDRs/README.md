# aCDRs input data for PyPSA-Eur

Reference: https://arxiv.org/abs/2603.25663

---

## Overview

This package provides the preprocessed input datasets for three Carbon Dioxide Removal (CDR)
technologies implemented in PyPSA-Eur: **Afforestation**, **Perennialisation**, and
**Enhanced Rock Weathering (ERW)**. It accompanies the paper submitted to [journal].

---

## Folder structure

```
scripts/
  afforestation/
    download_afforestation_data_avitabile.py   ← Method 1: download Figshare biomass densities
    download_zenodo_pilli_afforestation.py     ← Method 2: download Pilli JRC growth library
    download_fluxcom.py                        ← Method 2: download FluxCom GPP data
    01_compute_afforestation_rates_pilli.py    ← Method 2: compute rotation-averaged MAI per NUTS-2
    02_compute_nuts2_profiles.py               ← Method 2: compute monthly GPP seasonal profiles
    03_plot_check.py                           ← diagnostic plots
  perennialisation/
    download_eurostat_crops.py                 ← download Eurostat crop harvest data
  run_all.py                                   ← run full pipeline end-to-end

data/
  fluxcom_raw/      ← FluxCom daily GPP NetCDF files (2010–2012, ~3.6 GB)
  nuts/             ← NUTS-2013 and NUTS-2021 boundary GeoJSON files
  zenodo_Pilli/     ← Pilli et al. (2024) JRC forest growth library
    Volume_increment_database/     ← standing stock and NAI curves
    Volume_biomass_bcef_database/  ← BCEF lookup table
    Volume_biomass_selection/      ← auxiliary biomass selection tables

outputs/
  afforestation/    ← afforestation results (both methods)
  perennialisation/ ← Eurostat crop data (input for PyPSA-Eur)
  ERW/              ← ERW outputs [placeholder]
```

## Quick start

To reproduce all outputs in one step:

```bash
python scripts/run_all.py
```

Individual steps can also be run in isolation — see the script docstrings for details.

---

# Afforestation datasets

Two independent methods are provided for deriving NUTS-2 afforestation CO₂ sequestration
rates. Both are available in PyPSA-Eur via the `afforestation[potential_type]` config key:
`"density"` for Method 1 and `"growth"` for Method 2.

## Method 1: Avitabile et al. biomass density dataset (direct download)

### Raw input

The dataset is downloaded directly from Figshare:

- URL: https://ndownloader.figshare.com/files/43678089
- Reference: Avitabile et al. (2024), *Scientific Data*, https://www.nature.com/articles/s41597-023-02868-8

Script: `scripts/afforestation/download_afforestation_data_avitabile.py`

### Output

* `outputs/afforestation/afforestation_nuts_biomass_densities.xlsx`
  Above-ground biomass density (t DM ha⁻¹) per NUTS-2 region, derived from the Avitabile
  et al. dataset. Used in PyPSA-Eur as the basis for the `"density"` sequestration rate.

---

## Method 2: Pilli et al. yield tables + FluxCom seasonal profiles

This is the recommended method described in the accompanying paper. It derives
rotation-averaged CO₂ sequestration rates from European forest inventory data and
distributes them across calendar months using observed Gross Primary Production (GPP).
A full methodological description is given in the supplementary material of the paper.

### Data sources

#### Pilli et al. (2024) — JRC forest growth library

- **Zenodo**: https://zenodo.org/records/11387301
- **Reference**: Pilli, R., Blujdea, V., Rougieux, P. (2024). *JRC Forest Carbon Model calibration
  data for the period 2010–2020*. https://publications.jrc.ec.europa.eu/repository/handle/JRC135639
- **Downloaded to**: `data/zenodo_Pilli/`

The library provides age-class-resolved volume and increment curves for 222 forest types
across 25 EU Member States (EU-27 excluding Cyprus and Malta), derived from National Forest
Inventories. It includes:

  - **Standing stock** (m³ ha⁻¹): net merchantable volume by 10-year age class for even-aged stands.
  - **Net Annual Increment** (NAI, m³ ha⁻¹ yr⁻¹): harmonised volume growth rate by age class.
  - **Biomass Conversion and Expansion Factors** (BCEF, t_DM m⁻³): age-class-resolved factors
    converting merchantable volume to total aboveground dry biomass.

Script: `scripts/afforestation/download_zenodo_pilli_afforestation.py`

#### FluxCom RS+METEO — daily GPP (Jung et al., 2020)

- **Source**: anonymous FTP at `ftp.bgc-jena.mpg.de` (Max Planck Institute for Biogeochemistry)
- **Reference**: Jung, M. et al. (2020). *Biogeosciences*, 17(5), 1343–1365.
  https://doi.org/10.5194/bg-17-1343-2020
- **Files**: `GPP.RS_METEO.FP-ALL.MLM-ALL.METEO-ERA5.720_360.daily.{year}.nc`
  (years 2010–2012, ~1.2 GB/year, ~3.6 GB total)
- **Unit**: gC m⁻² day⁻¹ | Resolution: 0.5° global
- **Downloaded to**: `data/fluxcom_raw/`

Three years (2010–2012) are used to construct a stable climatological seasonal cycle.
The ERA5-forced ensemble was selected for consistency with the climate data used in PyPSA-Eur.

Script: `scripts/afforestation/download_fluxcom.py`

### Methodology summary

**Annual sequestration rates** (`01_compute_afforestation_rates_pilli.py`):

For each combination of forest type, NUTS-2 region, and management type, the optimal
rotation age T\* is identified as the age that maximises the volumetric Mean Annual
Increment (MAI = standing stock / age). The rotation-averaged CO₂ sequestration rate is
then computed as:

```
MAI_CO2 = (V(T*) / T*) × BCEF(T*) × 0.5 × (44/12) × (1 + 0.25)
          [tCO₂ ha⁻¹ yr⁻¹]
```

where 0.5 is the IPCC carbon fraction, 44/12 is the CO₂-to-C molecular mass ratio,
and 0.25 is the root-to-shoot ratio. Only productive even-aged high-forest stands
(management types H, HP, HS, MAN) are included.

Regions not covered by the Pilli library (non-EU countries, small islands, city-states)
are filled via a cascade: NUTS-1 propagation → country-level propagation → distance-based
neighbour mean (≤100 km, iterated) → country mean → Mediterranean analogue for Malta and Cyprus.

**Monthly temporal profiles** (`02_compute_nuts2_profiles.py`):

For each NUTS-2 region, the 12-month GPP climatology (averaged over 2010–2012) is
aggregated from the 0.5° FluxCom grid using point-in-polygon assignment to NUTS-2013
boundaries, then normalised to unit sum. The monthly CO₂ sequestration rate is:

```
Rate_m,r = w_m,r × MAI_CO2,r     [tCO₂ ha⁻¹ month⁻¹]
```

where `w_m,r` are the normalised monthly GPP weights (sum to 1). Regions with no valid
GPP signal fall back to: neighbour mean (≤100 km) → country mean → uniform 1/12.

### Outputs

* `outputs/afforestation/afforestation_rates_per_forest_type.csv`
  Detailed per-forest-type rotation-averaged MAI_CO2 before NUTS-2 aggregation.

* `outputs/afforestation/afforestation_rates_nuts2.csv`
  NUTS-2 averaged sequestration rate for regions with direct Pilli library coverage.

* `outputs/afforestation/afforestation_rates_nuts2_full.csv`
  Full coverage: all ~320 NUTS-2 regions in the PyPSA-Eur model, with fallback values
  recorded per region. **Primary input for PyPSA-Eur** (`"growth"` method).

* `outputs/afforestation/afforestation_nuts2_monthly_weights.csv`
  12 normalised monthly GPP weights per NUTS-2 region (rows sum to 1).
  **Used by PyPSA-Eur** to set the time-varying efficiency of the afforestation Link.

* `outputs/afforestation/afforestation_nuts2_monthly_rates.csv`
  Monthly CO₂ sequestration rates (tCO₂ ha⁻¹ month⁻¹) per NUTS-2 region.

* `outputs/afforestation/fig_monthly_profiles_sample.png`
  Diagnostic figure: seasonal GPP profiles for a sample of NUTS-2 regions.

* `outputs/afforestation/fig_seasonal_map.png`
  Diagnostic map: spatial pattern of peak-month sequestration across Europe.

---

# Perennial crops and 1st-generation biofuels datasets

## Raw input

Crop harvest data are retrieved live from the **Eurostat API** (dataset `apro_cpshr`).
No preprocessing beyond format conversion is applied.

Script: `scripts/perennialisation/download_eurostat_crops.py`

## Outputs

* `outputs/perennialisation/eurostat_apro_cpshr_nuts2_raw.csv`
  Raw Eurostat crop harvest data at NUTS-2 resolution (dataset apro_cpshr, 2017–2020).
  **Used by PyPSA-Eur** as input for the perennialisation workflow.

* `outputs/perennialisation/eurostat_apro_cpshr_nuts0_raw.csv`
  Raw Eurostat crop harvest data at country (NUTS-0) resolution.
  **Used by PyPSA-Eur** as fallback for NUTS-2 regions with missing data.

---

# Enhanced Rock Weathering (ERW) dataset

*[Placeholder — methodology and scripts under development.]*

## Outputs

* `outputs/ERW/World_Ecological_BioVal_cluster.tif`
  Raster of global ecological biomes / biogeochemical value clusters used to estimate
  ERW CDR potential. Source and processing methodology to be documented upon completion.

---

# PyPSA-Eur output files summary

The following files from this package are directly read by the PyPSA-Eur workflow
(branch [`a_CDRs`](https://github.com/BertoGBG/pypsa-eur/tree/a_CDRs)):

| File | Technology | Description |
|------|------------|-------------|
| `outputs/afforestation/afforestation_nuts_biomass_densities.xlsx` | Afforestation | Biomass density per NUTS-2 (`"density"` method) |
| `outputs/afforestation/afforestation_rates_nuts2_full.csv` | Afforestation | Rotation-averaged MAI per NUTS-2 (`"growth"` method) |
| `outputs/afforestation/afforestation_nuts2_monthly_weights.csv` | Afforestation | Monthly GPP weights for seasonal dispatch profile |
| `outputs/perennialisation/eurostat_apro_cpshr_nuts2_raw.csv` | Perennialisation | Eurostat crop harvest at NUTS-2 |
| `outputs/perennialisation/eurostat_apro_cpshr_nuts0_raw.csv` | Perennialisation | Eurostat crop harvest at NUTS-0 |
| `outputs/ERW/World_Ecological_BioVal_cluster.tif` | ERW | Ecological biome clusters for ERW potential |

---

## Reproducibility note

Re-running the download scripts in the future may produce different results if the
upstream APIs, dataset coverage, labels, or definitions change. The archived files
in this package should therefore be treated as the fixed reference inputs for this
version of the workflow.
