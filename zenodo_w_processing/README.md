# Overall Reference:
https://arxiv.org/abs/2603.25663

---

## Folder structure

```
scripts/    ← all scripts (download + calculate)
downloads/  ← raw input files retrieved from external sources
outputs/    ← results produced by the calculate scripts
```

## Quick start

To reproduce all outputs in one step:

```bash
python scripts/run_all.py
```

This downloads all inputs and runs both calculation scripts in sequence.
Individual scripts can also be run separately — see the sections below.

---

# Afforestation datasets

## Raw input (downloads/)

* `downloads/Biomass_calculations.xlsx`
  CBM/JRC biomass dataset downloaded from Figshare:
  https://ndownloader.figshare.com/files/43678533
  Reference: https://www.nature.com/articles/s41597-023-02868-8

Run `scripts/download_afforestation_data.py` to reproduce this file.

## Outputs (outputs/)

* `outputs/afforestation/afforestation_nuts_biomass_densities.xlsx`
  Afforestation biomass densities dataset downloaded from Figshare:
  https://ndownloader.figshare.com/files/43678089
  Reference: https://www.nature.com/articles/s41597-023-02868-8

* `outputs/afforestation/afforestation_nuts2_growth_rates.csv`
  Afforestation growth rate (t ha⁻¹ y⁻¹) per NUTS2 region, derived from the CBM/JRC
  dataset (`Biomass_calculations.xlsx`). Missing values are filled using a multi-stage
  fallback strategy (NUTS1 propagation → neighbouring regions → country average → nearest
  within UK). Malta and Cyprus are assigned the value of Crete (EL43).

Run `scripts/calculate_afforestation_rates_nuts2.py` to reproduce this file.
Requires `downloads/Biomass_calculations.xlsx` — run the download script first.

---

# Perennial crops and 1st-generation biofuels datasets

## Raw inputs (downloads/)

* `downloads/eurostat_apro_cpshr_nuts2_raw.csv`
  Raw Eurostat crop harvest data at NUTS2 resolution (dataset apro_cpshr, 2017–2020).

* `downloads/eurostat_apro_cpshr_nuts0_raw.csv`
  Raw Eurostat crop harvest data at country (NUTS0) resolution.

Run `scripts/download_eurostat_crops.py` to reproduce these files.

## Outputs (outputs/)

* `outputs/perennialisation/yields_perennials_1G_biofuels.csv`
  Combined yield table for all categories (one column each): cereals, sugar beet,
  rape seed and perennial grasses — harmonized to NUTS2 2021.

* `outputs/perennialisation/yields_MINBIOCRP11_nuts2.csv` — cereals → bioethanol (MWh/ha)
* `outputs/perennialisation/yields_MINBIOCRP21_nuts2.csv` — sugar beet → bioethanol (MWh/ha)
* `outputs/perennialisation/yields_MINBIORPS1_nuts2.csv`  — rape/sunflower/soy → biodiesel (MWh/ha)
* `outputs/perennialisation/yields_perennials_nuts2.csv`  — perennial grasses, weighted mean (t/ha dry matter)
* `outputs/perennialisation/yields_perennials_max_nuts2.csv` — perennial grasses, regional max (t/ha dry matter)

Run `scripts/calculate_perennials_1G_biofuels_conversion.py` to reproduce these files.
Requires the Eurostat CSV files — run the download script first.

---

## Reproducibility note

Re-running the download scripts in the future may produce different results if the
upstream APIs, dataset coverage, labels, or definitions change. The archived files in
`downloads/` should therefore be treated as the fixed reference inputs for this version
of the workflow.
