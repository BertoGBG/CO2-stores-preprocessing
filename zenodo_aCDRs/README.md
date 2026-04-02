# aCDRs input data for PyPSA-Eur

Reference: https://arxiv.org/abs/2603.25663

---

## Folder structure

```
scripts/    ← download and calculation scripts
downloads/  ← raw input files retrieved from external sources
outputs/
  afforestation/     ← afforestation results
  perennialisation/  ← Eurostat crop data (input for PyPSA-Eur perennialisation)
outputs.zip ← all output files bundled for download
```

## Quick start

To reproduce all outputs in one step:

```bash
python scripts/run_all.py
```

---

# Afforestation datasets

## Raw input (downloads/)

* `downloads/Biomass_calculations.xlsx`
  CBM/JRC biomass dataset downloaded from Figshare:
  https://ndownloader.figshare.com/files/43678533
  Reference: https://www.nature.com/articles/s41597-023-02868-8

Run `scripts/download_afforestation_data.py` to reproduce this file.

## Outputs (outputs/afforestation/)

* `outputs/afforestation/afforestation_nuts_biomass_densities.xlsx`
  Afforestation biomass densities dataset downloaded from Figshare:
  https://ndownloader.figshare.com/files/43678089
  Reference: https://www.nature.com/articles/s41597-023-02868-8

* `outputs/afforestation/afforestation_nuts2.csv`
  Afforestation growth rate (t ha⁻¹ y⁻¹) per NUTS2 region, derived from the CBM/JRC
  dataset (`Biomass_calculations.xlsx`). Missing values are filled using a multi-stage
  fallback strategy (NUTS1 propagation → neighbouring regions → country average → nearest
  within UK). Malta and Cyprus are assigned the value of Crete (EL43).

Run `scripts/calculate_afforestation_rates_nuts2.py` to reproduce this file.
Requires `downloads/Biomass_calculations.xlsx` — run the download script first.

---

# Perennial crops and 1st-generation biofuels datasets

## Outputs (outputs/perennialisation/)

* `outputs/perennialisation/eurostat_apro_cpshr_nuts2_raw.csv`
  Raw Eurostat crop harvest data at NUTS2 resolution (dataset apro_cpshr, 2017–2020).

* `outputs/perennialisation/eurostat_apro_cpshr_nuts0_raw.csv`
  Raw Eurostat crop harvest data at country (NUTS0) resolution.

These files are used as inputs by the PyPSA-Eur perennialisation workflow.
Run `scripts/download_eurostat_crops.py` to reproduce them.

---

## Reproducibility note

Re-running the download scripts in the future may produce different results if the
upstream APIs, dataset coverage, labels, or definitions change. The archived files
should therefore be treated as the fixed reference inputs for this version of the workflow.
