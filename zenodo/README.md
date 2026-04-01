# Overall Reference:
https://arxiv.org/abs/2603.25663

# Afforestation datasets

## Raw input data

Two datasets are retrieved from Figshare:

* `data/afforestation_nuts_biomass_densities.xlsx`
  Downloaded from: https://ndownloader.figshare.com/files/43678089

* `data/Biomass_calculations.xlsx`
  Downloaded from: https://figshare.com/ndownloader/files/43678533

Reference: https://www.nature.com/articles/s41597-023-02868-8

The script `scripts/download_afforestation_data.py` reproduces these downloads.

## Preprocessed output

* `data/afforestation_nuts2.csv`
  Afforestation growth rate (t ha⁻¹ y⁻¹) per NUTS2 region, derived from the CBM/JRC
  dataset (`Biomass_calculations.xlsx`). Missing values are filled using a multi-stage
  fallback strategy (NUTS1 propagation → neighbouring regions → country average → nearest
  within UK). Malta and Cyprus are assigned the value of Crete (EL43).

The script `scripts/calculate_afforestation_rates_nuts2.py` reproduces this file from
the downloaded inputs. Run `download_afforestation_data.py` first.

# Raw Eurostat crop data for PyPSA-Eur

This archive contains raw CSV files downloaded from the Eurostat API and used as source data for the PyPSA-Eur workflow.

The datasets enable the conversion of 1st-generation biofuels potentials from the ENSPRESSO biomass database to perennial crops for bioenergy and protein production (perennialisation).

## Contents

- `data/eurostat_apro_cpshr_nuts2_raw.csv`
  Raw Eurostat download at NUTS2 resolution.

- `data/eurostat_apro_cpshr_nuts0_raw.csv`
  Raw Eurostat download at country (NUTS0) resolution.

- `scripts/download_eurostat_crops.py`
  Script used to retrieve the raw CSV files from the Eurostat API.

## Reproducibility note

The retrieval script documents the API queries used to obtain the raw files. Re-running the script in the future may produce different results if the upstream API, dataset coverage, labels, or definitions change. Therefore, the archived CSV files should be treated as the fixed reference inputs used for this version of the workflow.
