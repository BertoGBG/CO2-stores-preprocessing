# Overall Reference: 
https://arxiv.org/abs/2603.25663

# Afforestation datasets

Two datasets are retrieved from Figshare:

* `afforestation_nuts_biomass_densities.xlsx`
  Downloaded from: https://ndownloader.figshare.com/files/43678089

* `Biomass_calculations.xlsx`
  Downloaded from: https://figshare.com/ndownloader/files/43678533

The script `download_afforestation_and_biomass_data.py` reproduces these downloads.

No preprocessing is applied at this stage. All further data transformations are implemented within the PyPSA-Eur workflow.

The datasets enable the calculation of Afforestation potentials with two different methods

# Raw Eurostat crop data for PyPSA-Eur

This archive contains raw CSV files downloaded from the Eurostat API and used as source data for the PyPSA-Eur workflow.

The datasets enable the conversion of 1st - generation biofuels potentials from ENSPRESSO biomass database to perennials crops for bioenergy and protein production (perennialisation)

## Contents

- `data/eurostat_apro_cpshr_nuts2_raw.csv`  
  Raw Eurostat download at NUTS2 resolution.

- `data/eurostat_apro_cpshr_nuts0_raw.csv`  
  Raw Eurostat download at country (NUTS0) resolution.

- `scripts/download_eurostat_crop_data.py`  
  Script used to retrieve the raw CSV files from the Eurostat API.

## Scope of this archive

This Zenodo record contains only the original downloaded source data and the retrieval script.

These later processing steps are carried out within the PyPSA-Eur workflow.

## Reproducibility note

The retrieval script documents the API queries used to obtain the raw files. Re-running the script in the future may produce different results if the upstream API, dataset coverage, labels, or definitions change. Therefore, the archived CSV files should be treated as the fixed reference inputs used for this version of the workflow.