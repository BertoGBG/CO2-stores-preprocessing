# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
# SPDX-License-Identifier: MIT

"""
Run the full pipeline: download all inputs, then compute all outputs.

Steps
-----
1. download_afforestation_data       → downloads/Biomass_calculations.xlsx
                                       outputs/afforestation/afforestation_nuts_biomass_densities.xlsx
2. download_eurostat_crops           → downloads/eurostat_apro_cpshr_nuts{0,2}_raw.csv
3. calculate_afforestation_rates_nuts2          → outputs/afforestation/afforestation_nuts2_growth_rates.csv
4. calculate_perennials_1G_biofuels_conversion  → outputs/perennialisation/yields_*.csv

Usage
-----
    python zenodo_w_processing/scripts/run_all.py   # from repo root
    python scripts/run_all.py          # from inside the bundle folder
"""

import sys
from pathlib import Path

# Make the scripts folder importable regardless of working directory
sys.path.insert(0, str(Path(__file__).resolve().parent))

from download_afforestation_data import main as download_afforestation
from download_eurostat_crops import main as download_eurostat
from calculate_afforestation_rates_nuts2 import main as calc_afforestation
from calculate_perennials_1G_biofuels_conversion import main as calc_perennials


if __name__ == "__main__":
    print("=== Step 1/4: downloading afforestation data ===")
    download_afforestation()

    print("\n=== Step 2/4: downloading Eurostat crop data ===")
    download_eurostat()

    print("\n=== Step 3/4: calculating afforestation rates ===")
    calc_afforestation()

    print("\n=== Step 4/4: calculating perennial / 1G biofuel yields ===")
    calc_perennials()

    print("\nAll steps completed successfully.")
