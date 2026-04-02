# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
# SPDX-License-Identifier: MIT

"""
Run the full pipeline: download all inputs, then compute all outputs.

Steps
-----
1. download_afforestation_data   → downloads/Biomass_calculations.xlsx
                                   outputs/afforestation/afforestation_nuts_biomass_densities.xlsx
2. download_eurostat_crops       → outputs/perennialisation/eurostat_apro_cpshr_nuts{0,2}_raw.csv
3. calculate_afforestation_rates_nuts2  → outputs/afforestation/afforestation_nuts2.csv

Usage
-----
    python zenodo_aCDRs/scripts/run_all.py   # from repo root
    python scripts/run_all.py                # from inside the bundle folder
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from download_afforestation_data import main as download_afforestation
from download_eurostat_crops import main as download_eurostat
from calculate_afforestation_rates_nuts2 import main as calc_afforestation


if __name__ == "__main__":
    print("=== Step 1/3: downloading afforestation data ===")
    download_afforestation()

    print("\n=== Step 2/3: downloading Eurostat crop data ===")
    download_eurostat()

    print("\n=== Step 3/3: calculating afforestation rates ===")
    calc_afforestation()

    print("\nAll steps completed successfully.")
