# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
# SPDX-License-Identifier: MIT

"""
Run the full pipeline: download all inputs, then compute all outputs.

Steps
-----
Afforestation – Method 1 (Avitabile / Figshare):
  1. download_afforestation_data_avitabile  → outputs/afforestation/afforestation_nuts_biomass_densities.xlsx

Afforestation – Method 2 (Pilli + FluxCom):
  2. download_zenodo_pilli_afforestation    → data/zenodo_Pilli/
  3. download_fluxcom                       → data/fluxcom_raw/
  4. 01_compute_afforestation_rates_pilli   → outputs/afforestation/afforestation_rates_nuts2_full.csv
  5. 02_compute_nuts2_profiles              → outputs/afforestation/afforestation_nuts2_monthly_weights.csv
                                               outputs/afforestation/afforestation_nuts2_monthly_rates.csv
  6. 03_plot_check                          → outputs/afforestation/fig_*.png  (optional diagnostics)

Perennialisation:
  7. download_eurostat_crops                → outputs/perennialisation/eurostat_apro_cpshr_nuts{0,2}_raw.csv

Usage
-----
    python scripts/run_all.py   # from inside the bundle folder
"""

import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent

STEPS = [
    # (label, script_path)
    ("1/7 – download Avitabile/Figshare afforestation data",
     SCRIPTS / "afforestation" / "download_afforestation_data_avitabile.py"),
    ("2/7 – download Pilli et al. JRC forest growth library",
     SCRIPTS / "afforestation" / "download_zenodo_pilli_afforestation.py"),
    ("3/7 – download FluxCom GPP data (2010–2012, ~3.6 GB)",
     SCRIPTS / "afforestation" / "download_fluxcom.py"),
    ("4/7 – compute rotation-averaged MAI per NUTS-2 (Pilli method)",
     SCRIPTS / "afforestation" / "01_compute_afforestation_rates_pilli.py"),
    ("5/7 – compute monthly GPP seasonal profiles per NUTS-2",
     SCRIPTS / "afforestation" / "02_compute_nuts2_profiles.py"),
    ("6/7 – generate diagnostic plots",
     SCRIPTS / "afforestation" / "03_plot_check.py"),
    ("7/7 – download Eurostat crop harvest data",
     SCRIPTS / "perennialisation" / "download_eurostat_crops.py"),
]

if __name__ == "__main__":
    for label, script in STEPS:
        print(f"\n=== Step {label} ===")
        result = subprocess.run(
            [sys.executable, str(script)],
            cwd=SCRIPTS.parent,  # run from the bundle root
        )
        if result.returncode != 0:
            print(f"\nERROR: step '{label}' failed (exit code {result.returncode}).")
            sys.exit(result.returncode)

    print("\nAll steps completed successfully.")
