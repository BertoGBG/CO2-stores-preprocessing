# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
# SPDX-License-Identifier: MIT


"""
Download raw datasets for afforestation and biomass.

This script retrieves:
1. Figshare dataset (afforestation + biomass densities)
2. CBM/JRC biomass dataset

No preprocessing is performed here.

Outputs
-------
- afforestation_nuts_biomass_densities.xlsx
- CBM_Biomass_calculations.xlsx
"""

from pathlib import Path
import requests


def download_file(url: str, dest: Path):
    print(f"Downloading {url}")
    response = requests.get(url, timeout=120, allow_redirects=True)
    response.raise_for_status()
    if len(response.content) == 0:
        raise RuntimeError(f"Download returned empty content from {url}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(response.content)
    print(f"Saved to {dest} ({len(response.content) / 1024:.1f} kB)")


def main():
    ROOT_DIR = Path(__file__).resolve().parents[2]

    # ---- File 1: Figshare afforestation biomass densities (output dataset) ----
    # reference : https://www.nature.com/articles/s41597-023-02868-8
    url = "https://ndownloader.figshare.com/files/43678089"
    output = ROOT_DIR / "outputs" / "afforestation"/ "afforestation_nuts_biomass_densities.xlsx"
    download_file(url, output)
    print("All downloads completed successfully.")


if __name__ == "__main__":
    main()