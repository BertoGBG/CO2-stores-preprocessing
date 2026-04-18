"""
Visual check of the monthly CO₂ sequestration rates per NUTS-2.

Produces:
  - outputs/afforestation/fig_monthly_profiles_sample.png
      Line plot of monthly rates for 6 representative NUTS-2 regions.
  - outputs/afforestation/fig_seasonal_map.png
      12-panel map of monthly rates across Europe.

Source data:
  - outputs/afforestation/afforestation_nuts2_monthly_rates.csv   (tCO₂ ha⁻¹ month⁻¹)
  - data/nuts/NUTS_RG_03M_2013_4326_LEVL_2.geojson  (local NUTS-2013 boundaries)

Usage:
    python 03_plot_check.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import geopandas as gpd

ROOT_DIR   = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT_DIR / "outputs" / "afforestation"
RATES_CSV  = OUTPUT_DIR / "afforestation_nuts2_monthly_rates.csv"
NUTS_PATH  = ROOT_DIR / "data" / "nuts" / "NUTS_RG_03M_2013_4326_LEVL_2.geojson"

# PyPSA-EUR GeoJSON for Western Balkans proxy geometry (RS/AL/BA/XK absent from Eurostat)
PYPSA_GJ = Path(
    "/Users/albal/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/"
    "01_DTU_research/01_Projects/01_pypsa-eur_AA/pypsa-eur/resources/"
    "regions_onshore_base_s_90.geojson"
)
# Maps pseudo-NUTS2 code → PyPSA node name in regions_onshore_base_s_90.geojson
EXTRA_GEOM = {"RS00": "RS2 0", "AL00": "AL2 0", "BA00": "BA2 0", "XK00": "XK2 0"}

MONTH_NAMES = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]

# Representative NUTS-2 codes spanning a north–south gradient
SAMPLE_REGIONS = {
    "FI1D": "Finland (Lappi)",
    "SE33": "Sweden (Övre Norrland)",
    "DE30": "Germany (Berlin)",
    "FR10": "France (Île-de-France)",
    "ES51": "Spain (Catalonia)",
    "GR41": "Greece (Aegean)",
}


def plot_sample_profiles(rates: pd.DataFrame):
    """Line plot of monthly CO₂ rates for a selection of NUTS-2 regions."""
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.coolwarm_r(np.linspace(0, 1, len(SAMPLE_REGIONS)))

    for (nuts_id, label), color in zip(SAMPLE_REGIONS.items(), colors):
        if nuts_id not in rates.index:
            print(f"  WARNING: {nuts_id} not found in rates, skipping.")
            continue
        ax.plot(
            MONTH_NAMES,
            rates.loc[nuts_id, MONTH_NAMES],
            marker="o",
            label=f"{nuts_id} — {label}",
            color=color,
            linewidth=2,
        )

    ax.set_ylabel("CO₂ sequestration rate (tCO₂ ha⁻¹ month⁻¹)")
    ax.set_title("Monthly afforestation CO₂ rates per NUTS-2 (rotation-averaged MAI × GPP weights)")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_ylim(0, None)
    fig.tight_layout()
    out = OUTPUT_DIR / "fig_monthly_profiles_sample.png"
    fig.savefig(out, dpi=150)
    print(f"  Saved: {out}")
    plt.close(fig)


def plot_seasonal_map(rates: pd.DataFrame, nuts: gpd.GeoDataFrame):
    """12-panel map of monthly CO₂ rates across Europe."""
    merged = nuts.merge(rates[MONTH_NAMES], left_on="NUTS_ID", right_index=True, how="left")

    vmin = 0.0
    vmax = rates[MONTH_NAMES].max().max()

    fig, axes = plt.subplots(3, 4, figsize=(18, 10))
    axes = axes.ravel()
    cmap = "YlGn"

    for month_name, ax in zip(MONTH_NAMES, axes):
        merged.plot(
            column=month_name,
            ax=ax,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            linewidth=0.1,
            edgecolor="white",
            missing_kwds={"color": "lightgrey"},
        )
        ax.set_title(month_name, fontsize=10)
        ax.set_xlim(-25, 45)
        ax.set_ylim(34, 72)
        ax.axis("off")

    # Reserve space at the bottom for the colorbar; avoid tight_layout fighting it
    fig.subplots_adjust(left=0.01, right=0.99, top=0.91, bottom=0.08,
                        wspace=0.04, hspace=0.18)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar_ax = fig.add_axes([0.15, 0.03, 0.70, 0.022])   # [left, bottom, width, height]
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("CO₂ sequestration rate (tCO₂ ha⁻¹ month⁻¹)", fontsize=9)

    fig.suptitle(
        "Monthly afforestation CO₂ sequestration rates per NUTS-2\n"
        "(rotation-averaged MAI from Pilli et al. 2024, seasonalised with FluxCom GPP 2010–2012)",
        fontsize=11,
        y=0.98,
    )
    out = OUTPUT_DIR / "fig_seasonal_map.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out}")
    plt.close(fig)


def main():
    if not RATES_CSV.exists():
        print(f"ERROR: {RATES_CSV} not found. Run 02_compute_nuts2_profiles.py first.")
        return

    rates = pd.read_csv(RATES_CSV, index_col="NUTS_ID")
    print(f"Loaded rates: {rates.shape} — sample regions: {rates.index.tolist()[:5]} ...")

    print("Plotting sample profiles ...")
    plot_sample_profiles(rates)

    if not NUTS_PATH.exists():
        print(f"  Map skipped: NUTS GeoJSON not found at {NUTS_PATH}")
        print("  Run the pipeline first or check the data/nuts/ folder.")
    else:
        print(f"Loading NUTS-2013 geometry from {NUTS_PATH} ...")
        nuts = gpd.read_file(NUTS_PATH)
        if "LEVL_CODE" in nuts.columns:
            nuts = nuts[nuts["LEVL_CODE"] == 2].copy()
        nuts = nuts[["NUTS_ID", "geometry"]].to_crs("EPSG:4326")
        print(f"  {len(nuts)} NUTS-2 regions loaded.")

        # Add proxy geometry for Western Balkans (RS/AL/BA/XK absent from Eurostat GeoJSON)
        if PYPSA_GJ.exists():
            pypsa_gdf = gpd.read_file(str(PYPSA_GJ)).set_index("name")
            extra_rows = []
            for nuts_id, node_name in EXTRA_GEOM.items():
                if node_name in pypsa_gdf.index:
                    extra_rows.append({
                        "NUTS_ID": nuts_id,
                        "geometry": pypsa_gdf.loc[node_name, "geometry"],
                    })
            if extra_rows:
                extra_nuts = gpd.GeoDataFrame(extra_rows, crs="EPSG:4326")
                nuts = pd.concat([nuts, extra_nuts], ignore_index=True)
                print(f"  Added {len(extra_rows)} Western Balkans proxy geometries "
                      f"({[r['NUTS_ID'] for r in extra_rows]}).")
        else:
            print(f"  Warning: PyPSA GeoJSON not found at {PYPSA_GJ}; Balkans will be grey.")

        print("Plotting seasonal map ...")
        plot_seasonal_map(rates, nuts)

    print("\nDone.")


if __name__ == "__main__":
    main()
