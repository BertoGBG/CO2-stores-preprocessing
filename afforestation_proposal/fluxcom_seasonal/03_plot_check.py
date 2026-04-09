"""
Quick visual check of the monthly GPP weights per NUTS2.

Produces:
  - output/fig_monthly_profiles_sample.png  — line plot for 6 representative regions
  - output/fig_seasonal_map.png             — 12-panel map of weights across Europe

Usage:
    python 03_plot_check.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import geopandas as gpd

HERE = Path(__file__).parent
OUTPUT_DIR = HERE / "output"
WEIGHTS_CSV = OUTPUT_DIR / "nuts2_monthly_weights.csv"

MONTH_NAMES = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]

# Representative NUTS2 codes spanning a north–south gradient
SAMPLE_REGIONS = {
    "FI1D": "Finland (Lappi)",
    "SE33": "Sweden (Övre Norrland)",
    "DE30": "Germany (Berlin)",
    "FR10": "France (Île-de-France)",
    "ES51": "Spain (Catalonia)",
    "GR41": "Greece (Aegean)",
}

NUTS_URL = (
    "https://gisco-services.ec.europa.eu/distribution/v2/nuts/geojson/"
    "NUTS_RG_10M_2021_4326_LEVL_2.geojson"
)


def plot_sample_profiles(weights: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.coolwarm_r(np.linspace(0, 1, len(SAMPLE_REGIONS)))

    for (nuts_id, label), color in zip(SAMPLE_REGIONS.items(), colors):
        if nuts_id not in weights.index:
            print(f"  WARNING: {nuts_id} not found in weights, skipping.")
            continue
        ax.plot(
            MONTH_NAMES,
            weights.loc[nuts_id] * 12,  # ×12 → fraction of uniform baseline
            marker="o",
            label=f"{nuts_id} — {label}",
            color=color,
            linewidth=2,
        )

    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, label="Uniform baseline")
    ax.set_ylabel("Relative monthly GPP (1.0 = annual mean)")
    ax.set_title("Seasonal GPP profiles per NUTS2 (FluxCom climatology)")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_ylim(0, None)
    fig.tight_layout()
    out = OUTPUT_DIR / "fig_monthly_profiles_sample.png"
    fig.savefig(out, dpi=150)
    print(f"  Saved: {out}")
    plt.close(fig)


def plot_seasonal_map(weights: pd.DataFrame, nuts: gpd.GeoDataFrame):
    merged = nuts.merge(weights, left_on="NUTS_ID", right_index=True, how="left")

    fig, axes = plt.subplots(3, 4, figsize=(18, 10))
    axes = axes.ravel()
    cmap = "YlGn"
    vmin, vmax = 0.0, weights.max().max()

    for i, (month_name, ax) in enumerate(zip(MONTH_NAMES, axes)):
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

    # Shared colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.02, pad=0.02)
    cbar.set_label("Monthly weight (fraction of annual total)", fontsize=9)

    fig.suptitle(
        "Monthly GPP seasonal weights per NUTS2 (FluxCom RS ensemble, 2008–2012)",
        fontsize=12,
        y=1.01,
    )
    fig.tight_layout()
    out = OUTPUT_DIR / "fig_seasonal_map.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out}")
    plt.close(fig)


def main():
    if not WEIGHTS_CSV.exists():
        print(f"ERROR: {WEIGHTS_CSV} not found. Run 02_compute_nuts2_profiles.py first.")
        return

    weights = pd.read_csv(WEIGHTS_CSV, index_col="NUTS_ID")
    print(f"Loaded weights: {weights.shape} — {weights.index.tolist()[:5]} ...")

    print("Plotting sample profiles...")
    plot_sample_profiles(weights)

    print("Loading NUTS2 geometry for map...")
    try:
        nuts = gpd.read_file(NUTS_URL)
        nuts = nuts[nuts["LEVL_CODE"] == 2][["NUTS_ID", "geometry"]].to_crs("EPSG:4326")
        print("Plotting seasonal maps...")
        plot_seasonal_map(weights, nuts)
    except Exception as e:
        print(f"  Map skipped (geometry load failed): {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
