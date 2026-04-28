"""
Two-panel NUTS2 map:
  Left  — rotation age T* (years), i.e. the MAI year
  Right — forest carbon density at T* (tCO2/ha)

Data: afforestation_rates_nuts2_full.csv
NUTS: NUTS_RG_03M_2013_4326_LEVL_2.geojson
      + NUTS_RG_01M_2021_4326_LEVL_2.geojson (AL, RS dissolve)
      + Results_analysis/regions_onshore_base_s_90.geojson (BA, XK)
"""

from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parents[2]
RATES_CSV  = REPO / "zenodo_aCDRs/outputs/afforestation/afforestation_rates_nuts2_full.csv"
NUTS_GEO   = REPO / "zenodo_aCDRs/data/nuts/NUTS_RG_03M_2013_4326_LEVL_2.geojson"
NUTS21_GEO = REPO / "zenodo_aCDRs/data/nuts/NUTS_RG_01M_2021_4326_LEVL_2.geojson"
NET_GEO    = REPO / "Results_analysis/regions_onshore_base_s_90.geojson"
OUT_PDF    = Path(__file__).with_suffix(".pdf")
OUT_PNG    = Path(__file__).with_suffix(".png")

EUROPE_XLIM = (-25, 44)
EUROPE_YLIM = (34, 72)

# ── Load data ──────────────────────────────────────────────────────────────────
rates = pd.read_csv(RATES_CSV).set_index("NUTS2")
nuts  = gpd.read_file(NUTS_GEO)[["NUTS_ID", "geometry"]].set_index("NUTS_ID")
nuts  = nuts[nuts.index.isin(rates.index)]   # keep only data-covered regions

# ── Supplement with Western Balkan pseudo-NUTS2 geometries ────────────────────
# RS00 and AL00 are dissolved from NUTS 2021 sub-regions (AL01-03, RS11-22).
# BA00 and XK00 are taken from the PyPSA-Eur network region file (whole-country
# clusters) since Bosnia and Kosovo have no Eurostat NUTS classification.
extra_rows = []

if NUTS21_GEO.exists():
    nuts21 = gpd.read_file(NUTS21_GEO)[["NUTS_ID", "geometry"]]
    for prefix, pseudo_id in [("AL", "AL00"), ("RS", "RS00")]:
        if pseudo_id not in nuts.index and pseudo_id in rates.index:
            sub = nuts21[nuts21["NUTS_ID"].str.startswith(prefix)]
            if not sub.empty:
                dissolved = sub.dissolve().geometry.iloc[0]
                extra_rows.append({"NUTS_ID": pseudo_id, "geometry": dissolved})

if NET_GEO.exists():
    net = gpd.read_file(NET_GEO)[["name", "geometry"]]
    for prefix, pseudo_id in [("BA", "BA00"), ("XK", "XK00")]:
        if pseudo_id not in nuts.index and pseudo_id in rates.index:
            row = net[net["name"].str.startswith(prefix)]
            if not row.empty:
                extra_rows.append({"NUTS_ID": pseudo_id,
                                   "geometry": row.geometry.iloc[0]})

if extra_rows:
    extra_gdf = gpd.GeoDataFrame(extra_rows, crs=nuts.crs).set_index("NUTS_ID")
    nuts = pd.concat([nuts, extra_gdf])

gdf = nuts.join(rates[["rotation_age_years", "density tCO2/ha"]])

# ── Colour maps ────────────────────────────────────────────────────────────────
# rotation age: YlOrRd — short (yellow) to long (red)
# density: YlGn — low (yellow) to high (green)
ROT_CMAP  = "YlOrRd"
DENS_CMAP = "YlGn"

# cap density colour scale at 500 tCO2/ha (Portugal outlier at 437 looks fine,
# but if future data has higher values the cmap would compress everything else)
DENS_VMAX = 500

# ── Figure ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(
    1, 2, figsize=(13, 6.5),
    gridspec_kw={"wspace": 0.02},
)

def plot_panel(ax, col, cmap, vmin, vmax, label, fmt="{:.0f}"):
    gdf.plot(
        column=col,
        ax=ax,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        linewidth=0.25,
        edgecolor="white",
        missing_kwds={"color": "lightgrey"},
    )
    # grey background for sea / non-covered
    ax.set_facecolor("#d0d0d0")
    ax.set_xlim(*EUROPE_XLIM)
    ax.set_ylim(*EUROPE_YLIM)
    ax.axis("off")

    # colourbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical",
                        fraction=0.03, pad=0.01, shrink=0.75)
    cbar.set_label(label, fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    # annotate regional stats
    vals = gdf[col].dropna()
    stats = (
        f"median {vals.median():.0f}   "
        f"IQR [{vals.quantile(0.25):.0f}–{vals.quantile(0.75):.0f}]"
    )
    ax.text(0.02, 0.02, stats, transform=ax.transAxes,
            fontsize=7.5, color="0.3",
            va="bottom", ha="left",
            bbox=dict(fc="white", alpha=0.6, ec="none", pad=2))

# Left: rotation age — use data-driven range capped at MAX_ROTATION_AGE (100 yr)
rot_min = 20
rot_max = int(gdf["rotation_age_years"].dropna().max()) + 5   # e.g. ~70
plot_panel(
    axes[0],
    col="rotation_age_years",
    cmap=ROT_CMAP,
    vmin=rot_min,
    vmax=rot_max,
    label="Rotation age $T^*$ (years)",
)
axes[0].set_title(
    r"(a) Rotation age $T^*$ at MAI optimum (cap 100 yr)",
    fontsize=10, pad=6,
)

# Right: density
dens_min = 0
plot_panel(
    axes[1],
    col="density tCO2/ha",
    cmap=DENS_CMAP,
    vmin=dens_min,
    vmax=DENS_VMAX,
    label=r"Carbon density $\rho^*$ (tCO$_2$ ha$^{-1}$)",
)
n_clipped_d = int((gdf["density tCO2/ha"] > DENS_VMAX).sum())
if n_clipped_d:
    axes[1].text(0.02, 0.06,
                 f"({n_clipped_d} region(s) >{DENS_VMAX} tCO2/ha, shown at max colour)",
                 transform=axes[1].transAxes, fontsize=6.5, color="0.4",
                 va="bottom", ha="left")
axes[1].set_title(
    r"(b) Carbon density at MAI: $\rho^* = \overline{\mathrm{MAI}}_{\mathrm{CO}_2} \times T^*$",
    fontsize=10, pad=6,
)

fig.savefig(OUT_PDF, dpi=300, bbox_inches="tight")
fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
print(f"Saved {OUT_PDF}")
print(f"Saved {OUT_PNG}")
plt.close(fig)
