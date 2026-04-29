"""
Two-panel NUTS2 choropleth: afforestation capital cost (no discounting)

Both methods use the same annuity framework with r=0 (no discount):
  annuity(T*, r=0) = 1/T*

  (a) Growth method:
        capital_cost = I * (annuity(T*_r, r=0) + FOM/100) / MAI_CO2_r
                     = I * (1/T*_r + FOM/100) / MAI_CO2_r   [EUR/tCO2]
      T*_r from afforestation_rates_nuts2_full.csv column rotation_age_years
      MAI_CO2_r from column 'CO2 seq rate tCO2/(ha y)'
      Same formula as prepare_sector_network.py growth branch with r=0.

  (b) Density method:
        capital_cost = (I + FOM*L*I) / (rho_wood * co2_per_tonne)
      rho_wood [t/ha] from Avitabile et al. (2023) biomass data (NUTS2, falling back to NUTS0 / 117 t/ha)
      L = 30 yr fixed lifetime (from costs_2030.csv)

Cost parameters from technology-data costs_2030.csv:
  investment = 9743.09 EUR/ha,  FOM = 5 %/yr,  lifetime = 30 yr
"""

from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO       = Path(__file__).resolve().parents[2]
RATES_CSV  = REPO / "zenodo_aCDRs/outputs/afforestation/afforestation_rates_nuts2_full.csv"
BIOMASS_XL = REPO / "zenodo_aCDRs/outputs/afforestation/afforestation_nuts_biomass_densities.xlsx"
NUTS_GEO   = REPO / "zenodo_aCDRs/data/nuts/NUTS_RG_03M_2013_4326_LEVL_2.geojson"
NUTS21_GEO = REPO / "zenodo_aCDRs/data/nuts/NUTS_RG_01M_2021_4326_LEVL_2.geojson"
NET_GEO    = REPO / "Results_analysis/regions_onshore_base_s_90.geojson"
OUT_PDF    = Path(__file__).with_name("fig_affo_pypsa_comparison_capex.pdf")
OUT_PNG    = Path(__file__).with_name("fig_affo_pypsa_comparison_capex.png")

# ── Cost parameters (technology-data costs_2030.csv) ──────────────────────────
INVESTMENT    = 9743.0858   # EUR/ha
FOM_FRAC      = 0.05        # 5 %/year
LIFETIME      = 30.0        # years  (used only for density method maintenance)
CO2_PER_TONNE = 1.83        # tCO2 per tonne of dry biomass

# ── Annuity helper (mirrors pypsa-eur calculate_annuity) ──────────────────────
def annuity(T, r=0.0):
    """Capital recovery factor: r=0 → 1/T (no discounting)."""
    T = np.asarray(T, dtype=float)
    if r == 0.0:
        return 1.0 / T
    return r / (1.0 - (1.0 + r) ** (-T))

# ── Growth method (no discount, r=0): I*(annuity(T*,0)+FOM/100)/MAI ──────────
# Unified formula: setting r=0 gives annuity(T*,0)=1/T*, so
#   capital_cost = I*(1/T* + FOM/100)/MAI  [EUR/tCO2]
# This includes FOM as an annual maintenance fraction, consistent with
# the with-discount branch in prepare_sector_network.py at r=0.
DISCOUNT_RATE = 0.0   # zero discount: no time-value penalty on tied-up capital
rates = pd.read_csv(RATES_CSV).set_index("NUTS2")
crf = annuity(rates["rotation_age_years"].values, DISCOUNT_RATE)
rates["capex_growth"] = INVESTMENT * (crf + FOM_FRAC) / rates["CO2 seq rate tCO2/(ha y)"]

# ── Density method: capital_cost = (I + FOM*L*I) / (rho_wood * co2_per_tonne) ─
# Mirrors build_afforestation_potentials.py density mode:
#   1. NUTS2 row from Avitabile if available
#   2. NUTS0 (country-level) row from same Excel if not
#   3. European fallback of 117 t/ha if country not in dataset
BIOMASS_FALLBACK = 117.0   # t/ha  (same default as pypsa-eur)
maintenance = INVESTMENT * FOM_FRAC * LIFETIME   # EUR/ha total maintenance over lifetime

xl  = pd.ExcelFile(BIOMASS_XL)
bio_raw = xl.parse("BIOMASS 2020", header=2)
bio_raw.columns = [
    "Name", "ISO", "NUTS",
    "Forest_ha", "FAWS_ha", "FNAWS_ha",
    "AGB_tons", "AGB_tons_per_ha",
    "BAWS_tons", "BNAWS_tons", "_", "LEGEND",
]
bio_raw = bio_raw[bio_raw["NUTS"].notna()].copy()
bio_raw["NUTS"] = bio_raw["NUTS"].astype(str)

# NUTS0 (2-char) country densities
bio_nuts0 = (
    bio_raw[bio_raw["NUTS"].str.len() == 2]
    .set_index("NUTS")[["AGB_tons_per_ha"]].dropna()
)
# NUTS2 (4-char) densities where available
bio_nuts2 = (
    bio_raw[bio_raw["NUTS"].str.len() == 4]
    .set_index("NUTS")[["AGB_tons_per_ha"]].dropna()
)

# Build NUTS2 density series with country-level / European fallback
def lookup_density(nuts2_id):
    if nuts2_id in bio_nuts2.index:
        return bio_nuts2.loc[nuts2_id, "AGB_tons_per_ha"]
    country = nuts2_id[:2]
    if country in bio_nuts0.index:
        return bio_nuts0.loc[country, "AGB_tons_per_ha"]
    return BIOMASS_FALLBACK

bio = pd.DataFrame(
    {"AGB_tons_per_ha": {rid: lookup_density(rid) for rid in rates.index}}
)
bio["capex_density"] = (INVESTMENT + maintenance) / (bio["AGB_tons_per_ha"] * CO2_PER_TONNE)

# ── Load NUTS2 geometries ──────────────────────────────────────────────────────
# All regions come from the growth rates index (bio is already aligned to it)
nuts = gpd.read_file(NUTS_GEO)[["NUTS_ID", "geometry"]].set_index("NUTS_ID")
nuts = nuts[nuts.index.isin(rates.index)]

# Balkan supplement (same logic as fig_rotation_density_maps.py)
extra_rows = []
if NUTS21_GEO.exists():
    nuts21 = gpd.read_file(NUTS21_GEO)[["NUTS_ID", "geometry"]]
    for prefix, pseudo_id in [("AL", "AL00"), ("RS", "RS00")]:
        if pseudo_id not in nuts.index and pseudo_id in rates.index:
            sub = nuts21[nuts21["NUTS_ID"].str.startswith(prefix)]
            if not sub.empty:
                extra_rows.append({"NUTS_ID": pseudo_id,
                                   "geometry": sub.dissolve().geometry.iloc[0]})

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

gdf = nuts.join(rates[["capex_growth"]]).join(bio[["capex_density"]])
# Balkan pseudo-codes may be absent from bio if not in rates.index — fill via fallback
missing_dens = gdf["capex_density"].isna() & gdf["capex_growth"].notna()
for idx in gdf.index[missing_dens]:
    dens = lookup_density(idx)
    gdf.at[idx, "capex_density"] = (INVESTMENT + maintenance) / (dens * CO2_PER_TONNE)

# ── Shared colormap & norm ─────────────────────────────────────────────────────
CMAP = mpl.colormaps["YlOrRd"]
all_costs = pd.concat([gdf["capex_growth"], gdf["capex_density"]]).dropna()
norm = mpl.colors.Normalize(
    vmin=all_costs.min(),
    vmax=np.percentile(all_costs, 98),
)

# ── Area-weighted average (EPSG:3035 for correct areas) ───────────────────────
area_w = gdf.to_crs(3035).area

def wtd_avg(col):
    valid = gdf[col].notna()
    return (gdf.loc[valid, col] * area_w[valid]).sum() / area_w[valid].sum()

# ── Figure ─────────────────────────────────────────────────────────────────────
LAMBERT = ccrs.LambertConformal(
    central_longitude=10, central_latitude=52, standard_parallels=(35, 65)
)
EXTENT = [-12, 30, 34, 72]

fig, axes = plt.subplots(
    1, 2, figsize=(13, 5.5),
    subplot_kw={"projection": LAMBERT},
)
fig.subplots_adjust(wspace=0.04, left=0.01, right=0.88, top=0.88, bottom=0.03)


def base_map(ax):
    ax.set_extent(EXTENT, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND,      facecolor="#efefea", zorder=0)
    ax.add_feature(cfeature.OCEAN,     facecolor="#cde8f0", zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.35, zorder=2)
    ax.add_feature(cfeature.BORDERS,   linewidth=0.25, linestyle=":", zorder=2)


def draw_panel(ax, col, title, footer):
    base_map(ax)
    for nuts_id, row in gdf.iterrows():
        v = row[col]
        color = "#d0d0d0" if not np.isfinite(float(v)) else CMAP(norm(v))
        ax.add_geometries(
            [row.geometry], crs=ccrs.PlateCarree(),
            facecolor=color, edgecolor="white", linewidth=0.25, zorder=1,
        )
    ax.set_title(title, fontsize=10, fontweight="bold", pad=4)
    ax.text(
        0.03, 0.04, footer, transform=ax.transAxes,
        fontsize=7.5, va="bottom", ha="left",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.5", alpha=0.88),
    )


draw_panel(
    axes[0], "capex_growth",
    title=r"(a)  Growth method — $I\cdot(1/T^*\!+\!\mathrm{FOM})\,/\,\overline{\mathrm{MAI}}_{\mathrm{CO}_2}$  ($r=0$)",
    footer=f"Wtd avg: {wtd_avg('capex_growth'):.0f} EUR tCO₂⁻¹",
)
draw_panel(
    axes[1], "capex_density",
    title=r"(b)  Density method — $(I + \mathrm{FOM}{\cdot}L{\cdot}I)\,/\,(\rho_\mathrm{wood}\cdot c)$  ($r=0$)",
    footer=f"Wtd avg: {wtd_avg('capex_density'):.0f} EUR tCO₂⁻¹",
)

# Shared colorbar
cax = fig.add_axes([0.90, 0.10, 0.018, 0.72])
sm  = cm.ScalarMappable(norm=norm, cmap=CMAP)
sm.set_array([])
cb  = fig.colorbar(sm, cax=cax)
cb.set_label("Capital cost\n[EUR tCO₂⁻¹]", fontsize=9)
cb.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}"))

fig.suptitle(
    "Afforestation capital cost at NUTS2 — growth vs. density method (no discount, 2030 costs)",
    fontsize=11, y=0.97,
)

fig.savefig(OUT_PDF, dpi=300, bbox_inches="tight")
fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
print(f"Saved {OUT_PDF}")
print(f"Saved {OUT_PNG}")
plt.close(fig)
