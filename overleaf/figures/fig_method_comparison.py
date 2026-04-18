"""
Comparison of growth vs density afforestation method — based purely on input data.
No CLC area calculations involved (those are identical for both methods).

Growth method:
  - Annual rate:   afforestation_rates_nuts2_full.csv  [tCO2/ha/yr]
  - Monthly rate:  annual_rate × monthly_weight  (from afforestation_nuts2_monthly_weights.csv)

Density method (matches build_afforestation_potentials.py in pypsa-eur):
  - Annual rate:   Avitabile AGB (NUTS0 country-level only) / lifetime × co2_per_tonne
                   Countries absent from Avitabile → hardcoded fallback 117 t_DM/ha
                   No sub-national cascade applied. f_land excluded for comparison.
  - Monthly rate:  annual_rate / 12  (uniform — no seasonal variation)

Western Balkans (RS/AL/BA/XK): not in Eurostat NUTS2 GeoJSON.
  Added as pseudo-codes (RS00, AL00, BA00, XK00) via neighbour-mean fallback.
  Drawn using PyPSA-EUR node shapes as proxy geometry.

Outputs:
  fig_method_comparison_maps.png       — 3 maps: growth | density | difference
  fig_method_comparison_seasonal.png   — 2×2 seasonal profiles by macro-region
"""

from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import warnings
warnings.filterwarnings("ignore")

# ── Config ─────────────────────────────────────────────────────────────────────
LIFETIME      = 30      # years (from pypsa config)
CO2_PER_TONNE = 1.83    # tCO2 per t dry biomass
MAX_LAND      = 0.6     # land usage fraction

MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]

# Western Balkans neighbour assignments (same as Step 7 in pilli script)
EXTRA_PYPSA = {
    "RS00": ["HR", "HU", "RO", "BG"],
    "AL00": ["EL", "MK"],
    "BA00": ["HR", "ME"],
    "XK00": ["MK", "RS00"],
}
# PyPSA node name → pseudo NUTS2 code (for geometry lookup)
EXTRA_GEOM = {"RS00": "RS2 0", "AL00": "AL2 0", "BA00": "BA2 0", "XK00": "XK2 0"}

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE    = Path("/Users/albal/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/"
               "01_DTU_research/01_Projects/04_CO2-stores-preprocessing/"
               "CO2-stores-preprocessing/zenodo_aCDRs")
OUT_DIR = Path(__file__).parent

GROWTH_RATES_CSV = BASE / "outputs/afforestation/afforestation_rates_nuts2_full.csv"
GROWTH_WGTS_CSV  = BASE / "outputs/afforestation/afforestation_nuts2_monthly_weights.csv"
DENSITY_XLS      = BASE / "outputs/afforestation/afforestation_nuts_biomass_densities.xlsx"
NUTS2_GJ         = BASE / "data/nuts/NUTS_RG_03M_2013_4326_LEVL_2.geojson"
PYPSA_GJ         = Path("/Users/albal/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/"
                         "01_DTU_research/01_Projects/01_pypsa-eur_AA/pypsa-eur/resources/"
                         "regions_onshore_base_s_90.geojson")

# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD GEOMETRY
# ══════════════════════════════════════════════════════════════════════════════
print("Loading geometry...")
nuts_gdf = gpd.read_file(str(NUTS2_GJ))
if "LEVL_CODE" in nuts_gdf.columns:
    nuts_gdf = nuts_gdf[nuts_gdf["LEVL_CODE"] == 2]
nuts_gdf = nuts_gdf[["NUTS_ID","geometry"]].set_index("NUTS_ID").to_crs("EPSG:4326")

pypsa_gdf = gpd.read_file(str(PYPSA_GJ)).set_index("name").to_crs("EPSG:4326")
extra_gdf = gpd.GeoDataFrame(
    geometry=[pypsa_gdf.loc[v, "geometry"] for v in EXTRA_GEOM.values()
              if v in pypsa_gdf.index],
    index=pd.Index([k for k, v in EXTRA_GEOM.items() if v in pypsa_gdf.index],
                   name="NUTS_ID"),
    crs="EPSG:4326"
)
full_gdf = pd.concat([nuts_gdf[["geometry"]], extra_gdf])  # 324 rows

# ══════════════════════════════════════════════════════════════════════════════
# 2. GROWTH METHOD
# ══════════════════════════════════════════════════════════════════════════════
print("Loading growth rates and monthly weights...")

# Annual rates
gr_raw = pd.read_csv(GROWTH_RATES_CSV, index_col="NUTS2")
gr_raw.columns = gr_raw.columns.str.strip()
growth_annual = gr_raw["CO2 seq rate tCO2/(ha y)"].copy()  # 320 entries

# Monthly weights
mw = pd.read_csv(GROWTH_WGTS_CSV, index_col="NUTS_ID")     # 320 × 12

# Step 7: append RS00/AL00/BA00/XK00 via neighbour mean (only if not already in CSV)
def add_western_balkans_rates(series, extra_dict):
    """Append pseudo-NUTS2 entries using neighbour country mean (skips existing entries)."""
    extra = {}
    for code, neighbours in extra_dict.items():
        if code in series.index:
            extra[code] = series[code]  # already present — carry forward for XK00 chaining
            continue
        vals = []
        for nb in neighbours:
            if nb.endswith("00"):
                if nb in extra: vals.append(extra[nb])
            else:
                nbv = series[series.index.str[:2] == nb].dropna()
                if not nbv.empty: vals.append(nbv.mean())
        if vals:
            extra[code] = float(np.mean(vals))
    new_codes = [c for c in extra if c not in series.index]
    if new_codes:
        return pd.concat([series, pd.Series({c: extra[c] for c in new_codes}, dtype=float)])
    return series

def add_western_balkans_weights(wdf, extra_dict):
    """Append pseudo-NUTS2 monthly weights using neighbour country mean (skips existing)."""
    extra_rows = {}
    for code, neighbours in extra_dict.items():
        if code in wdf.index:
            extra_rows[code] = wdf.loc[code]  # carry forward for XK00 chaining
            continue
        profiles = []
        for nb in neighbours:
            if nb.endswith("00"):
                if nb in extra_rows: profiles.append(extra_rows[nb])
            else:
                nbv = wdf[wdf.index.str[:2] == nb]
                if not nbv.empty: profiles.append(nbv.mean())
        if profiles:
            mean_profile = pd.concat(profiles, axis=1).mean(axis=1)
            extra_rows[code] = mean_profile / mean_profile.sum()
    new_codes = [c for c in extra_rows if c not in wdf.index]
    if new_codes:
        wdf = pd.concat([wdf, pd.DataFrame({c: extra_rows[c] for c in new_codes}).T])
    return wdf

growth_annual = add_western_balkans_rates(growth_annual, EXTRA_PYPSA)
mw            = add_western_balkans_weights(mw, EXTRA_PYPSA)

# Monthly rates: weight × annual rate
growth_monthly = mw.multiply(growth_annual.reindex(mw.index), axis=0)  # tCO2/ha/month

print(f"  Growth: {growth_annual.notna().sum()} NUTS2 regions with annual rate")
print(f"  Monthly weights: {len(mw)} regions, row-sum ≈ {mw.sum(axis=1).mean():.3f}")

# ══════════════════════════════════════════════════════════════════════════════
# 3. DENSITY METHOD — matching build_afforestation_potentials.py exactly
# ══════════════════════════════════════════════════════════════════════════════
# PyPSA-Eur uses NUTS0 (country-level) data only from Avitabile.
# Countries absent from the dataset receive a hardcoded fallback of 117 t/ha
# (the European average used in build_afforestation_potentials.py).
# No NUTS1/NUTS2 cascade or spatial neighbour filling is applied.
# Rates are intrinsic (without f_land = 0.6) for a like-for-like comparison.
print("Building density rates (NUTS0-only, matching pypsa-eur logic)...")

DENSITY_FALLBACK_TDMHA = 117.0   # t_DM/ha — default in build_afforestation_potentials.py
DENSITY_FALLBACK_RATE  = DENSITY_FALLBACK_TDMHA / LIFETIME * CO2_PER_TONNE

raw = pd.read_excel(DENSITY_XLS, sheet_name="BIOMASS 2020", header=1)
raw.columns = ["name","iso","nuts","forest_area_ha","faws_ha","fnaws_ha",
               "agb_t","biomass_density","baws_t","bnaws_t","_","legend"]
raw = raw[raw["iso"].notna() & (raw["name"] != "Name")].copy()
raw["biomass_density"] = pd.to_numeric(raw["biomass_density"], errors="coerce")
raw["nuts"] = raw["nuts"].astype(str).str.strip()

# Build NUTS0 lookup table (len==2 codes only, same filter as pypsa-eur)
nuts0_rate = {}
for _, row in raw[raw["nuts"].str.len() == 2].iterrows():
    if pd.notna(row["biomass_density"]):
        nuts0_rate[row["nuts"]] = row["biomass_density"] / LIFETIME * CO2_PER_TONNE

# Apply to all NUTS2 regions: NUTS0 lookup → fallback 117 t/ha
density_annual = pd.Series(np.nan, index=nuts_gdf.index, dtype=float)
for nid in density_annual.index:
    cc = nid[:2]
    density_annual[nid] = nuts0_rate.get(cc, DENSITY_FALLBACK_RATE)

n_avit  = sum(1 for nid in density_annual.index if nid[:2] in nuts0_rate)
n_fall  = len(density_annual) - n_avit
print(f"  NUTS0 (Avitabile): {n_avit} regions  |  Fallback (117 t/ha): {n_fall} regions")

# Western Balkans pseudo-codes — also not in Avitabile → fallback rate
for code in EXTRA_GEOM:
    density_annual[code] = DENSITY_FALLBACK_RATE

# Monthly rates: uniform 1/12 (no seasonal variation in density method)
density_monthly = pd.DataFrame(
    {m: density_annual / 12 for m in MONTH_NAMES},
    index=density_annual.index
)

print(f"  Density: {density_annual.notna().sum()} NUTS2 regions covered")

# ══════════════════════════════════════════════════════════════════════════════
# 4. DERIVED QUANTITIES
# ══════════════════════════════════════════════════════════════════════════════
# Annual difference (growth − density), aligned on common index
common_idx = growth_annual.index.intersection(density_annual.index)
diff_annual = growth_annual.reindex(common_idx) - density_annual.reindex(common_idx)

# ══════════════════════════════════════════════════════════════════════════════
# 5. COLORMAPS & HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def trunc_cmap(base, lo=0.12, hi=1.0, n=256):
    c = cm.get_cmap(base, n)(np.linspace(lo, hi, n))
    return LinearSegmentedColormap.from_list(f"tr_{base}", c)

CMAP_G  = trunc_cmap("YlGn")
CMAP_D  = trunc_cmap("YlGn") # trunc_cmap("YlOrBr")
LAMBERT = ccrs.LambertConformal(central_longitude=10, central_latitude=52,
                                 standard_parallels=(35, 65))
EUROPE_EXTENT = [-12, 30, 34, 72]
VMIN, VMAX = 1.0, 10.0
norm_rate = mcolors.Normalize(vmin=VMIN, vmax=VMAX)

abs_max  = max(diff_annual.dropna().abs().quantile(0.97), 2.0)
norm_diff = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)

def draw_choro(ax, rates, cmap, norm, title, fig, cb_label):
    ax.set_extent(EUROPE_EXTENT, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND,      facecolor="#efefea", zorder=0)
    ax.add_feature(cfeature.OCEAN,     facecolor="#cde8f0", zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.3,        zorder=3)
    ax.add_feature(cfeature.BORDERS,   linewidth=0.2, linestyle=":", zorder=3)
    for nid, row in full_gdf.iterrows():
        geom = row["geometry"]
        if geom is None or geom.is_empty:
            continue
        val = rates.get(nid, np.nan)
        color = "#cccccc" if not np.isfinite(val) else cmap(norm(val))
        ax.add_geometries([geom], crs=ccrs.PlateCarree(),
                          facecolor=color, edgecolor="white",
                          linewidth=0.15, zorder=1)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=4)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, orientation="horizontal", pad=0.03, fraction=0.045)
    cb.set_label(cb_label, fontsize=8)

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Annual rate maps: growth | density | difference
# ══════════════════════════════════════════════════════════════════════════════
print("\nFigure 1: annual rate maps...")
fig1, axes = plt.subplots(1, 3, figsize=(21, 8),
                           subplot_kw={"projection": LAMBERT})
fig1.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.08, wspace=0.06)

draw_choro(axes[0], growth_annual,  CMAP_G, norm_rate,
           "Growth method — annual rate\n(Pilli MAI, NUTS2)", fig1, "tCO₂/ha/yr")
draw_choro(axes[1], density_annual, CMAP_D, norm_rate,
           "Density method — annual rate\n(Avitabile AGB, NUTS2 cascade; excl. $f_\\mathrm{land}$)", fig1, "tCO₂/ha/yr")
draw_choro(axes[2], diff_annual,    cm.get_cmap("RdBu"), norm_diff,
           "Difference: growth − density\n[tCO₂/ha/yr]", fig1,
           "tCO₂/ha/yr  (red = growth higher, blue = density higher)")

fig1.suptitle(
    "Annual CO₂ sequestration rate — growth vs. density method (input data only, before CLC areas)\n"
    "Density rates shown without land-utilization scaling coefficient "
    r"($f_\mathrm{land} = 0.6$, applied identically to both methods in PyPSA-Eur)",
    fontsize=10, y=0.97
)
for ext in ("png",):
    fig1.savefig(OUT_DIR / f"fig_method_comparison_maps.{ext}", dpi=200, bbox_inches="tight")
print("  Saved fig_method_comparison_maps")
plt.close(fig1)

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Seasonal profiles (2×2)
# ══════════════════════════════════════════════════════════════════════════════
print("Figure 2: seasonal profiles...")

fig2, axes2 = plt.subplots(2, 2, figsize=(14, 9))
fig2.subplots_adjust(left=0.07, right=0.97, top=0.91, bottom=0.09,
                     wspace=0.30, hspace=0.42)

ax_sea = axes2[0, 0]   # top-left:     Europe-wide average
ax_n   = axes2[0, 1]   # top-right:    Northern Europe
ax_c   = axes2[1, 0]   # bottom-left:  Central Europe
ax_s   = axes2[1, 1]   # bottom-right: Southern Europe

# ── Seasonal profile helper ────────────────────────────────────────────────────
UNIFORM = np.array([1.0/12]*12)

def seasonal_ax(ax, country_codes, label, growth_wdf, density_annual_s):
    """
    Plot monthly sequestration rate for growth (seasonal) vs density (uniform).
    Both shown as tCO2/ha/month.
    country_codes: list of 2-char country prefixes to include.
    """
    mask_g = growth_wdf.index.str[:2].isin(country_codes)
    mask_d = density_annual_s.index.str[:2].isin(country_codes)

    if mask_g.sum() == 0:
        ax.set_visible(False); return

    # Growth monthly rates in common units: weight × annual_rate
    g_annual_sub = growth_annual.reindex(growth_wdf.index)[mask_g]
    g_monthly    = growth_wdf[mask_g].multiply(g_annual_sub, axis=0)  # tCO2/ha/month
    g_mean = g_monthly.mean()     # mean across NUTS2 regions
    g_q25  = g_monthly.quantile(0.25)
    g_q75  = g_monthly.quantile(0.75)

    # Density monthly rate: annual/12 for each NUTS2, then mean
    d_annual_sub  = density_annual_s[mask_d]
    d_mean_annual = d_annual_sub.mean()
    d_monthly_val = d_mean_annual / 12   # scalar

    x = np.arange(12)
    ax.fill_between(x, g_q25.values, g_q75.values,
                    color="#2c7bb6", alpha=0.20, label="Growth IQR")
    ax.plot(x, g_mean.values, "o-",
            color="#2c7bb6", lw=1.8, ms=5, label="Growth (mean)")
    ax.axhline(d_monthly_val, color="#d7191c", lw=1.8,
               linestyle="--", label=f"Density (uniform, {d_monthly_val:.3f})")

    ax.set_xticks(x); ax.set_xticklabels(MONTH_NAMES, fontsize=7, rotation=45)
    ax.set_ylabel("tCO₂/ha/month", fontsize=8)
    ax.set_title(label, fontsize=9, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="upper right")

    # Annotate annual totals
    ax.text(0.03, 0.97,
            f"Growth: {g_annual_sub.mean():.2f} tCO₂/ha/yr\n"
            f"Density: {d_mean_annual:.2f} tCO₂/ha/yr",
            transform=ax.transAxes, ha="left", va="top", fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.5", alpha=0.85))

# Europe-wide average
seasonal_ax(ax_sea, list(mw.index.str[:2].unique()),
            "Europe-wide average", mw, density_annual)

# Regional subsets
seasonal_ax(ax_n,
            ["FI","SE","NO","EE","LV","LT","DK"],
            "Northern Europe\n(FI, SE, NO, EE, LV, LT, DK)", mw, density_annual)
seasonal_ax(ax_c,
            ["DE","PL","CZ","AT","CH","HU","SK"],
            "Central Europe\n(DE, PL, CZ, AT, CH, HU, SK)", mw, density_annual)
seasonal_ax(ax_s,
            ["IT","ES","PT","EL","HR","SI","ME","AL","BA"],
            "Southern Europe\n(IT, ES, PT, EL, HR, SI, ME, AL, BA)", mw, density_annual)

fig2.suptitle(
    "Afforestation sequestration rates: growth (seasonal, Pilli) vs. density (uniform, Avitabile)\n"
    "Input data only — CLC areas identical for both methods",
    fontsize=11, y=0.97
)
for ext in ("png",):
    fig2.savefig(OUT_DIR / f"fig_method_comparison_seasonal.{ext}", dpi=200, bbox_inches="tight")
print("  Saved fig_method_comparison_seasonal")
plt.close(fig2)

print("\nDone.")
