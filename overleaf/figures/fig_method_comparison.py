"""
Comparison of growth vs density afforestation method — based purely on input data.
No CLC area calculations involved (those are identical for both methods).

Growth method:
  - Annual rate:   afforestation_rates_nuts2_full.csv  [tCO2/ha/yr]
  - Monthly rate:  annual_rate × monthly_weight  (from nuts2_monthly_weights.csv)

Density method:
  - Annual rate:   Avitabile AGB / lifetime × co2_per_tonne × max_land_usage  [tCO2/ha/yr]
                   Cascade (same steps as growth): NUTS2 → NUTS1 → NUTS0 → neighbours → country mean
  - Monthly rate:  annual_rate / 12  (uniform — no seasonal variation)

Western Balkans (RS/AL/BA/XK): not in Eurostat NUTS2 GeoJSON.
  Added as pseudo-codes (RS00, AL00, BA00, XK00) via neighbour-mean fallback.
  Drawn using PyPSA-EUR node shapes as proxy geometry.

Outputs:
  fig_method_comparison_maps.png       — 3 maps: growth | density | difference
  fig_method_comparison_seasonal.png   — seasonal profiles + NUTS2 scatter
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
GROWTH_WGTS_CSV  = BASE / "outputs/afforestation/nuts2_monthly_weights.csv"
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
# 3. DENSITY METHOD (cascade to full NUTS2 coverage)
# ══════════════════════════════════════════════════════════════════════════════
print("Building density cascade...")

raw = pd.read_excel(DENSITY_XLS, sheet_name="BIOMASS 2020", header=1)
raw.columns = ["name","iso","nuts","forest_area_ha","faws_ha","fnaws_ha",
               "agb_t","biomass_density","baws_t","bnaws_t","_","legend"]
raw = raw[raw["iso"].notna() & (raw["name"] != "Name")].copy()
raw["biomass_density"] = pd.to_numeric(raw["biomass_density"], errors="coerce")
raw["rate"] = raw["biomass_density"] / LIFETIME * CO2_PER_TONNE * MAX_LAND
raw["nuts"] = raw["nuts"].astype(str).str.strip()

density_annual = pd.Series(np.nan, index=nuts_gdf.index, dtype=float)

# Step 1: direct NUTS2 (len=4)
for _, row in raw[raw["nuts"].str.len() == 4].iterrows():
    nid = row["nuts"]
    if nid in density_annual.index and pd.isna(density_annual[nid]):
        density_annual[nid] = row["rate"]

# Step 2: NUTS1 propagation (len=3)
for _, row in raw[raw["nuts"].str.len() == 3].iterrows():
    parent = row["nuts"]
    for child in [n for n in density_annual.index if n.startswith(parent)
                  and pd.isna(density_annual[n])]:
        density_annual[child] = row["rate"]

# Step 3: NUTS0 propagation (len=2)
for _, row in raw[raw["nuts"].str.len() == 2].iterrows():
    prefix = row["nuts"]
    for child in [n for n in density_annual.index if n[:2] == prefix
                  and pd.isna(density_annual[n])]:
        density_annual[child] = row["rate"]

n_miss = int(density_annual.isna().sum())
print(f"  After direct/NUTS1/NUTS0: {len(density_annual)-n_miss} filled, {n_miss} missing")

# Step 4: distance neighbour (100 km, iterated)
if density_annual.isna().any():
    gdf3035 = nuts_gdf.to_crs(3035)
    nbrs = {rid: [n for n in gdf3035[
                gdf3035.geometry.intersects(geom.buffer(100_000))].index if n != rid]
            if (geom and not geom.is_empty) else []
            for rid, geom in gdf3035.geometry.items()}
    changed = True
    while changed:
        changed = False
        for rid in list(density_annual.index[density_annual.isna()]):
            valid = [n for n in nbrs.get(rid, []) if pd.notna(density_annual[n])]
            if valid:
                density_annual[rid] = density_annual[valid].mean()
                changed = True
    n_miss2 = int(density_annual.isna().sum())
    print(f"  After neighbours: {n_miss-n_miss2} more filled ({n_miss2} left)")

# Step 5: country mean
if density_annual.isna().any():
    cmean = density_annual.groupby(density_annual.index.str[:2]).mean()
    for rid in density_annual.index[density_annual.isna()]:
        key = rid[:2]
        if key in cmean.index and pd.notna(cmean[key]):
            density_annual[rid] = cmean[key]

# Step 7: Western Balkans
density_annual = add_western_balkans_rates(density_annual, EXTRA_PYPSA)

# Monthly rates: uniform 1/12
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
           "Density method — annual rate\n(Avitabile AGB, NUTS2 cascade)", fig1, "tCO₂/ha/yr")
draw_choro(axes[2], diff_annual,    cm.get_cmap("RdBu"), norm_diff,
           "Difference: growth − density\n[tCO₂/ha/yr]", fig1,
           "tCO₂/ha/yr  (red = growth higher, blue = density higher)")

fig1.suptitle(
    "Annual CO₂ sequestration rate — growth vs. density method (input data only, before CLC areas)",
    fontsize=11, y=0.97
)
for ext in ("png"):
    fig1.savefig(OUT_DIR / f"fig_method_comparison_maps.{ext}", dpi=200, bbox_inches="tight")
print("  Saved fig_method_comparison_maps")
plt.close(fig1)

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Seasonal profiles + NUTS2 scatter
# ══════════════════════════════════════════════════════════════════════════════
print("Figure 2: seasonal profiles + scatter...")

fig2 = plt.figure(figsize=(18, 10))
fig2.subplots_adjust(left=0.06, right=0.98, top=0.91, bottom=0.09,
                     wspace=0.28, hspace=0.38)

ax_sc  = fig2.add_subplot(2, 3, (1, 2))  # top-left wide: NUTS2 scatter
ax_sea = fig2.add_subplot(2, 3, 3)       # top-right: Europe-avg seasonal
ax_n   = fig2.add_subplot(2, 3, 4)       # bottom-left: Northern Europe
ax_c   = fig2.add_subplot(2, 3, 5)       # bottom-centre: Central Europe
ax_s   = fig2.add_subplot(2, 3, 6)       # bottom-right: Southern Europe

# ── Scatter: NUTS2 annual growth vs density ───────────────────────────────────
both = pd.DataFrame({"growth": growth_annual, "density": density_annual}).dropna()
diff_col = both["growth"] - both["density"]
sc = ax_sc.scatter(both["growth"], both["density"],
                   c=diff_col, cmap="RdBu", norm=norm_diff,
                   s=14, alpha=0.72, zorder=3)
fig2.colorbar(sc, ax=ax_sc, label="growth − density [tCO₂/ha/yr]", pad=0.01, shrink=0.85)
lim = (0.5, 11.5)
ax_sc.plot(lim, lim, "k--", lw=0.8, alpha=0.5, label="1 : 1")
ax_sc.set_xlim(lim); ax_sc.set_ylim(lim)
ax_sc.set_xlabel("Growth method [tCO₂/ha/yr]", fontsize=9)
ax_sc.set_ylabel("Density method [tCO₂/ha/yr]", fontsize=9)
ax_sc.set_title("NUTS2-level annual rates (n={})".format(len(both)),
                fontsize=10, fontweight="bold")
ax_sc.legend(fontsize=8); ax_sc.grid(True, alpha=0.3)

corr = both["growth"].corr(both["density"])
rmse = np.sqrt(((both["growth"] - both["density"])**2).mean())
bias = (both["growth"] - both["density"]).mean()
ax_sc.text(0.03, 0.97,
           f"r = {corr:.2f}   RMSE = {rmse:.2f}   Bias = {bias:+.2f} tCO₂/ha/yr",
           transform=ax_sc.transAxes, ha="left", va="top", fontsize=8.5,
           bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.5", alpha=0.88))

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
for ext in ("png"):
    fig2.savefig(OUT_DIR / f"fig_method_comparison_seasonal.{ext}", dpi=200, bbox_inches="tight")
print("  Saved fig_method_comparison_seasonal")
plt.close(fig2)

print("\nDone.")
