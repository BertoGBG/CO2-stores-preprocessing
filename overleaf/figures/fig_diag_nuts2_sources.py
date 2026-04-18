"""
Diagnostic figure: origin of NUTS2 CO₂ sequestration rates.

Two-panel choropleth showing, for each NUTS2 region, which step provided
the rate — for both the growth method (Pilli MAI) and the density method
(Avitabile AGB) as implemented in pypsa-eur.

Categories
----------
Growth (from Source column in afforestation_rates_nuts2_full.csv):
  Direct            — Pilli library contains a direct NUTS2 entry
  NUTS1 propagation — Pilli NUTS1 code propagated to child NUTS2
  NUTS0 propagation — Pilli national value propagated to all NUTS2
  Country mean      — mean of resolved NUTS2 in the same country
  Neighbour mean    — mean of NUTS2 regions within 100 km
  Western Balkans   — pseudo-NUTS2 (RS00/AL00/BA00/XK00)

Density (matches build_afforestation_potentials.py in pypsa-eur):
  NUTS0 (Avitabile) — country-level AGB density from Avitabile et al.
  Default (117 t/ha) — hardcoded fallback for countries absent from dataset
  Western Balkans   — pseudo-NUTS2 (RS00/AL00/BA00/XK00) → Default rate

Note: pypsa-eur applies NO spatial cascade for the density method.
      All NUTS2 within a country receive the same national AGB density.

Output
------
  fig_diag_nuts2_sources.png
"""

from pathlib import Path
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import warnings
warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE    = Path("/Users/albal/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/"
               "01_DTU_research/01_Projects/04_CO2-stores-preprocessing/"
               "CO2-stores-preprocessing/zenodo_aCDRs")
OUT_DIR = Path(__file__).parent

GROWTH_RATES_CSV = BASE / "outputs/afforestation/afforestation_rates_nuts2_full.csv"
DENSITY_XLS      = BASE / "outputs/afforestation/afforestation_nuts_biomass_densities.xlsx"
NUTS2_GJ         = BASE / "data/nuts/NUTS_RG_03M_2013_4326_LEVL_2.geojson"
PYPSA_GJ         = Path("/Users/albal/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/"
                         "01_DTU_research/01_Projects/01_pypsa-eur_AA/pypsa-eur/resources/"
                         "regions_onshore_base_s_90.geojson")

EXTRA_GEOM = {"RS00": "RS2 0", "AL00": "AL2 0", "BA00": "BA2 0", "XK00": "XK2 0"}
LIFETIME, CO2_PER_TONNE = 30, 1.83
DENSITY_FALLBACK_TDMHA = 117.0   # hardcoded in build_afforestation_potentials.py

# ── Growth source categories & colours ────────────────────────────────────────
GROWTH_CATS = [
    ("Direct",             "#1a7a4a"),   # dark green
    ("NUTS1 propagation",  "#74c476"),   # medium green
    ("NUTS0 propagation",  "#bae4b3"),   # light green
    ("Neighbour mean",     "#fdae6b"),   # orange
    ("Country mean",       "#e6550d"),   # dark orange
    ("Western Balkans",    "#756bb1"),   # purple
]
GROWTH_CAT_COLOR = dict(GROWTH_CATS)
GROWTH_CAT_ORDER = [c for c, _ in GROWTH_CATS]

# ── Density source categories & colours (NUTS0-only, as in pypsa-eur) ─────────
DENSITY_CATS = [
    ("NUTS0 (Avitabile)",  "#2171b5"),   # blue — country-level AGB data exists
    ("Default (117 t/ha)", "#f7fbff"),   # very light blue — hardcoded fallback
    ("Western Balkans",    "#756bb1"),   # purple
]
DENSITY_CAT_COLOR = dict(DENSITY_CATS)
DENSITY_CAT_ORDER = [c for c, _ in DENSITY_CATS]

# ── Load geometry ──────────────────────────────────────────────────────────────
print("Loading geometry...")
nuts_gdf = gpd.read_file(str(NUTS2_GJ))
if "LEVL_CODE" in nuts_gdf.columns:
    nuts_gdf = nuts_gdf[nuts_gdf["LEVL_CODE"] == 2]
nuts_gdf = nuts_gdf[["NUTS_ID", "geometry"]].set_index("NUTS_ID").to_crs("EPSG:4326")

pypsa_gdf = gpd.read_file(str(PYPSA_GJ)).set_index("name").to_crs("EPSG:4326")
extra_geoms = {k: pypsa_gdf.loc[v, "geometry"]
               for k, v in EXTRA_GEOM.items() if v in pypsa_gdf.index}
extra_gdf = gpd.GeoDataFrame(
    geometry=list(extra_geoms.values()),
    index=pd.Index(list(extra_geoms.keys()), name="NUTS_ID"),
    crs="EPSG:4326"
)
full_gdf = pd.concat([nuts_gdf[["geometry"]], extra_gdf])

# ══════════════════════════════════════════════════════════════════════════════
# 1. GROWTH SOURCE — from the Source column in the full rates CSV
# ══════════════════════════════════════════════════════════════════════════════
print("Classifying growth sources...")
gr = pd.read_csv(GROWTH_RATES_CSV)
gr.columns = gr.columns.str.strip()
gr = gr.set_index("NUTS2")

def classify_growth_source(src: str) -> str:
    if pd.isna(src):
        return "Unknown"
    s = str(src).strip()
    if s == "direct":
        return "Direct"
    if "NUTS1" in s:
        return "NUTS1 propagation"
    if "NUTS0" in s:
        return "NUTS0 propagation"
    if s.startswith("avg country"):
        return "Country mean"
    if s.startswith("avg neighbours") and any(
            x in s for x in ["HR, HU", "EL, MK", "HR, ME", "MK, RS"]):
        return "Western Balkans"
    if s.startswith("avg neighbours"):
        return "Neighbour mean"
    return "Unknown"

growth_src = gr["Source"].apply(classify_growth_source)
# Add Western Balkans pseudo-codes (not in the CSV index, added via Step 7)
for code in EXTRA_GEOM:
    if code not in growth_src.index:
        growth_src[code] = "Western Balkans"

# ══════════════════════════════════════════════════════════════════════════════
# 2. DENSITY SOURCE — NUTS0-only, matching build_afforestation_potentials.py
#    pypsa-eur uses country-level (NUTS0) AGB density only; if the country is
#    absent from the Avitabile dataset it falls back to 117 t/ha.
#    No spatial cascade (NUTS1/NUTS2/neighbours) is applied.
# ══════════════════════════════════════════════════════════════════════════════
print("Classifying density sources (NUTS0-only, as in pypsa-eur)...")
raw = pd.read_excel(DENSITY_XLS, sheet_name="BIOMASS 2020", header=1)
raw.columns = ["name","iso","nuts","forest_area_ha","faws_ha","fnaws_ha",
               "agb_t","biomass_density","baws_t","bnaws_t","_","legend"]
raw = raw[raw["iso"].notna() & (raw["name"] != "Name")].copy()
raw["biomass_density"] = pd.to_numeric(raw["biomass_density"], errors="coerce")
raw["nuts"] = raw["nuts"].astype(str).str.strip()

# Build set of country codes that have a valid NUTS0 entry
nuts0_countries = set(
    row["nuts"] for _, row in raw[raw["nuts"].str.len() == 2].iterrows()
    if pd.notna(row["biomass_density"])
)

density_src = pd.Series("", index=nuts_gdf.index, dtype=str)

for nid in nuts_gdf.index:
    cc = nid[:2]
    if cc in nuts0_countries:
        density_src[nid] = "NUTS0 (Avitabile)"
    else:
        density_src[nid] = "Default (117 t/ha)"

# Western Balkans pseudo-codes — use default fallback rate in pypsa-eur
for code in EXTRA_GEOM:
    density_src[code] = "Western Balkans"

n_avitabile = (density_src == "NUTS0 (Avitabile)").sum()
n_default   = (density_src == "Default (117 t/ha)").sum()
print(f"  Density: NUTS0 (Avitabile)={n_avitabile}, Default={n_default}, "
      f"Western Balkans={len(EXTRA_GEOM)}")

# ══════════════════════════════════════════════════════════════════════════════
# 3. FIGURE — side-by-side choropleth
# ══════════════════════════════════════════════════════════════════════════════
LAMBERT = ccrs.LambertConformal(central_longitude=10, central_latitude=52,
                                 standard_parallels=(35, 65))
EUROPE_EXTENT = [-12, 30, 34, 72]

fig, (ax_g, ax_d) = plt.subplots(1, 2, figsize=(16, 8),
                                   subplot_kw={"projection": LAMBERT})
fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.10, wspace=0.06)

def base_map(ax):
    ax.set_extent(EUROPE_EXTENT, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND,      facecolor="#f0f0f0", zorder=0)
    ax.add_feature(cfeature.OCEAN,     facecolor="#cde8f0", zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.35, zorder=3)
    ax.add_feature(cfeature.BORDERS,   linewidth=0.2, linestyle=":", zorder=3)

def draw_source_map(ax, source_series, cat_color, title):
    base_map(ax)
    for nid, row in full_gdf.iterrows():
        geom = row["geometry"]
        if geom is None or geom.is_empty:
            continue
        cat = source_series.get(nid, "")
        color = cat_color.get(cat, "#cccccc")
        ax.add_geometries([geom], crs=ccrs.PlateCarree(),
                          facecolor=color, edgecolor="white",
                          linewidth=0.15, zorder=1)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=6)

draw_source_map(ax_g, growth_src, GROWTH_CAT_COLOR,
                "Growth method (Pilli MAI)\nSource of NUTS2 sequestration rate")
draw_source_map(ax_d, density_src, DENSITY_CAT_COLOR,
                "Density method (Avitabile AGB)\nSource of NUTS2 sequestration rate")

# ── Legends — one per panel ────────────────────────────────────────────────────
growth_patches = [
    mpatches.Patch(facecolor=color, edgecolor="grey", linewidth=0.5, label=cat)
    for cat, color in GROWTH_CATS
]
density_patches = [
    mpatches.Patch(facecolor=color, edgecolor="grey", linewidth=0.5, label=cat)
    for cat, color in DENSITY_CATS
]

ax_g.legend(handles=growth_patches, loc="lower left", fontsize=8,
            frameon=True, framealpha=0.9, title="Growth source", title_fontsize=8)
ax_d.legend(handles=density_patches, loc="lower left", fontsize=8,
            frameon=True, framealpha=0.9, title="Density source", title_fontsize=8)

# ── Count annotations ──────────────────────────────────────────────────────────
def count_text(src_series, cat_order):
    counts = src_series.value_counts()
    return "  |  ".join(f"{cat}: {counts.get(cat, 0)}"
                        for cat in cat_order if counts.get(cat, 0) > 0)

fig.text(0.02, 0.93, count_text(growth_src,  GROWTH_CAT_ORDER),  fontsize=7.5, color="#333")
fig.text(0.52, 0.93, count_text(density_src, DENSITY_CAT_ORDER), fontsize=7.5, color="#333")

fig.suptitle(
    "Origin of NUTS2 CO₂ sequestration rates — growth vs. density method",
    fontsize=13, fontweight="bold", y=0.98
)

out = OUT_DIR / "fig_diag_nuts2_sources.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"\nSaved: {out}")
plt.close(fig)

# ── Print summary table ────────────────────────────────────────────────────────
print("\nGrowth filling source summary:")
print(growth_src.value_counts().reindex(GROWTH_CAT_ORDER, fill_value=0).to_string())
print("\nDensity filling source summary:")
print(density_src.value_counts().reindex(DENSITY_CAT_ORDER, fill_value=0).to_string())
