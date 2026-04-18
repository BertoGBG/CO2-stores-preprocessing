"""
Comparison of afforestation potentials and seasonal profiles:
  - Growth method  (Pilli MAI-based, seasonal)
  - Density method (biomass-density-based, linear)

Outputs (saved to same directory as this script):
  fig1_afforestation_potential_map.pdf/.png
  fig2_monthly_cumulative_growth.pdf/.png
  fig3_monthly_cumulative_density.pdf/.png
  fig4_nuts0_timeseries.pdf/.png
"""

from pathlib import Path
import re
import numpy as np
import pypsa
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import LinearSegmentedColormap
from shapely.geometry import Point
import warnings
warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
GROWTH_NC  = Path("/Users/albal/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/01_DTU_research/01_Projects/01_pypsa-eur_AA/pypsa-eur/results/CDRs_2050/CDRs_2050/networks/base_s_90__3h_2050.nc")
DENSITY_NC = Path("/Users/albal/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/01_DTU_research/01_Projects/04_CO2-stores-preprocessing/CO2-stores-preprocessing/Results_analysis/results_aCDRs_RM/base_s_90__Co2L0-3H-T-H-B-I-A-solar+p3-dist1_2050.nc")
GEOJSON    = Path("/Users/albal/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/01_DTU_research/01_Projects/01_pypsa-eur_AA/pypsa-eur/resources/regions_onshore_base_s_90.geojson")
OUT_DIR    = Path(__file__).parent
OUT_DIR.mkdir(parents=True, exist_ok=True)

MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]

LAMBERT = ccrs.LambertConformal(
    central_longitude=10, central_latitude=52,
    standard_parallels=(35, 65)
)
EUROPE_EXTENT = [-12, 30, 34, 72]   # lon_min, lon_max, lat_min, lat_max

# ── Load GeoJSON and compute node areas ───────────────────────────────────────
print("Loading GeoJSON...")
gdf = gpd.read_file(GEOJSON).set_index("name")      # index = node name (growth naming)
gdf_proj = gdf.to_crs("EPSG:3035")
gdf_proj["area_km2"] = gdf_proj.geometry.area / 1e6
gdf["area_km2"] = gdf_proj["area_km2"]

# ── Load networks ─────────────────────────────────────────────────────────────
print("Loading networks...")
ng = pypsa.Network(str(GROWTH_NC))
nd = pypsa.Network(str(DENSITY_NC))

# ── Build node→GeoJSON name mapping via spatial join ─────────────────────────
def build_node_map(n):
    """
    Map network node names → GeoJSON node names (from growth network naming).
    Uses point-in-polygon, falling back to nearest polygon for edge cases.
    Growth nodes already match the GeoJSON; density nodes differ in voltage-
    level digit so we re-derive the match from bus coordinates.
    """
    stores = n.stores[n.stores.carrier.str.contains("afforestation", case=False, na=False)]
    nodes  = stores.index.str.replace(" co2 afforestation", "", regex=False)
    x = n.buses.loc[nodes, "x"].values
    y = n.buses.loc[nodes, "y"].values

    pts = gpd.GeoDataFrame(
        {"node": nodes.values},
        geometry=[Point(xi, yi) for xi, yi in zip(x, y)],
        crs="EPSG:4326"
    )
    joined = gpd.sjoin(pts, gdf[["geometry"]].reset_index(),
                       how="left", predicate="within")

    # Nearest-polygon fallback for any unmatched (e.g. border/island nodes)
    unmatched = joined["name"].isna()
    if unmatched.any():
        fallback = gpd.sjoin_nearest(
            pts[unmatched.values],
            gdf[["geometry"]].reset_index(),
            how="left"
        )
        joined.loc[unmatched.values, "name"] = fallback["name"].values

    return dict(zip(joined["node"], joined["name"]))

print("Building node→GeoJSON mappings...")
node_map_g = build_node_map(ng)   # growth  nodes → GeoJSON names
node_map_d = build_node_map(nd)   # density nodes → GeoJSON names
print(f"  Growth  mapped: {sum(v is not None for v in node_map_g.values())}/90")
print(f"  Density mapped: {sum(v is not None for v in node_map_d.values())}/90")

def get_affo_data(n, node_map):
    """Extract afforestation stores with timeseries.
    node_map: dict {network_node_name → GeoJSON_node_name}
    """
    stores = n.stores[n.stores.carrier.str.contains("afforestation", case=False, na=False)].copy()
    nodes = stores.index.str.replace(" co2 afforestation", "", regex=False)
    stores["node"]      = nodes.values
    # GeoJSON name used for polygon lookup and area
    stores["geojson_node"] = [node_map.get(nm, nm) for nm in nodes]
    # Potential in tCO2/yr (finite only)
    stores["potential_tCO2yr"] = np.where(
        stores["e_nom_max"] < 1e18, stores["e_nom_max"], np.nan
    )
    stores["potential_MtCO2"] = stores["potential_tCO2yr"] / 1e6
    # Timeseries: cumulative CO2 per node (tCO2), indexed by network node name
    e_ts = n.stores_t.e[stores.index].copy()
    e_ts.columns = stores["node"].values
    return stores, e_ts

stores_g, e_g = get_affo_data(ng, node_map_g)
stores_d, e_d = get_affo_data(nd, node_map_d)

print(f"  Growth  — total potential: {stores_g['potential_MtCO2'].sum():.1f} MtCO2/yr")
print(f"  Density — total potential: {stores_d['potential_MtCO2'].sum():.1f} MtCO2/yr")

# ── Node areas aligned to stores ──────────────────────────────────────────────
def get_node_areas(stores_df):
    """Return area_km2 array aligned to stores_df index (uses GeoJSON node names)."""
    gnodes = stores_df["geojson_node"].values
    areas  = np.array([gdf.loc[n, "area_km2"] if n in gdf.index else np.nan for n in gnodes])
    return areas

areas_g = get_node_areas(stores_g)   # km²
areas_d = get_node_areas(stores_d)

# Avg rate: tCO2/yr / (km² * 100 ha/km²) = tCO2/ha/yr
stores_g["rate_tCO2_ha_yr"] = stores_g["potential_tCO2yr"] / (areas_g * 100)
stores_d["rate_tCO2_ha_yr"] = stores_d["potential_tCO2yr"] / (areas_d * 100)


# ── Colormap helpers ──────────────────────────────────────────────────────────
def truncated_cmap(base_cmap, minval=0.15, maxval=1.0, n=256):
    """Slice a matplotlib colormap to avoid the near-white end."""
    base = cm.get_cmap(base_cmap, n)
    colors = base(np.linspace(minval, maxval, n))
    return LinearSegmentedColormap.from_list(
        f"trunc_{base_cmap}_{minval:.2f}_{maxval:.2f}", colors
    )

CMAP_POT   = truncated_cmap("YlOrRd",  minval=0.12)
CMAP_RATE  = truncated_cmap("YlGn",    minval=0.12)
CMAP_MO    = truncated_cmap("Blues",   minval=0.10)


# ── Choropleth map drawing ─────────────────────────────────────────────────────
def _base_map(ax):
    ax.set_extent(EUROPE_EXTENT, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND,      facecolor="#efefea", zorder=0)
    ax.add_feature(cfeature.OCEAN,     facecolor="#cde8f0", zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.35, zorder=2)
    ax.add_feature(cfeature.BORDERS,   linewidth=0.25, linestyle=":", zorder=2)


def draw_choropleth(ax, stores_df, values, norm, cmap, title, total_label,
                    missing_color="#d0d0d0"):
    """
    Fill node polygons with choropleth colour.
    values: array aligned to stores_df rows.
    Uses geojson_node column for polygon lookup (handles cross-run name differences).
    """
    _base_map(ax)

    gnodes = stores_df["geojson_node"].values
    geoms  = [gdf.loc[n, "geometry"] if n in gdf.index else None for n in gnodes]
    vals   = np.asarray(values, dtype=float)

    for geom, v in zip(geoms, vals):
        if geom is None:
            continue
        color = missing_color if (not np.isfinite(v) or v <= 0) else cmap(norm(v))
        ax.add_geometries(
            [geom], crs=ccrs.PlateCarree(),
            facecolor=color, edgecolor="white", linewidth=0.25, zorder=1
        )

    ax.set_title(title, fontsize=10.5, fontweight="bold", pad=5)
    ax.text(0.03, 0.04, total_label,
            transform=ax.transAxes, fontsize=8, va="bottom", ha="left",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.5", alpha=0.88))


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — 2×2: potential maps (top) + avg rate maps (bottom)
# ═══════════════════════════════════════════════════════════════════════════════
print("\nFigure 1: potential + rate maps...")

pot_g = stores_g["potential_MtCO2"].values
pot_d = stores_d["potential_MtCO2"].values
rat_g = stores_g["rate_tCO2_ha_yr"].values
rat_d = stores_d["rate_tCO2_ha_yr"].values

# Log-scale norm for potential (a few large-area nodes dominate)
valid_pot = np.concatenate([pot_g[np.isfinite(pot_g) & (pot_g > 0)],
                            pot_d[np.isfinite(pot_d) & (pot_d > 0)]])
norm_pot = mcolors.LogNorm(vmin=valid_pot.min(), vmax=valid_pot.max())

# Linear norm for rate
valid_rat = np.concatenate([rat_g[np.isfinite(rat_g) & (rat_g > 0)],
                            rat_d[np.isfinite(rat_d) & (rat_d > 0)]])
norm_rat = mcolors.Normalize(vmin=0, vmax=np.nanpercentile(valid_rat, 98))

fig1, axes1 = plt.subplots(
    2, 2, figsize=(13, 11),
    subplot_kw={"projection": LAMBERT}
)
fig1.subplots_adjust(wspace=0.05, hspace=0.06,
                     left=0.02, right=0.87, top=0.91, bottom=0.03)

# Top row — total potential
draw_choropleth(
    axes1[0, 0], stores_g, pot_g, norm_pot, CMAP_POT,
    title="Growth method — total potential",
    total_label=f"Total: {np.nansum(pot_g):.1f} Mt CO₂/yr"
)
draw_choropleth(
    axes1[0, 1], stores_d, pot_d, norm_pot, CMAP_POT,
    title="Density method — total potential",
    total_label=f"Total: {np.nansum(pot_d):.1f} Mt CO₂/yr"
)

# Bottom row — avg rate
draw_choropleth(
    axes1[1, 0], stores_g, rat_g, norm_rat, CMAP_RATE,
    title="Growth method — avg seq. rate",
    total_label=f"Mean: {np.nanmean(rat_g):.2f} t CO₂/ha/yr"
)
draw_choropleth(
    axes1[1, 1], stores_d, rat_d, norm_rat, CMAP_RATE,
    title="Density method — avg seq. rate",
    total_label=f"Mean: {np.nanmean(rat_d):.2f} t CO₂/ha/yr"
)

# Row labels
for row_idx, row_label in enumerate(
        ["Annual sequestration\npotential [Mt CO₂/yr/node]",
         "Avg seq. rate\n[t CO₂/ha/yr]"]):
    cax = fig1.add_axes([0.89, 0.55 - row_idx * 0.50, 0.016, 0.38])
    sm  = cm.ScalarMappable(
        norm=norm_pot if row_idx == 0 else norm_rat,
        cmap=CMAP_POT if row_idx == 0 else CMAP_RATE
    )
    sm.set_array([])
    cb = fig1.colorbar(sm, cax=cax)
    cb.set_label(row_label, fontsize=8.5)
    if row_idx == 0:
        cb.ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{x:.2g}")
        )

fig1.suptitle(
    "Afforestation CO₂ sequestration — growth vs. density method (90 nodes, 2050)",
    fontsize=12, y=0.97
)

for ext in ("pdf", "png"):
    fig1.savefig(OUT_DIR / f"fig1_afforestation_potential_map.{ext}",
                 dpi=200, bbox_inches="tight")
print("  Saved fig1")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURES 2 & 3 — Monthly State of Charge (0–100%)
# ═══════════════════════════════════════════════════════════════════════════════
def monthly_soc(e_ts, stores_df):
    """
    For each month 1–12, get the SoC (%) = e(t_last_of_month) / e_nom_opt × 100
    per node. Returns dict {month: Series[node → %]}.
    """
    # Use e_nom_opt if present, else e_nom_max
    if "e_nom_opt" in stores_df.columns:
        cap = stores_df.set_index("node")["e_nom_opt"]
    else:
        cap = stores_df.set_index("node")["e_nom_max"]

    out = {}
    for m in range(1, 13):
        mask = e_ts.index.month == m
        if mask.any():
            e_end = e_ts[mask].iloc[-1]
        else:
            e_end = e_ts.iloc[0] * 0.0
        soc = (e_end / cap.reindex(e_end.index)) * 100.0
        soc = soc.clip(0, 100)
        out[m] = soc
    return out

monthly_soc_g = monthly_soc(e_g, stores_g)
monthly_soc_d = monthly_soc(e_d, stores_d)

norm_soc = mcolors.Normalize(vmin=0, vmax=100)

def make_monthly_soc_figure(monthly_vals, stores_df, method_label, fig_tag):
    fig, axes = plt.subplots(
        3, 4, figsize=(18, 12),
        subplot_kw={"projection": LAMBERT}
    )
    fig.subplots_adjust(
        wspace=0.04, hspace=0.08,
        left=0.01, right=0.87, top=0.92, bottom=0.03
    )

    for idx, m in enumerate(range(1, 13)):
        ax   = axes[idx // 4][idx % 4]
        vals = monthly_vals[m].reindex(stores_df["node"]).fillna(0).values

        total_abs = (monthly_soc_g if "growth" in fig_tag else monthly_soc_d)[m]
        # For label use e_ts directly
        draw_choropleth(
            ax, stores_df, vals, norm_soc, CMAP_MO,
            title=MONTH_NAMES[m - 1],
            total_label=f"Avg SoC: {np.nanmean(vals):.1f}%"
        )

    cax = fig.add_axes([0.89, 0.10, 0.015, 0.76])
    sm  = cm.ScalarMappable(norm=norm_soc, cmap=CMAP_MO)
    sm.set_array([])
    cb  = fig.colorbar(sm, cax=cax)
    cb.set_label("State of Charge [%]\n(cumul. CO₂ / annual capacity)", fontsize=9)

    fig.suptitle(
        f"Monthly afforestation store State of Charge — {method_label} (90 nodes, 2050)\n"
        "SoC = cumulative CO₂ sequestered / annual potential capacity × 100%",
        fontsize=12, y=0.97
    )

    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"{fig_tag}.{ext}", dpi=200, bbox_inches="tight")
    print(f"  Saved {fig_tag}")
    return fig

print("\nFigure 2: monthly SoC maps — growth method...")
fig2 = make_monthly_soc_figure(monthly_soc_g, stores_g,
                               "Growth method (Pilli MAI, seasonal)",
                               "fig2_monthly_soc_growth")

print("Figure 3: monthly SoC maps — density method...")
fig3 = make_monthly_soc_figure(monthly_soc_d, stores_d,
                               "Density method (biomass density, linear)",
                               "fig3_monthly_soc_density")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — NUTS0 stacked area: growth vs density over time
# ═══════════════════════════════════════════════════════════════════════════════
print("\nFigure 4: NUTS0 stacked area timeseries...")

def nuts0_timeseries(e_ts):
    """Aggregate tCO2 timeseries to NUTS0 (country) level. Returns DataFrame."""
    countries = {col: col[:2] for col in e_ts.columns}
    country_ts = e_ts.rename(columns=countries).T.groupby(level=0).sum().T
    return country_ts / 1e6   # → MtCO2

ts_g_nuts0 = nuts0_timeseries(e_g)
ts_d_nuts0 = nuts0_timeseries(e_d)

# Sort countries by total (end-of-year) for consistent stacking
order_g = ts_g_nuts0.iloc[-1].sort_values(ascending=False).index
order_d = ts_d_nuts0.iloc[-1].sort_values(ascending=False).index
# Use union of countries, sorted by growth total
all_countries = ts_g_nuts0.columns.union(ts_d_nuts0.columns)
final_order   = ts_g_nuts0.reindex(columns=all_countries).iloc[-1]\
                          .sort_values(ascending=False).index

# Color palette
n_ctry = len(final_order)
cmap_ctry = cm.get_cmap("tab20", n_ctry)
ctry_colors = {c: cmap_ctry(i) for i, c in enumerate(final_order)}

def plot_stacked(ax, ts, order, title, ylabel=True):
    ts_ordered = ts.reindex(columns=order, fill_value=0.0)
    time = ts_ordered.index

    # Build cumulative stacks
    cumsum = np.zeros(len(time))
    polys  = []
    labels = []
    for ctry in order:
        vals  = ts_ordered[ctry].values
        upper = cumsum + vals
        polys.append((cumsum.copy(), upper.copy()))
        labels.append(ctry)
        cumsum = upper

    for (lo, hi), ctry in zip(polys, labels):
        ax.fill_between(time, lo, hi,
                        color=ctry_colors[ctry], alpha=0.88,
                        label=ctry, linewidth=0)

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlim(time[0], time[-1])
    ax.set_ylim(0)
    if ylabel:
        ax.set_ylabel("Cumulative CO₂ sequestered [Mt CO₂]", fontsize=9)
    ax.set_xlabel("Time", fontsize=9)
    ax.tick_params(axis="x", labelsize=8)

    # Annotate total at year end
    total_end = ts_ordered.iloc[-1].sum()
    ax.text(0.98, 0.97, f"Year total: {total_end:.1f} Mt CO₂",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.5", alpha=0.85))

fig4, (ax4a, ax4b) = plt.subplots(2, 1, figsize=(12, 9), sharex=False)
fig4.subplots_adjust(hspace=0.35, left=0.09, right=0.82, top=0.91, bottom=0.08)

plot_stacked(ax4a, ts_g_nuts0, final_order,
             "Growth method (Pilli MAI, seasonal)", ylabel=True)
plot_stacked(ax4b, ts_d_nuts0, final_order,
             "Density method (biomass density, linear)", ylabel=True)

# Shared legend
handles = [plt.Rectangle((0, 0), 1, 1, color=ctry_colors[c])
           for c in final_order]
fig4.legend(handles, list(final_order),
            loc="center right", bbox_to_anchor=(1.0, 0.5),
            ncol=1, fontsize=7.5, frameon=True, title="Country")

fig4.suptitle(
    "NUTS0 cumulative afforestation CO₂ sequestration — growth vs. density method (2050)",
    fontsize=12, y=0.97
)

for ext in ("pdf", "png"):
    fig4.savefig(OUT_DIR / f"fig4_nuts0_timeseries.{ext}",
                 dpi=200, bbox_inches="tight")
print("  Saved fig4")

print("\nDone. All figures saved to:", OUT_DIR)
plt.close("all")
