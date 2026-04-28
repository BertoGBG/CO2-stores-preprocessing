"""
Comparison of PyPSA-Eur afforestation CDR: growth method vs density method.

Generates:
  fig_affo_pypsa_comparison_map.png/.pdf   — 1×3 choropleth showing annual
      potentials: growth | density (scaled ×0.6) | density (intrinsic ÷0.6)
  fig_affo_pypsa_comparison_country.png/.pdf — country-level bar chart

Notes:
  - CRCF efficiency (η=0.8) is applied only in the growth method (forced
    seasonal dispatch, p_min_pu = p_max_pu). The density method runs with
    free dispatch and no CRCF efficiency discount.
  - Density method ran with f_land = 0.6 applied to the intrinsic potential.
  - The intrinsic density potential (e_nom_max / 0.6) is shown to allow a
    like-for-like comparison of the underlying physical potentials.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pypsa
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import warnings

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE = Path("/Users/albal/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/"
            "01_DTU_research/01_Projects/04_CO2-stores-preprocessing/CO2-stores-preprocessing")
GROWTH_NC  = BASE / "Results_analysis/affo_growth_90/base_s_90__168h_2050.nc"
DENSITY_NC = BASE / "Results_analysis/affo_density_90/base_s_90__168h_2050.nc"
GEOJSON    = Path("/Users/albal/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/"
                  "01_DTU_research/01_Projects/01_pypsa-eur_AA/pypsa-eur/resources/"
                  "regions_onshore_base_s_90.geojson")
OUT_DIR    = Path(__file__).parent
F_LAND     = 0.6

# ── Map projection & extent ────────────────────────────────────────────────────
LAMBERT = ccrs.LambertConformal(
    central_longitude=10, central_latitude=52,
    standard_parallels=(35, 65)
)
EXTENT = [-12, 30, 34, 72]


def trunc_cmap(name, lo=0.12, hi=1.0, n=256):
    base = cm.get_cmap(name, n)
    return LinearSegmentedColormap.from_list(
        f"tr_{name}", base(np.linspace(lo, hi, n))
    )


# Green colormap consistent with Fig. 4 in the paper
CMAP_GREEN = trunc_cmap("YlGn", lo=0.12)

# ── Load GeoJSON ───────────────────────────────────────────────────────────────
print("Loading GeoJSON …")
gdf = gpd.read_file(GEOJSON).set_index("name")

# ── Load networks ──────────────────────────────────────────────────────────────
print("Loading networks …")
ng = pypsa.Network(str(GROWTH_NC))
nd = pypsa.Network(str(DENSITY_NC))


def extract_affo(n):
    """Return per-node DataFrame with potential, used potential and annual CDR."""
    al  = n.links[n.links.carrier == "co2 afforestation"].copy()
    as_ = n.stores[n.stores.carrier == "co2 afforestation"].copy()
    wt  = n.snapshot_weightings["generators"]
    node = al.index.str.replace(" afforestation", "", regex=False)
    al["node"]  = node.values
    as_["node"] = node.values

    # Annual CDR from atmosphere (p0 × snapshot weights)
    p0     = n.links_t["p0"][al.index]
    ann_p0 = p0.multiply(wt, axis=0).sum()
    ann_p0.index = node.values

    as_idx = as_.set_index("node")
    pot      = as_idx["e_nom_max"]    # tCO2/yr — total available potential
    used     = as_idx["e_nom_opt"]    # tCO2/yr — optimally deployed potential
    cap_cost = as_idx["capital_cost"] # EUR/tCO2
    return pd.DataFrame({
        "potential_tCO2yr": pot,
        "used_tCO2yr":      used,
        "cdr_atm_tCO2yr":   ann_p0,
        "capital_cost":     cap_cost,
        "country": [c[:2] for c in pot.index],
    })


print("Extracting afforestation data …")
df_g = extract_affo(ng)
df_d = extract_affo(nd)

# Intrinsic density potential (before f_land = 0.6)
df_d["pot_intrinsic_tCO2yr"] = df_d["potential_tCO2yr"] / F_LAND

# ── Summary statistics ─────────────────────────────────────────────────────────
co2_price_g = abs(ng.global_constraints.loc["CO2Limit", "mu"])
co2_price_d = abs(nd.global_constraints.loc["CO2Limit", "mu"])
seq_mu_g    = ng.global_constraints.loc["co2_sequestration_limit", "mu"]
seq_mu_d    = nd.global_constraints.loc["co2_sequestration_limit", "mu"]

print("\n=== Summary Statistics ===")
print(f"Growth  potential (e_nom_max)              : {df_g['potential_tCO2yr'].sum()/1e6:.2f} MtCO2/yr")
print(f"Density potential (scaled ×{F_LAND})           : {df_d['potential_tCO2yr'].sum()/1e6:.2f} MtCO2/yr")
print(f"Density potential (intrinsic, ÷{F_LAND})       : {df_d['pot_intrinsic_tCO2yr'].sum()/1e6:.2f} MtCO2/yr")
print(f"Growth  CDR from atm (η=0.8 applied)      : {df_g['cdr_atm_tCO2yr'].sum()/1e6:.2f} MtCO2/yr")
print(f"  → credited CDR (×0.8)                  : {df_g['cdr_atm_tCO2yr'].sum()*0.8/1e6:.2f} MtCO2/yr")
print(f"Density CDR from atm (no η discount)      : {df_d['cdr_atm_tCO2yr'].sum()/1e6:.2f} MtCO2/yr")
print(f"Growth  utilisation                       : {df_g['cdr_atm_tCO2yr'].sum()*0.8/df_g['potential_tCO2yr'].sum()*100:.0f}%")
print(f"Density utilisation                       : {df_d['cdr_atm_tCO2yr'].sum()/df_d['potential_tCO2yr'].sum()*100:.0f}%")
# Weighted average capital cost: sum(capital_cost_i * e_nom_opt_i) / sum(e_nom_opt_i)
used_g = df_g["used_tCO2yr"]
used_d = df_d["used_tCO2yr"]
wt_cap_g = (df_g["capital_cost"] * used_g).sum() / used_g.sum() if used_g.sum() > 0 else float("nan")
wt_cap_d = (df_d["capital_cost"] * used_d).sum() / used_d.sum() if used_d.sum() > 0 else float("nan")

print(f"Growth  CO2 price : {co2_price_g:.0f} EUR/tCO2")
print(f"Density CO2 price : {co2_price_d:.0f} EUR/tCO2")
print(f"Growth  seq. limit shadow price : {seq_mu_g:.0f} EUR/tCO2")
print(f"Density seq. limit shadow price : {seq_mu_d:.0f} EUR/tCO2")
print(f"Growth  weighted avg capital cost: {wt_cap_g:.0f} EUR/tCO2")
print(f"Density weighted avg capital cost: {wt_cap_d:.0f} EUR/tCO2")

print("\n=== Table 2 values (copy into main.tex) ===")
print(f"Wtd. avg. capital cost  growth : {wt_cap_g:.0f} EUR/tCO2")
print(f"Wtd. avg. capital cost  density: {wt_cap_d:.0f} EUR/tCO2")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — 1×3 choropleth: growth potential | density scaled | density intrinsic
# ══════════════════════════════════════════════════════════════════════════════
print("\nBuilding Figure 1: 1×3 potential choropleth …")

nodes = df_g.index.values


def base_map(ax):
    ax.set_extent(EXTENT, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND,      facecolor="#efefea", zorder=0)
    ax.add_feature(cfeature.OCEAN,     facecolor="#cde8f0", zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.35, zorder=2)
    ax.add_feature(cfeature.BORDERS,   linewidth=0.25, linestyle=":", zorder=2)


def draw_choropleth(ax, nodes, values_tCO2yr, norm, cmap, title, footer,
                    missing="#d0d0d0"):
    base_map(ax)
    vals = np.asarray(values_tCO2yr, dtype=float) / 1e6   # → MtCO2/yr
    for node, v in zip(nodes, vals):
        if node not in gdf.index:
            continue
        color = missing if (not np.isfinite(v) or v <= 0) else cmap(norm(v))
        ax.add_geometries([gdf.loc[node, "geometry"]], crs=ccrs.PlateCarree(),
                          facecolor=color, edgecolor="white", linewidth=0.25, zorder=1)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=4)
    ax.text(0.03, 0.04, footer, transform=ax.transAxes,
            fontsize=7.5, va="bottom", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.5", alpha=0.88))


# Shared norm across all three panels
all_pot = np.concatenate([
    df_g["potential_tCO2yr"].values / 1e6,
    df_d["potential_tCO2yr"].values / 1e6,
    df_d["pot_intrinsic_tCO2yr"].values / 1e6,
])
pos = all_pot[np.isfinite(all_pot) & (all_pot > 0)]
norm_pot = mcolors.LogNorm(vmin=pos.min(), vmax=pos.max())

fig1, axes = plt.subplots(
    1, 3, figsize=(18, 6.5),
    subplot_kw={"projection": LAMBERT}
)
fig1.subplots_adjust(wspace=0.04, left=0.01, right=0.90, top=0.88, bottom=0.03)

draw_choropleth(
    axes[0], nodes, df_g["potential_tCO2yr"].values, norm_pot, CMAP_GREEN,
    title="(a)  Growth method",
    footer=f"Total: {df_g['potential_tCO2yr'].sum()/1e6:.1f}\u2009Mt\u2009CO\u2082\u2009yr\u207b\u00b9"
)
draw_choropleth(
    axes[1], nodes, df_d["potential_tCO2yr"].values, norm_pot, CMAP_GREEN,
    title=r"(b)  Density method  ($\times\,0.6$  scaling applied)",
    footer=f"Total: {df_d['potential_tCO2yr'].sum()/1e6:.1f}\u2009Mt\u2009CO\u2082\u2009yr\u207b\u00b9"
)
draw_choropleth(
    axes[2], nodes, df_d["pot_intrinsic_tCO2yr"].values, norm_pot, CMAP_GREEN,
    title=r"(c)  Density method  (intrinsic $\tilde{r}$, before $\times\,0.6$)",
    footer=f"Total: {df_d['pot_intrinsic_tCO2yr'].sum()/1e6:.1f}\u2009Mt\u2009CO\u2082\u2009yr\u207b\u00b9"
)

# Single shared colorbar
cax = fig1.add_axes([0.92, 0.10, 0.015, 0.72])
sm  = cm.ScalarMappable(norm=norm_pot, cmap=CMAP_GREEN)
sm.set_array([])
cb  = fig1.colorbar(sm, cax=cax)
cb.set_label("Annual potential\n[Mt\u2009CO\u2082\u2009yr\u207b\u00b9 per node]", fontsize=9)
cb.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2g}"))

fig1.suptitle(
    "Afforestation annual CO\u2082 sequestration potential — growth vs. density method (90 nodes, 2050)",
    fontsize=12, y=0.97
)

for ext in ("pdf", "png"):
    fig1.savefig(OUT_DIR / f"fig_affo_pypsa_comparison_map.{ext}",
                 dpi=200, bbox_inches="tight")
print("  Saved fig_affo_pypsa_comparison_map")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Country-level bar chart: potential vs. used potential
#   Top panel    : growth method  — potential (e_nom_max) | used (e_nom_opt)
#   Bottom panel : density method (scaled ×0.6 only) — same two columns
# ══════════════════════════════════════════════════════════════════════════════
print("Building Figure 2: country bar chart …")

# Colours: lighter shade = available potential, darker = deployed
C_POT_G  = "#a1d99b"   # light green  — growth potential
C_USED_G = "#238b45"   # dark green   — growth used
C_POT_D  = "#9ecae1"   # light blue   — density potential
C_USED_D = "#2171b5"   # dark blue    — density used


def country_agg(df, col):
    return df.groupby("country")[col].sum() / 1e6


ctry_pot_g  = country_agg(df_g, "potential_tCO2yr")
ctry_used_g = country_agg(df_g, "used_tCO2yr")
ctry_pot_d  = country_agg(df_d, "potential_tCO2yr")   # scaled ×0.6
ctry_used_d = country_agg(df_d, "used_tCO2yr")

# Sort countries by growth potential (descending)
countries = ctry_pot_g.reindex(
    ctry_pot_g.index.union(ctry_pot_d.index)
).fillna(0).sort_values(ascending=False).index

x = np.arange(len(countries))
w = 0.35

fig2, axes2 = plt.subplots(2, 1, figsize=(14, 10))
fig2.subplots_adjust(hspace=0.45, left=0.07, right=0.97, top=0.93, bottom=0.08)

# ── Panel A: growth method ────────────────────────────────────────────────────
ax = axes2[0]
ax.bar(x - w/2,
       [ctry_pot_g.get(c, 0) for c in countries], w,
       label="Potential ($e_{\\mathrm{nom,max}}$)", color=C_POT_G, alpha=0.9)
ax.bar(x + w/2,
       [ctry_used_g.get(c, 0) for c in countries], w,
       label="Deployed ($e_{\\mathrm{nom,opt}}$)", color=C_USED_G, alpha=0.9)
ax.set_xticks(x)
ax.set_xticklabels(countries, fontsize=8, rotation=45, ha="right")
ax.set_ylabel("Annual CO\u2082 sequestration [Mt\u2009yr\u207b\u00b9]", fontsize=9)
ax.set_title("(a)  Growth method", fontsize=10, fontweight="bold")
ax.legend(fontsize=8, loc="upper right", framealpha=0.9)
ax.set_ylim(bottom=0)
ax.grid(axis="y", linewidth=0.4, alpha=0.5)

# ── Panel B: density method (scaled ×0.6 only) ───────────────────────────────
ax = axes2[1]
ax.bar(x - w/2,
       [ctry_pot_d.get(c, 0) for c in countries], w,
       label="Potential ($e_{\\mathrm{nom,max}}$)", color=C_POT_D, alpha=0.9)
ax.bar(x + w/2,
       [ctry_used_d.get(c, 0) for c in countries], w,
       label="Deployed ($e_{\\mathrm{nom,opt}}$)", color=C_USED_D, alpha=0.9)
ax.set_xticks(x)
ax.set_xticklabels(countries, fontsize=8, rotation=45, ha="right")
ax.set_ylabel("Annual CO\u2082 sequestration [Mt\u2009yr\u207b\u00b9]", fontsize=9)
ax.set_title(r"(b)  Density method ($\times\,0.6$ scaling)", fontsize=10, fontweight="bold")
ax.legend(fontsize=8, loc="upper right", framealpha=0.9)
ax.set_ylim(bottom=0)
ax.grid(axis="y", linewidth=0.4, alpha=0.5)

fig2.suptitle(
    "Country-level afforestation CDR potential and deployment (90 nodes, 2050)",
    fontsize=11, y=0.98
)

for ext in ("pdf", "png"):
    fig2.savefig(OUT_DIR / f"fig_affo_pypsa_comparison_country.{ext}",
                 dpi=200, bbox_inches="tight")
print("  Saved fig_affo_pypsa_comparison_country")

# ── Print LaTeX table values ───────────────────────────────────────────────────
print("\n=== LaTeX table values ===")
print(f"Growth  potential                   : {df_g['potential_tCO2yr'].sum()/1e6:.1f} MtCO2/yr")
print(f"Density potential (intrinsic, /0.6) : {df_d['pot_intrinsic_tCO2yr'].sum()/1e6:.1f} MtCO2/yr")
print(f"Density potential (scaled x0.6)     : {df_d['potential_tCO2yr'].sum()/1e6:.1f} MtCO2/yr")
print(f"Growth  credited CDR (eta=0.8)      : {df_g['cdr_atm_tCO2yr'].sum()*0.8/1e6:.1f} MtCO2/yr")
print(f"Growth  CDR from atmosphere         : {df_g['cdr_atm_tCO2yr'].sum()/1e6:.1f} MtCO2/yr")
print(f"Density CDR from atmosphere (no eta): {df_d['cdr_atm_tCO2yr'].sum()/1e6:.1f} MtCO2/yr")
print(f"Growth  utilisation (credited/pot)  : {df_g['cdr_atm_tCO2yr'].sum()*0.8/df_g['potential_tCO2yr'].sum()*100:.0f}%")
print(f"Density utilisation (CDR/scaled)    : {df_d['cdr_atm_tCO2yr'].sum()/df_d['potential_tCO2yr'].sum()*100:.0f}%")
print(f"Growth  CO2 price                   : {co2_price_g:.0f} EUR/tCO2")
print(f"Density CO2 price                   : {co2_price_d:.0f} EUR/tCO2")
print(f"Growth  seq. limit shadow price     : {seq_mu_g:.0f} EUR/tCO2")
print(f"Density seq. limit shadow price     : {seq_mu_d:.0f} EUR/tCO2")
print(f"Growth  weighted avg capital cost   : {wt_cap_g:.0f} EUR/tCO2")
print(f"Density weighted avg capital cost   : {wt_cap_d:.0f} EUR/tCO2")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Capital cost map: EUR/tCO2 per node, growth vs density
# ══════════════════════════════════════════════════════════════════════════════
print("Building Figure 3: capital cost choropleth …")

CMAP_COST = trunc_cmap("YlOrRd", lo=0.10, hi=0.98)

cap_g = df_g["capital_cost"].values   # EUR/tCO2
cap_d = df_d["capital_cost"].values   # EUR/tCO2

# Shared linear norm across both panels
all_cost = np.concatenate([cap_g, cap_d])
finite_cost = all_cost[np.isfinite(all_cost) & (all_cost > 0)]
norm_cost = mcolors.Normalize(vmin=finite_cost.min(), vmax=np.percentile(finite_cost, 98))


def draw_cost_map(ax, nodes, cap_values, norm, cmap, title, footer,
                  missing="#d0d0d0"):
    base_map(ax)
    for node, v in zip(nodes, cap_values):
        if node not in gdf.index:
            continue
        color = missing if (not np.isfinite(v) or v <= 0) else cmap(norm(v))
        ax.add_geometries([gdf.loc[node, "geometry"]], crs=ccrs.PlateCarree(),
                          facecolor=color, edgecolor="white", linewidth=0.25, zorder=1)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=4)
    ax.text(0.03, 0.04, footer, transform=ax.transAxes,
            fontsize=7.5, va="bottom", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.5", alpha=0.88))


fig3, axes3 = plt.subplots(
    1, 2, figsize=(13, 5.5),
    subplot_kw={"projection": LAMBERT}
)
fig3.subplots_adjust(wspace=0.04, left=0.01, right=0.88, top=0.88, bottom=0.03)

draw_cost_map(
    axes3[0], nodes, cap_g, norm_cost, CMAP_COST,
    title="(a)  Growth method",
    footer=f"Wtd avg: {wt_cap_g:.0f}\u2009EUR\u2009tCO\u2082\u207b\u00b9"
)
draw_cost_map(
    axes3[1], nodes, cap_d, norm_cost, CMAP_COST,
    title=r"(b)  Density method ($\times\,0.6$ scaling)",
    footer=f"Wtd avg: {wt_cap_d:.0f}\u2009EUR\u2009tCO\u2082\u207b\u00b9"
)

# Shared colorbar
cax3 = fig3.add_axes([0.90, 0.10, 0.018, 0.72])
sm3  = cm.ScalarMappable(norm=norm_cost, cmap=CMAP_COST)
sm3.set_array([])
cb3  = fig3.colorbar(sm3, cax=cax3)
cb3.set_label("Capital cost\n[EUR\u2009tCO\u2082\u207b\u00b9]", fontsize=9)
cb3.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}"))

fig3.suptitle(
    "Afforestation capital cost per node — growth vs. density method (90 nodes, 2050)",
    fontsize=11, y=0.97
)

for ext in ("pdf", "png"):
    fig3.savefig(OUT_DIR / f"fig_affo_pypsa_comparison_capex.{ext}",
                 dpi=200, bbox_inches="tight")
print("  Saved fig_affo_pypsa_comparison_capex")

print("\nDone. Figures saved to:", OUT_DIR)
plt.close("all")
