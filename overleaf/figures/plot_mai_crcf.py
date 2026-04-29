"""
Explanatory figure: MAI growth dynamics, cumulative certified stock, and CRCF
credit decomposition.

Produces:  overleaf/figures/mai_crcf_explained.pdf  (and .png)

Three-panel figure:
  Panel A – Stand growth dynamics: CAI and MAI vs. stand age.
  Panel B – Cumulative carbon stock decomposed into certified stock (must be
             held), buffer-pool reserve, and UNC-based harvest/safety margin.
  Panel C – CRCF credit decomposition for three disturbance scenarios (bars).

Key identity:  (1-UNC)(1-r) + (1-UNC)·r + UNC = 1

Assumptions (CRCF draft delegated act, 22-Jan-2026):
  UNC = 10 %,  risk rates: Low 8 %, Central 11.1 %, Medium 17 %
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import os

# ── style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ── colours ───────────────────────────────────────────────────────────────────
CAI_COLOR = "#1565C0"
MAI_COLOR = "#E65100"
COLOR_NET = "#2e7d32"
COLOR_BUF = "#f57c00"
COLOR_UNC = "#b71c1c"

# ── growth model ──────────────────────────────────────────────────────────────
TARGET_T_ROT = 80.0   # years
TARGET_MAI   = 5.2    # tCO2/ha/yr

tau  = TARGET_T_ROT / 1.7932
ages = np.linspace(0.1, 160, 3200)

def cai_fn(t, c, tau):
    return c * t * np.exp(-t / tau)

def cum_fn(t, c, tau):
    return c * tau**2 * (1.0 - np.exp(-t / tau) * (1.0 + t / tau))

def mai_fn(t, c, tau):
    return cum_fn(t, c, tau) / t

c = TARGET_MAI / mai_fn(np.array([TARGET_T_ROT]), 1.0, tau)[0]

cai_vals = cai_fn(ages, c, tau)
mai_vals = mai_fn(ages, c, tau)
cum_vals = cum_fn(ages, c, tau)

# rotation age
diff = cai_vals - mai_vals
idx  = np.where(np.diff(np.sign(diff)) < 0)[0]
t1, t2 = ages[idx[0]], ages[idx[0] + 1]
T_rot = t1 - diff[idx[0]] * (t2 - t1) / (diff[idx[0] + 1] - diff[idx[0]])
MAI_max   = mai_fn(np.array([T_rot]), c, tau)[0]
CUM_T_rot = cum_fn(np.array([T_rot]), c, tau)[0]
T_CRCF    = 30.0
CUM_30    = cum_fn(np.array([T_CRCF]), c, tau)[0]

# ── CRCF fractions (central estimate) ─────────────────────────────────────────
UNC          = 0.10
RISK_CENTRAL = 0.111
f_net = (1 - UNC) * (1 - RISK_CENTRAL)   # 0.80
f_buf = (1 - UNC) * RISK_CENTRAL          # 0.10
f_unc = UNC                               # 0.10

# ── bar chart data ────────────────────────────────────────────────────────────
scenarios = {
    "Low\n(mixed, 8 %)":     {"risk": 0.08},
    "Central\n(default, 11 %)": {"risk": 0.111},
    "Medium\n(mixed, 17 %)": {"risk": 0.17},
}
for s in scenarios.values():
    s["net"] = (1 - UNC) * (1 - s["risk"])
    s["buf"] = (1 - UNC) * s["risk"]
    s["unc"] = UNC

# ── figure ────────────────────────────────────────────────────────────────────
fig, (ax1, ax2, ax3) = plt.subplots(
    1, 3,
    figsize=(13.5, 4.6),
    gridspec_kw={"width_ratios": [1.25, 1.35, 0.9]},
)
fig.subplots_adjust(left=0.07, right=0.97, top=0.91, bottom=0.13, wspace=0.32)

# ─────────────────────────────────────────────────────────────────────────────
# Panel A – growth dynamics
# ─────────────────────────────────────────────────────────────────────────────
ax1.plot(ages, cai_vals, color=CAI_COLOR, lw=1.8, label="CAI  (current annual increment)")
ax1.plot(ages, mai_vals, color=MAI_COLOR, lw=1.8, ls="--", label="MAI  (mean annual increment)")

mask = ages <= T_rot
ax1.fill_between(ages[mask], cai_vals[mask], alpha=0.10, color=CAI_COLOR)

ax1.axvline(T_rot,  color="0.40", lw=0.85, ls=":")
ax1.axvline(T_CRCF, color="0.60", lw=0.85, ls="--")
ax1.hlines(MAI_max, 0, T_rot, colors="0.40", lw=0.85, ls=":")

# rotation age label
ax1.text(T_rot + 2, 0.30, f"$T_{{\\rm rot}}\\approx{T_rot:.0f}$ yr",
         fontsize=7.5, color="0.35", va="bottom")

# MAI_max label – below the dotted hline
ax1.text(2, MAI_max + 0.18,
         f"MAI$_{{\\rm max}}\\approx{MAI_max:.1f}$ tCO$_2$ ha$^{{-1}}$ yr$^{{-1}}$",
         fontsize=7.5, color="0.35", va="bottom")

# CRCF period label
ax1.text(T_CRCF + 1.5, 0.30, f"$T_{{\\rm CRCF}}=30$ yr",
         fontsize=7.0, color="0.55", va="bottom")

# cumulative stock annotation inside shaded area
mid_age  = T_rot * 0.55
mid_cum  = cai_fn(np.array([mid_age]), c, tau)[0] * 0.45
ax1.text(mid_age, mid_cum,
         f"Cumul. stock\n$\\approx{CUM_T_rot:.0f}$ tCO$_2$/ha\nat $T_{{\\rm rot}}$",
         fontsize=7, color=CAI_COLOR, alpha=0.85, ha="center", va="center")

ax1.set_xlabel("Stand age (years)")
ax1.set_ylabel("C increment (tCO$_2$ ha$^{-1}$ yr$^{-1}$)")
ax1.set_title("(a)  Forest growth dynamics")
ax1.set_xlim(0, 155)
ax1.set_ylim(0)
ax1.xaxis.set_minor_locator(MultipleLocator(10))

# Right axis: accumulated carbon stock [tCO2/ha]
STOCK_COLOR = "#5e35b1"   # purple — distinct from CAI/MAI blues and oranges
ax1r = ax1.twinx()
ax1r.plot(ages, cum_vals, color=STOCK_COLOR, lw=1.5, ls="-.",
          label="Cumul. stock  (tCO$_2$ ha$^{-1}$)")
ax1r.set_ylabel("Cumulative stock (tCO$_2$ ha$^{-1}$)", color=STOCK_COLOR)
ax1r.tick_params(axis="y", colors=STOCK_COLOR, labelsize=8)
ax1r.spines["right"].set_visible(True)
ax1r.spines["right"].set_color(STOCK_COLOR)
ax1r.spines["right"].set_linewidth(0.8)
ax1r.set_ylim(0)

# Combined legend from both axes
handles1, labels1 = ax1.get_legend_handles_labels()
handles1r, labels1r = ax1r.get_legend_handles_labels()
ax1.legend(handles1 + handles1r, labels1 + labels1r,
           loc="upper right", frameon=False, handlelength=1.5)

# ─────────────────────────────────────────────────────────────────────────────
# Panel B – cumulative stock decomposition
# ─────────────────────────────────────────────────────────────────────────────
cum_net = f_net * cum_vals
cum_buf = f_buf * cum_vals

ax2.fill_between(ages, 0,                 cum_net,           color=COLOR_NET, alpha=0.75,
                 label=f"Net certified stock  ({f_net:.0%}, must be held)")
ax2.fill_between(ages, cum_net,           cum_net + cum_buf, color=COLOR_BUF, alpha=0.70,
                 label=f"Buffer-pool reserve  ({f_buf:.0%})")
ax2.fill_between(ages, cum_net + cum_buf, cum_vals,          color=COLOR_UNC, alpha=0.50,
                 label=f"Harvest / safety margin  (UNC = {f_unc:.0%})")
ax2.plot(ages, cum_vals, color="0.20", lw=1.3, label="Gross cumulative stock")

ax2.axvline(T_rot,  color="0.40", lw=0.85, ls=":")
ax2.axvline(T_CRCF, color="0.60", lw=0.85, ls="--")

# annotate harvest margin at T_rot
harvest_Trot  = f_unc * CUM_T_rot
mid_unc_y     = (f_net + f_buf) * CUM_T_rot + harvest_Trot / 2
ax2.annotate(
    f"Harvest margin\n$\\approx{harvest_Trot:.0f}$ tCO$_2$/ha",
    xy=(T_rot, mid_unc_y),
    xytext=(T_rot + 25, mid_unc_y * 1.0),
    arrowprops=dict(arrowstyle="->", color="0.35", lw=0.8),
    fontsize=7.5, color="0.25", va="center",
)

ax2.text(T_rot  + 2, CUM_T_rot * 0.06, f"$T_{{\\rm rot}}={T_rot:.0f}$ yr",
         fontsize=7, color="0.40", va="bottom")
ax2.text(T_CRCF + 2, CUM_T_rot * 0.06, f"$T_{{\\rm CRCF}}=30$ yr",
         fontsize=7, color="0.55", va="bottom")

ax2.set_xlabel("Stand age (years)")
ax2.set_ylabel("Cumulative carbon stock (tCO$_2$ ha$^{-1}$)")
ax2.set_title("(b)  Certified stock decomposition  ($\\eta_\\mathrm{CRCF}=0.80$, central estimate)")
ax2.set_xlim(0, 155)
ax2.set_ylim(0)
ax2.xaxis.set_minor_locator(MultipleLocator(10))
ax2.legend(loc="upper left", frameon=False, handlelength=1.4)

# ─────────────────────────────────────────────────────────────────────────────
# Panel C – bar chart decomposition
# ─────────────────────────────────────────────────────────────────────────────
labels = list(scenarios.keys())
x      = np.arange(len(labels))
bar_w  = 0.52

nets = [s["net"] for s in scenarios.values()]
bufs = [s["buf"] for s in scenarios.values()]
uncs = [s["unc"] for s in scenarios.values()]

ax3.bar(x, nets, bar_w, color=COLOR_NET, label="Net certified ($\\eta_\\mathrm{CRCF}$)")
ax3.bar(x, bufs, bar_w, bottom=nets,
        color=COLOR_BUF, alpha=0.85, label="Buffer pool")
ax3.bar(x, uncs, bar_w, bottom=[n + b for n, b in zip(nets, bufs)],
        color=COLOR_UNC, alpha=0.80, label="Harvest margin (UNC)")

for i, (net, buf, unc) in enumerate(zip(nets, bufs, uncs)):
    ax3.text(i, net / 2,           f"{net:.2f}", ha="center", va="center",
             fontsize=8, color="white", fontweight="bold")
    ax3.text(i, net + buf / 2,     f"{buf:.2f}", ha="center", va="center",
             fontsize=8, color="white", fontweight="bold")
    ax3.text(i, net + buf + unc/2, f"{unc:.2f}", ha="center", va="center",
             fontsize=8, color="white", fontweight="bold")

ax3.axhline(1.0, color="0.4", lw=0.9, ls="--")
ax3.text(2.35, 1.013, "Gross = 1", fontsize=7, color="0.4", va="bottom")

ax3.set_xticks(x)
ax3.set_xticklabels(labels, fontsize=8)
ax3.set_ylim(0, 1.13)
ax3.set_ylabel("Fraction of gross MAI")
ax3.set_title("(c)  Credit decomposition\nby disturbance scenario")
ax3.yaxis.set_minor_locator(MultipleLocator(0.05))
ax3.legend(loc="lower left", frameon=False, handlelength=1.2)

# ── save ──────────────────────────────────────────────────────────────────────
outdir = os.path.dirname(os.path.abspath(__file__))
for ext in ("pdf", "png"):
    path = os.path.join(outdir, f"mai_crcf_explained.{ext}")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"Saved: {path}")

plt.show()
