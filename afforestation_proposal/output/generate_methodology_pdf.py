"""Generate a PDF methodology document using matplotlib."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

OUT = Path(__file__).parent / "methodology_afforestation_pilli.pdf"

# ── Page setup ───────────────────────────────────────────────────────────────
PAGE_W, PAGE_H = 8.27, 11.69  # A4 inches
MARGIN_L, MARGIN_R = 0.9, 0.9
MARGIN_T, MARGIN_B = 0.8, 0.8
TEXT_W = PAGE_W - MARGIN_L - MARGIN_R

FONT_BODY = 10
FONT_HEADING = 13
FONT_SUBHEADING = 11
FONT_SMALL = 8.5
LINE_H = 0.018  # line height as fraction of page


def new_page(pdf):
    fig = plt.figure(figsize=(PAGE_W, PAGE_H))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, PAGE_W)
    ax.set_ylim(0, PAGE_H)
    ax.axis("off")
    return fig, ax


def put(ax, x, y, text, size=FONT_BODY, weight="normal", style="normal",
        family="serif", ha="left", va="top", wrap_width=None, color="black"):
    ax.text(
        x, y, text,
        fontsize=size, fontweight=weight, fontstyle=style,
        fontfamily=family, ha=ha, va=va, color=color,
        transform=ax.transData,
        clip_on=False,
    )


def write_document():
    with PdfPages(str(OUT)) as pdf:

        # ── PAGE 1 ───────────────────────────────────────────────────────
        fig, ax = new_page(pdf)
        x0 = MARGIN_L
        y = PAGE_H - MARGIN_T

        put(ax, x0, y, "Afforestation CO\u2082 Sequestration Rates for PyPSA-Eur:",
            size=15, weight="bold")
        y -= 0.35
        put(ax, x0, y, "Rotation-Averaged MAI from Pilli et al. (2024)",
            size=15, weight="bold")
        y -= 0.45
        put(ax, x0, y, "Methodology Description", size=12, style="italic", color="gray")
        y -= 0.6

        # ── Section 1
        put(ax, x0, y, "1  Data Source", size=FONT_HEADING, weight="bold")
        y -= 0.30

        body1 = (
            "The sequestration rates are derived from the JRC forest growth library compiled by "
            "Pilli et al. (2024, Zenodo), which provides age-class-resolved volume and increment "
            "curves for 222 forest types across 25 EU Member States (EU-27 excl. Cyprus and Malta). "
            "The library contains three key datasets:"
        )
        t = ax.text(x0, y, body1, fontsize=FONT_BODY, fontfamily="serif",
                    va="top", ha="left", wrap=True)
        t._get_wrap_line_width = lambda: TEXT_W * 72  # points
        y -= 0.75

        items = [
            "\u2022  Standing stock (m\u00b3/ha): net merchantable volume under bark, by 10-year age class.",
            "\u2022  Net Annual Increment (NAI, m\u00b3/ha/yr): merchantable volume growth rate by age class,\n"
            "    harmonised across countries using species-specific correction factors.",
            "\u2022  Biomass Conversion and Expansion Factors (BCEF, t_DM/m\u00b3): age-class-resolved\n"
            "    factors converting merchantable volume to total aboveground dry biomass.",
        ]
        for item in items:
            t = ax.text(x0 + 0.15, y, item, fontsize=FONT_BODY, fontfamily="serif",
                        va="top", ha="left")
            y -= 0.35

        y -= 0.10
        body1b = (
            "The data are classified by country, administrative region (mostly NUTS-2), forest type "
            "(leading species), management type, and management strategy (even-aged or uneven-aged). "
            "Original NFI reference years span 1992\u20132018 (mostly 2005\u20132015)."
        )
        t = ax.text(x0, y, body1b, fontsize=FONT_BODY, fontfamily="serif",
                    va="top", ha="left", wrap=True)
        t._get_wrap_line_width = lambda: TEXT_W * 72
        y -= 0.55

        # ── Section 2
        put(ax, x0, y, "2  Rotation-Averaged Sequestration Rate", size=FONT_HEADING, weight="bold")
        y -= 0.30

        body2 = (
            "For new afforestation, we compute a rotation-averaged Mean Annual Increment (MAI), "
            "representing the long-term average CO\u2082 uptake over a full forest rotation. This "
            "approach is consistent with the JRC Carbon Budget Model (EU-CBM-HAT) methodology "
            "(Pilli et al. 2024, Grassi et al. 2018)."
        )
        t = ax.text(x0, y, body2, fontsize=FONT_BODY, fontfamily="serif",
                    va="top", ha="left", wrap=True)
        t._get_wrap_line_width = lambda: TEXT_W * 72
        y -= 0.55

        put(ax, x0, y, "Step 1: Determine optimal rotation age", size=FONT_SUBHEADING, weight="bold")
        y -= 0.25
        step1 = (
            "The rotation age T* is defined as the age at which the volumetric MAI is maximised:\n\n"
            "    T* = argmax(T \u2265 T_min)  [ V(T) / T ]\n\n"
            "where V(T) is the standing stock (m\u00b3/ha) at age T and T_min = 25 yr is a minimum "
            "rotation age imposed to avoid numerical artefacts at very young age classes. Age classes "
            "are reported at 10-year intervals with midpoints at 5, 15, 25, ..., 195 yr."
        )
        t = ax.text(x0, y, step1, fontsize=FONT_BODY, fontfamily="serif",
                    va="top", ha="left", wrap=True)
        t._get_wrap_line_width = lambda: TEXT_W * 72
        y -= 0.90

        put(ax, x0, y, "Step 2: Compute volumetric MAI", size=FONT_SUBHEADING, weight="bold")
        y -= 0.25
        step2 = "    MAI_vol = V(T*) / T*    [m\u00b3/ha/yr]"
        put(ax, x0, y, step2, size=FONT_BODY, family="monospace")
        y -= 0.30

        put(ax, x0, y, "Step 3: Convert volume to CO\u2082", size=FONT_SUBHEADING, weight="bold")
        y -= 0.25
        step3a = "    MAI_CO\u2082 = MAI_vol \u00d7 BCEF(T*) \u00d7 C_f \u00d7 (M_CO\u2082 / M_C) \u00d7 (1 + R)    [tCO\u2082/ha/yr]"
        put(ax, x0, y, step3a, size=FONT_BODY, family="monospace")
        y -= 0.35

        params = (
            "where:\n"
            "    BCEF(T*) = Biomass Conversion and Expansion Factor at rotation age (t_DM/m\u00b3)\n"
            "    C_f = 0.5 = carbon fraction of dry biomass (IPCC 2006)\n"
            "    M_CO\u2082/M_C = 44/12 \u2248 3.667 = molecular mass ratio CO\u2082 to C\n"
            "    R = 0.25 = root-to-shoot ratio for belowground biomass (IPCC 2006, Mokany et al. 2006)"
        )
        t = ax.text(x0, y, params, fontsize=FONT_SMALL, fontfamily="serif",
                    va="top", ha="left")
        y -= 0.75

        put(ax, x0, y, "Step 4: Regional aggregation", size=FONT_SUBHEADING, weight="bold")
        y -= 0.25
        step4 = (
            "For each NUTS-2 region r, the rate is the equal-weight average across all productive "
            "forest types present in that region:\n\n"
            "    MAI_CO\u2082,r  =  (1 / |F_r|)  \u00d7  \u03a3  MAI_CO\u2082,f,r     for f \u2208 F_r\n\n"
            "Only even-aged, productive high-forest stands are included (management types H, HP, HS, MAN); "
            "coppice, non-productive, and protected forests are excluded."
        )
        t = ax.text(x0, y, step4, fontsize=FONT_BODY, fontfamily="serif",
                    va="top", ha="left", wrap=True)
        t._get_wrap_line_width = lambda: TEXT_W * 72

        pdf.savefig(fig)
        plt.close(fig)

        # ── PAGE 2 ───────────────────────────────────────────────────────
        fig, ax = new_page(pdf)
        y = PAGE_H - MARGIN_T

        put(ax, x0, y, "3  Country-Level Data Handling", size=FONT_HEADING, weight="bold")
        y -= 0.30
        body3 = (
            "For countries where Pilli et al. data are at NUTS-0 level (BG, EE, GR, HR, HU, IE, LT, LU, "
            "LV, NL, SI, SK), the national rate is applied to all NUTS-2 regions within that country. "
            "For countries with non-standard regions (DE, FI, IT, PL), a mapping to NUTS-2 codes is "
            "applied using the correspondence tables provided in the dataset."
        )
        t = ax.text(x0, y, body3, fontsize=FONT_BODY, fontfamily="serif",
                    va="top", ha="left", wrap=True)
        t._get_wrap_line_width = lambda: TEXT_W * 72
        y -= 0.65

        put(ax, x0, y, "4  Consistency with EU Policy Frameworks", size=FONT_HEADING, weight="bold")
        y -= 0.30
        body4 = (
            "The rotation-averaged MAI is consistent with the EU LULUCF Regulation (2018/841, amended "
            "2023/839), which requires gross-net accounting during a 20\u201330 year transition period for "
            "afforested land. The use of rotation-averaged (rather than peak-phase) rates avoids "
            "overestimating CDR potential and is compatible with the EU Carbon Removal Certification "
            "Framework (CRCF, 2024/3012) monitoring requirements."
        )
        t = ax.text(x0, y, body4, fontsize=FONT_BODY, fontfamily="serif",
                    va="top", ha="left", wrap=True)
        t._get_wrap_line_width = lambda: TEXT_W * 72
        y -= 0.75

        put(ax, x0, y, "5  Integration in PyPSA-Eur", size=FONT_HEADING, weight="bold")
        y -= 0.30
        body5 = (
            "Afforestation is represented as a Store component on a dedicated CO\u2082 bus at each "
            "network node, connected to the co2_atmosphere bus via a Link. The maximum storage capacity "
            "(e_nom_max) is determined by available CORINE land area, sequestration rate, and a "
            "configurable maximum land-usage fraction. Capital cost is computed from establishment and "
            "maintenance costs, annualised over a configurable lifetime (default 30 yr, consistent with "
            "the LULUCF transition period)."
        )
        t = ax.text(x0, y, body5, fontsize=FONT_BODY, fontfamily="serif",
                    va="top", ha="left", wrap=True)
        t._get_wrap_line_width = lambda: TEXT_W * 72
        y -= 0.75

        # ── Section 6: Results
        put(ax, x0, y, "6  Results and Validation", size=FONT_HEADING, weight="bold")
        y -= 0.30
        body6 = (
            "The computed rates cover 23 EU countries and 114 NUTS-2 regions (936 forest-type \u00d7 "
            "region combinations). Key statistics:"
        )
        t = ax.text(x0, y, body6, fontsize=FONT_BODY, fontfamily="serif",
                    va="top", ha="left", wrap=True)
        t._get_wrap_line_width = lambda: TEXT_W * 72
        y -= 0.40

        stats = (
            "    European median:    5.2 tCO\u2082/ha/yr\n"
            "    European mean:      5.5 tCO\u2082/ha/yr\n"
            "    Minimum region:     2.1 tCO\u2082/ha/yr  (IT \u2013 Marche)\n"
            "    Maximum region:    12.6 tCO\u2082/ha/yr  (IT \u2013 Campania)\n"
            "    Expected range:     5\u201315  tCO\u2082/ha/yr  (Nabuurs et al. 2013, Avitabile et al. 2024)"
        )
        put(ax, x0, y, stats, size=FONT_BODY, family="monospace")
        y -= 0.80

        # ── Table
        put(ax, x0, y, "Table 1: Rotation-averaged CO\u2082 sequestration rates for selected NUTS-2 regions",
            size=FONT_SMALL, weight="bold")
        y -= 0.20

        col_x = [x0, x0 + 1.0, x0 + 2.0, x0 + 4.2, x0 + 5.2]
        headers = ["Country", "NUTS-2", "Rate [tCO\u2082/ha/yr]", "Rot. [yr]", "n"]
        for cx, h in zip(col_x, headers):
            put(ax, cx, y, h, size=FONT_SMALL, weight="bold")
        y -= 0.05
        ax.plot([x0, x0 + 6.0], [y, y], "k-", lw=0.8)
        y -= 0.18

        rows = [
            ("Austria",  "AT21", "6.8",  "47", "8"),
            ("Belgium",  "BE20", "6.9",  "34", "7"),
            ("Czechia",  "CZ01", "5.2",  "48", "4"),
            ("Denmark",  "DK01", "5.8",  "39", "7"),
            ("Denmark",  "DK05", "3.9",  "39", "7"),
            ("Finland",  "FI1A", "2.5",  "25", "3"),
            ("France",   "FRF1", "5.4",  "27", "6"),
            ("Germany",  "DE2",  "7.2",  "34", "8"),
            ("Ireland",  "IE",   "7.1",  "33", "6"),
            ("Italy",    "ITH5", "5.6",  "40", "14"),
            ("Poland",   "SL",   "5.9",  "31", "8"),
            ("Romania",  "RO21", "6.0",  "28", "9"),
            ("Sweden",   "SE11", "4.3", "100", "4"),
            ("Sweden",   "SE33", "2.1",  "33", "4"),
        ]
        for row in rows:
            for cx, val in zip(col_x, row):
                put(ax, cx, y, val, size=FONT_SMALL)
            y -= 0.19

        ax.plot([x0, x0 + 6.0], [y + 0.08, y + 0.08], "k-", lw=0.8)
        y -= 0.35

        # Denmark case study
        put(ax, x0, y, "Denmark case study:", size=FONT_BODY, weight="bold")
        y -= 0.25
        dk_text = (
            "The five Danish NUTS-2 regions yield rates of 3.9\u20135.8 tCO\u2082/ha/yr, with a national "
            "average of ~5.1 tCO\u2082/ha/yr. This agrees with the ~6.5 tCO\u2082/ha/yr estimated for a "
            "60% conifer / 40% broadleaf mix (Fernandes et al. 2026); the difference is due to "
            "equal-weight species averaging vs. area-weighted averaging that gives more weight to "
            "higher-productivity conifer species."
        )
        t = ax.text(x0, y, dk_text, fontsize=FONT_BODY, fontfamily="serif",
                    va="top", ha="left", wrap=True)
        t._get_wrap_line_width = lambda: TEXT_W * 72

        pdf.savefig(fig)
        plt.close(fig)

        # ── PAGE 3: References ───────────────────────────────────────────
        fig, ax = new_page(pdf)
        y = PAGE_H - MARGIN_T

        put(ax, x0, y, "7  Key Assumptions and Parameters", size=FONT_HEADING, weight="bold")
        y -= 0.30

        assumptions = (
            "Parameter                          Value         Source\n"
            "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
            "Min. rotation age (T_min)          25 yr         Methodological choice\n"
            "Carbon fraction (C_f)              0.50          IPCC 2006, Vol. 4\n"
            "CO\u2082/C molecular ratio              3.667         Stoichiometric\n"
            "Root-to-shoot ratio (R)            0.25          IPCC 2006; Mokany et al. 2006\n"
            "Management types included          H,HP,HS,MAN   Productive high forests\n"
            "Status filter                      excl. FNAWS   Forest avail. for wood supply\n"
            "Species averaging                  Equal-weight  Could use NFI area weights\n"
            "Reversal risk buffer               Not applied   Optional: 10\u201320% (CRCF)\n"
            "Soil carbon                        Not included  Optional: +0.5\u20131.5 tCO\u2082/ha/yr\n"
        )
        put(ax, x0, y, assumptions, size=FONT_SMALL, family="monospace")
        y -= 1.8

        put(ax, x0, y, "References", size=FONT_HEADING, weight="bold")
        y -= 0.30

        refs = [
            "Avitabile, V. et al. (2024). Harmonised statistics and maps of forest biomass and\n"
            "    increment in Europe. Scientific Data 11, 274. doi:10.1038/s41597-024-03107-w",

            "Boudewyn, P. et al. (2007). Model-based, volume-to-biomass conversion for forested\n"
            "    and vegetated land in Canada. NRCan, Canadian Forest Service, BC-X-411.",

            "EU (2018). Regulation 2018/841 on LULUCF greenhouse gas emissions and removals.",

            "EU (2023). Regulation 2023/839 amending Regulation 2018/841 (LULUCF amendment).",

            "EU (2024). Regulation 2024/3012 establishing a Union certification framework for\n"
            "    carbon removals (CRCF).",

            "Grassi, G. et al. (2018). Reconciling global-model estimates and country reporting\n"
            "    of anthropogenic forest CO\u2082 sinks. Nature Clim. Change 8, 914\u2013920.",

            "IPCC (2006). 2006 IPCC Guidelines for National Greenhouse Gas Inventories,\n"
            "    Vol. 4: Agriculture, Forestry and Other Land Use. IGES, Japan.",

            "Mokany, K. et al. (2006). Critical analysis of root:shoot ratios in terrestrial\n"
            "    biomes. Global Change Biology 12, 84\u201396.",

            "Nabuurs, G.-J. et al. (2013). First signs of carbon sink saturation in European\n"
            "    forest biomass. Nature Climate Change 3, 792\u2013796.",

            "Pilli, R., Blujdea, V. & Rougieux, P. (2024). Forest aboveground biomass and\n"
            "    volume increment library for EU-27. Zenodo. doi:10.5281/zenodo.10214062",
        ]
        for ref in refs:
            t = ax.text(x0, y, ref, fontsize=FONT_SMALL, fontfamily="serif",
                        va="top", ha="left")
            y -= 0.40

        pdf.savefig(fig)
        plt.close(fig)

    print(f"PDF saved: {OUT}")


if __name__ == "__main__":
    write_document()
