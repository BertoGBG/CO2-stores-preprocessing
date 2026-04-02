# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
# SPDX-License-Identifier: MIT

"""
Compute perennial crop and 1st-generation biofuel yield conversions per NUTS2 region.

Requires the raw Eurostat CSV files downloaded by download_eurostat_crops.py and
the NUTS2 2021 GeoJSON (NUTS_RG_01M_2021_4326_LEVL_2.geojson).

Mapping logic:
- Compute area-weighted average yields from Eurostat apro_cpshr (2017-2020)
- Map crop codes to 1G biofuel categories (ENSPRESO convention)
- Harmonize to NUTS2 2021 with spatial fallbacks for missing regions

UNITS:
- 1G biofuel yields: MWh/ha
- Perennial grass yields: t/ha (dry matter, std moisture 65 %)

Outputs (all in zenodo_w_processing/data/perennialisation/)
-------
- yields_perennials_1G_biofuels.csv      (all categories, one column each)
- yields_MINBIOCRP11_nuts2.csv           (cereals -> bioethanol)
- yields_MINBIOCRP21_nuts2.csv           (sugar beet -> bioethanol)
- yields_MINBIORPS1_nuts2.csv            (rape/sunflower/soy -> biodiesel)
- yields_perennials_nuts2.csv            (perennial grasses, weighted mean)
- yields_perennials_max_nuts2.csv        (perennial grasses, regional max)
"""

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
logger = logging.getLogger(__name__)


# ---- helpers ----

def harmonize_to_nuts2021(df, keep_col, nuts2021_n2):
    """
    Harmonize a yield DataFrame (indexed by geo codes) to the full NUTS2 2021 set.

    Falls back in order:
    1. Direct NUTS2 match
    2. NUTS0 country value
    3. Spatial neighbours (touching) mean
    4. Nearest valid NUTS2 region (for islands)
    """
    target = nuts2021_n2.index.sort_values()

    df = df.copy()
    df["is_nuts2"] = df.index.str.len() == 4

    df_nuts2 = df[df["is_nuts2"]]
    df_nuts0 = df[~df["is_nuts2"]]

    df_work = df_nuts2[[keep_col]].reindex(target)
    df_work["country"] = df_work.index.str[:2]

    df_work = df_work.join(
        df_nuts0[[keep_col]].rename(columns={keep_col: "fallback"}),
        on="country",
    )
    df_work[keep_col] = df_work[keep_col].fillna(df_work["fallback"])
    df_work.drop(columns=["fallback"], inplace=True)

    nuts_proj = nuts2021_n2.to_crs(epsg=3035)
    gdf = nuts_proj.join(df_work)[[keep_col, "country", "geometry"]]

    mask_missing = gdf[keep_col].isna() | (gdf[keep_col] <= 0)
    if mask_missing.any():
        logger.info("Spatial fallback required for %d regions", mask_missing.sum())
        valid = gdf[gdf[keep_col].notna() & (gdf[keep_col] > 0)]

        for idx in gdf[mask_missing].index:
            region = gdf.loc[idx, "geometry"]
            neigh_idxs = gdf[gdf.geometry.touches(region)].index.tolist()
            neigh_vals = gdf.loc[neigh_idxs, keep_col].dropna()
            neigh_vals = neigh_vals[neigh_vals > 0]

            if len(neigh_vals) > 0:
                gdf.at[idx, keep_col] = neigh_vals.mean()
            else:
                nearest_idx = valid.distance(region).idxmin()
                gdf.at[idx, keep_col] = valid.at[nearest_idx, keep_col]
                logger.info("Island fallback for %s: nearest = %s", idx, nearest_idx)

    result = gdf[[keep_col]]
    result.index.name = "NUTS2"
    result.sort_index()
    return result


def calculate_yields(filepath_nuts2, filepath_nuts0, crops_sel, crops_mapping, biofuel_yields):
    df_crops_raw_nuts2 = pd.read_csv(filepath_nuts2)
    df_crops_raw_nuts2["TIME_PERIOD"] = df_crops_raw_nuts2["TIME_PERIOD"].astype(int)

    df_crops_raw_nuts0 = pd.read_csv(filepath_nuts0)
    df_crops_raw_nuts0["TIME_PERIOD"] = df_crops_raw_nuts0["TIME_PERIOD"].astype(int)

    df_crops_raw = pd.concat([df_crops_raw_nuts0, df_crops_raw_nuts2], ignore_index=True)

    columns_to_drop = [
        "Observation value",
        "OBS_FLAG",
        "Observation status (Flag) V2 structure",
        "CONF_STATUS",
        "Confidentiality status (flag)",
        "Time",
        "STRUCTURE_ID",
        "STRUCTURE",
        "STRUCTURE_NAME",
        "Geopolitical entity (reporting)",
        "Time frequency",
    ]

    df_crops = df_crops_raw.drop(columns=columns_to_drop, errors="ignore")
    df_crops["OBS_VALUE"] = df_crops["OBS_VALUE"].fillna(0)

    df_sub = df_crops[
        (df_crops["strucpro"].isin(["AR", "PR_HU_EU"])) &
        (df_crops["crops"].isin(crops_sel))
    ][["crops", "geo", "TIME_PERIOD", "strucpro", "OBS_VALUE"]]

    df_pivot = (
        df_sub.pivot_table(
            index=["crops", "geo", "TIME_PERIOD"],
            columns="strucpro",
            values="OBS_VALUE",
        )
        .dropna(subset=["AR", "PR_HU_EU"])
        .reset_index()
    )

    df_pivot["YL_(t/ha)"] = np.divide(
        df_pivot["PR_HU_EU"],
        df_pivot["AR"],
        out=np.zeros_like(df_pivot["PR_HU_EU"], dtype=float),
        where=df_pivot["AR"] != 0,
    )

    df_avg_yield = (
        df_pivot
        .groupby(["crops", "geo"], as_index=False)[["AR", "PR_HU_EU", "YL_(t/ha)"]]
        .mean()
    )

    min_year = df_pivot["TIME_PERIOD"].min()
    max_year = df_pivot["TIME_PERIOD"].max()
    df_avg_yield["TIME_PERIOD"] = f"{min_year}-{max_year}"

    rev_map = {
        code: key
        for key, val in crops_mapping.items()
        for code in (val if isinstance(val, list) else [val])
    }
    df_avg_yield["mapping"] = df_avg_yield["crops"].map(rev_map)

    df_avg_yield["PR_share"] = (
        df_avg_yield["PR_HU_EU"]
        / df_avg_yield.groupby(["geo", "TIME_PERIOD", "mapping"])["PR_HU_EU"].transform("sum")
    )
    df_avg_yield["PR_share"] = df_avg_yield["PR_share"].fillna(0)

    df_avg_yield["weighted_YL_(t/ha)"] = df_avg_yield["YL_(t/ha)"] * df_avg_yield["PR_share"]

    thresholds = {
        "MINBIOCRP11": 2.0,
        "MINBIOCRP21": 50.0,
        "MINBIORPS1": 1.5,
        "PERENNIALS": 5.0,
    }
    df_avg_yield["weighted_YL_(t/ha)"] = df_avg_yield.apply(
        lambda row: row["weighted_YL_(t/ha)"]
        if row["weighted_YL_(t/ha)"] >= thresholds.get(row["mapping"], 0)
        else 0,
        axis=1,
    )

    weighted_yields = df_avg_yield.groupby(["geo", "TIME_PERIOD", "mapping"])["weighted_YL_(t/ha)"].sum()

    unsustainable_biofuels_yields = pd.DataFrame(weighted_yields)
    unsustainable_biofuels_yields = unsustainable_biofuels_yields[
        unsustainable_biofuels_yields.index.get_level_values("mapping") != "PERENNIALS"
    ]
    unsustainable_biofuels_yields["energy_yields_(MWh/ha)"] = (
        unsustainable_biofuels_yields["weighted_YL_(t/ha)"]
        * unsustainable_biofuels_yields.index.get_level_values("mapping").map(biofuel_yields)
    )

    # Standard moisture for perennials: 0.65 (tH2O/t_fresh)
    std_moist_perennials = 0.65

    perennial_yields = pd.DataFrame(weighted_yields)
    perennial_yields = perennial_yields[
        perennial_yields.index.get_level_values("mapping") == "PERENNIALS"
    ] * (1 - std_moist_perennials)

    max_yields = df_avg_yield.groupby(["geo", "TIME_PERIOD", "mapping"])["YL_(t/ha)"].max()
    perennial_yields_max = pd.DataFrame(max_yields)
    perennial_yields_max = perennial_yields_max[
        perennial_yields_max.index.get_level_values("mapping") == "PERENNIALS"
    ] * (1 - std_moist_perennials)

    return unsustainable_biofuels_yields, perennial_yields, perennial_yields_max


def main():
    ROOT_DIR = Path(__file__).resolve().parents[1]
    PROJECT_ROOT = ROOT_DIR.parent

    CROPS_CSV_NUTS2 = ROOT_DIR / "downloads" / "eurostat_apro_cpshr_nuts2_raw.csv"
    CROPS_CSV_NUTS0 = ROOT_DIR / "downloads" / "eurostat_apro_cpshr_nuts0_raw.csv"
    NUTS2_2021_GEOJSON = PROJECT_ROOT / "afforestation_perennials" / "data" / "nuts" / "NUTS_RG_01M_2021_4326_LEVL_2.geojson"

    OUT_DIR = ROOT_DIR / "outputs" / "perennialisation"
    OUT_CSV_YIELDS_ALL    = OUT_DIR / "yields_perennials_1G_biofuels.csv"
    OUT_CSV_MINBIOCRP11   = OUT_DIR / "yields_MINBIOCRP11_nuts2.csv"
    OUT_CSV_MINBIOCRP21   = OUT_DIR / "yields_MINBIOCRP21_nuts2.csv"
    OUT_CSV_MINBIORPS1    = OUT_DIR / "yields_MINBIORPS1_nuts2.csv"
    OUT_CSV_perennials    = OUT_DIR / "yields_perennials_nuts2.csv"
    OUT_CSV_perennials_max = OUT_DIR / "yields_perennials_max_nuts2.csv"

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not CROPS_CSV_NUTS2.exists() or not CROPS_CSV_NUTS0.exists():
        raise FileNotFoundError(
            "Raw Eurostat CSV files not found. Run download_eurostat_crops.py first.\n"
            f"  Expected: {CROPS_CSV_NUTS2}\n"
            f"  Expected: {CROPS_CSV_NUTS0}"
        )

    # ---- crop code definitions ----
    # Mapping of Eurostat crop codes to ENSPRESO 1G biofuel categories:
    # MINBIOCRP11  Bioethanol from starchy crops (cereals)
    # MINBIOCRP21  Bioethanol from sugar beet
    # MINBIORPS1   Biodiesel from rape seed, sunflower seed, soy
    # PERENNIALS   Perennial grasses (for green biorefining)

    perennial_codes = ["G0000", "G1000", "G2000", "G2100", "G2900"]

    crops_mapping = dict(
        MINBIOCRP11=["C0000", "C1000", "C1210", "C1300", "C1310", "C1320"],
        MINBIOCRP21="R2000",
        MINBIORPS1=["I1110", "I1120", "I1130", "I1110-1130", "I0000"],
        PERENNIALS=perennial_codes,
    )

    LHV_fuels = dict(
        ethanol=26.81 / 3.6,   # MWh/t
        biodiesel=36.7 / 3.6,  # MWh/t
    )

    # Biofuel conversion efficiencies (doi:10.2760/69179, same as CAPRI 2015):
    # MJ biofuel / MJ feedstock -> converted to MWh_biofuel / t_crop
    biofuel_yields = dict(
        MINBIOCRP11=0.295 * LHV_fuels["ethanol"],           # t_ethanol/t_wheat (13.5 % H2O), Table 93
        MINBIOCRP21=0.07777 * LHV_fuels["ethanol"],         # t_ethanol/t_SB (16 % sugar), Table 133
        MINBIORPS1=0.420 / 1.0063 * LHV_fuels["biodiesel"], # kg crude oil/kg rapeseed (9 % H2O), Tables 155+159
    )

    other_crops_codes = [item for v in crops_mapping.values() for item in (v if isinstance(v, list) else [v])]
    crops_sel = perennial_codes + other_crops_codes

    # ---- processing ----
    logger.info("Calculating yields from %s and %s", CROPS_CSV_NUTS2.name, CROPS_CSV_NUTS0.name)
    unsustainable_biofuels_yields, perennial_yields, perennial_yields_max = calculate_yields(
        filepath_nuts2=CROPS_CSV_NUTS2,
        filepath_nuts0=CROPS_CSV_NUTS0,
        crops_sel=crops_sel,
        crops_mapping=crops_mapping,
        biofuel_yields=biofuel_yields,
    )

    yield_MINBIOCRP11 = unsustainable_biofuels_yields[
        unsustainable_biofuels_yields.index.get_level_values("mapping") == "MINBIOCRP11"
    ].droplevel(["TIME_PERIOD", "mapping"])
    yield_MINBIOCRP11.index.name = "NUTS2"

    yield_MINBIOCRP21 = unsustainable_biofuels_yields[
        unsustainable_biofuels_yields.index.get_level_values("mapping") == "MINBIOCRP21"
    ].droplevel(["TIME_PERIOD", "mapping"])
    yield_MINBIOCRP21.index.name = "NUTS2"

    yield_MINBIORPS1 = unsustainable_biofuels_yields[
        unsustainable_biofuels_yields.index.get_level_values("mapping") == "MINBIORPS1"
    ].droplevel(["TIME_PERIOD", "mapping"])
    yield_MINBIORPS1.index.name = "NUTS2"

    perennial_yields = perennial_yields.droplevel(["TIME_PERIOD", "mapping"])
    perennial_yields.index.name = "NUTS2"

    perennial_yields_max = perennial_yields_max.droplevel(["TIME_PERIOD", "mapping"])
    perennial_yields_max.index.name = "NUTS2"

    # ---- harmonize to full NUTS2 2021 set ----
    logger.info("Loading NUTS2 2021 geometry from %s", NUTS2_2021_GEOJSON)
    nuts2021_n2 = (
        gpd.read_file(NUTS2_2021_GEOJSON)
        .loc[:, ["NUTS_ID", "NUTS_NAME", "CNTR_CODE", "geometry"]]
        .set_index("NUTS_ID")
    )

    yield_MINBIOCRP11_full   = harmonize_to_nuts2021(yield_MINBIOCRP11,   "energy_yields_(MWh/ha)", nuts2021_n2)
    yield_MINBIOCRP21_full   = harmonize_to_nuts2021(yield_MINBIOCRP21,   "energy_yields_(MWh/ha)", nuts2021_n2)
    yield_MINBIORPS1_full    = harmonize_to_nuts2021(yield_MINBIORPS1,    "energy_yields_(MWh/ha)", nuts2021_n2)
    yields_perennials_full   = harmonize_to_nuts2021(perennial_yields,    "weighted_YL_(t/ha)",     nuts2021_n2)
    yields_perennials_max_full = harmonize_to_nuts2021(perennial_yields_max, "YL_(t/ha)",           nuts2021_n2)

    # ---- combined output table ----
    df_yields_all = pd.concat(
        {
            "MINBIOCRP11": yield_MINBIOCRP11_full["energy_yields_(MWh/ha)"],
            "MINBIOCRP21": yield_MINBIOCRP21_full["energy_yields_(MWh/ha)"],
            "MINBIORPS1":  yield_MINBIORPS1_full["energy_yields_(MWh/ha)"],
            "PERENNIALS_MAX": yields_perennials_max_full["YL_(t/ha)"],
        },
        axis=1,
    )
    df_yields_all.columns = [
        "Bioethanol barley, wheat, grain maize, oats, other cereals and rye",
        "Sugar from sugar beet",
        "Rape seed",
        "perennials",
    ]
    df_yields_all = df_yields_all.sort_index()

    # ---- save outputs ----
    df_yields_all.to_csv(OUT_CSV_YIELDS_ALL, index=True)
    yield_MINBIOCRP11_full.to_csv(OUT_CSV_MINBIOCRP11, index=True)
    yield_MINBIOCRP21_full.to_csv(OUT_CSV_MINBIOCRP21, index=True)
    yield_MINBIORPS1_full.to_csv(OUT_CSV_MINBIORPS1, index=True)
    yields_perennials_full.to_csv(OUT_CSV_perennials, index=True)
    yields_perennials_max_full.to_csv(OUT_CSV_perennials_max, index=True)

    logger.info("Outputs written to %s", OUT_DIR)


if __name__ == "__main__":
    main()
