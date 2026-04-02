# SPDX-FileCopyrightText: 2024 Alberto Alamia <alamia@dtu.dk>
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Compute afforestation growth rate and stock per NUTS2 region.
Reference: 10.2760/222407

Requires the CBM/JRC Excel file downloaded by download_afforestation_data.py.

UNITS (stored in documentation):
- Afforestation growth rate: t ha-1 y-1

Mapping logic from CBM dataset:

NUTS2 afforestation rate, fall-back strategy:
 1) Use NUTS2 value directly if available in the dataset
 2) Else, fallback to NUTS1 value if available
 3) Fill with avg NUTS2 Neighbours values
 4) Fill with NUTS0 average data
 5) UK (missing data): fill with nearest region with data
 6) Malta and Cyprus are set equal to EL43 (Crete)

Author: Alberto Alamia
Date: 2025-10-14
"""

from pathlib import Path
import pandas as pd
import geopandas as gpd
import numpy as np


avg_life_young_forest = 10  # years

# ---- helpers ----

def build_direct_mapping(df_nai: pd.DataFrame, nuts_index: pd.Index) -> pd.Series:
    value_col = df_nai.columns[-1]
    colN, colO = df_nai.columns[1], df_nai.columns[2]
    cols = [colN, colO]
    cols_sorted = sorted(cols, key=lambda c: 0 if "ISO" in c.upper() else 1)
    mapping = pd.Series(dtype=float)
    for c in cols_sorted:
        tmp = df_nai[[c, value_col]].copy()
        tmp[c] = tmp[c].astype(str)
        tmp = tmp.assign(code=tmp[c].str.split(",")).explode("code")
        tmp["code"] = tmp["code"].str.strip()
        tmp = tmp[tmp["code"].isin(nuts_index)].copy()
        if tmp.empty:
            continue
        s = tmp.drop_duplicates("code").set_index("code")[value_col]
        mapping = mapping.combine_first(s)
    return mapping


def fill_from_neighbors(row, neighbors_dict, df_affo):
    if pd.isna(row["value"]):
        vals = df_affo.loc[neighbors_dict.get(row.name, []), "value"].dropna()
        if not vals.empty:
            return vals.mean()
    return row["value"]


def main():
    ROOT_DIR = Path(__file__).resolve().parents[1]
    PROJECT_ROOT = ROOT_DIR.parent

    CBM_XLS = ROOT_DIR / "downloads" / "Biomass_calculations.xlsx"
    CBM_XLS_FALLBACK = PROJECT_ROOT / "afforestation_perennials" / "resources" / "forests" / "Biomass_calculations.xlsx"
    NUTS2_2013_GEOJSON = PROJECT_ROOT / "afforestation_perennials" / "data" / "nuts" / "NUTS_RG_03M_2013_4326_LEVL_2.geojson"
    OUT_CSV = ROOT_DIR / "outputs" / "afforestation" / "afforestation_nuts2_growth_rates.csv"

    if not CBM_XLS.exists() or CBM_XLS.stat().st_size == 0:
        if CBM_XLS_FALLBACK.exists():
            print(f"[warn] {CBM_XLS.name} not found in downloads/, using repo copy: {CBM_XLS_FALLBACK}")
            CBM_XLS = CBM_XLS_FALLBACK
        else:
            raise FileNotFoundError(
                f"CBM Excel not found: {CBM_XLS}\n"
                "Run download_afforestation_data.py first."
            )

    # 1) Load Excel (M:P) and NUTS2 2013 geometry
    df_nai = pd.read_excel(CBM_XLS, sheet_name="INPUT CBM", usecols="M:P", skiprows=1, engine="openpyxl")
    df_nai.columns = [c.strip() for c in df_nai.columns]
    value_col = df_nai.columns[-1]
    df_nai[value_col] = pd.to_numeric(df_nai[value_col], errors="coerce")

    nuts2 = gpd.read_file(str(NUTS2_2013_GEOJSON)) \
               .loc[:, ["NUTS_ID", "NUTS_NAME", "CNTR_CODE", "geometry"]] \
               .set_index("NUTS_ID")

    # 2) Direct mapping from CBM file to NUTS2 if available
    direct_map = build_direct_mapping(df_nai, nuts2.index)

    # 3) Output df
    df_affo = pd.DataFrame(index=nuts2.index, columns=["value", "Source"], data=np.nan) \
        .astype({"Source": "string"})
    df_affo.loc[direct_map.index, "value"] = direct_map.values
    df_affo.loc[direct_map.index, "Source"] = "direct"

    # 4) Fallback 1: NUTS-1 → NUTS-2 propagation (if data available only at NUTS1)
    colN, colO = df_nai.columns[1], df_nai.columns[2]
    nuts1_series = pd.Series(dtype=float)
    for c in (colN, colO):
        t = df_nai[[c, value_col]].copy()
        t[c] = t[c].astype(str).str.strip()
        t[value_col] = pd.to_numeric(t[value_col], errors="coerce")
        t = t[t[c].str.fullmatch(r"^[A-Z]{2}[A-Z0-9]$")]
        if not t.empty:
            t = t.drop_duplicates(c).set_index(c)[value_col]
            nuts1_series = nuts1_series.combine_first(t)
    for parent, val in nuts1_series.items():
        children = [rid for rid in df_affo.index if rid.startswith(parent)]
        if not children:
            continue
        needs = df_affo.loc[children, "value"].isna()
        if needs.any():
            df_affo.loc[needs.index[needs], "value"] = val
            df_affo.loc[needs.index[needs], "Source"] = f"{parent}→NUTS1 copy"

    # 5) Fallback 2: Fill with avg NUTS2 Neighbours values
    gdf = gpd.GeoDataFrame(df_affo.join(nuts2[["geometry"]]), geometry="geometry", crs=nuts2.crs)
    neighbors_dict = {}
    for rid, geom in gdf.geometry.items():
        neighbors_dict[rid] = [] if (geom is None or geom.is_empty) else gdf[gdf.geometry.touches(geom)].index.tolist()
    df_affo["value"] = df_affo.apply(lambda r: fill_from_neighbors(r, neighbors_dict, df_affo), axis=1)
    df_affo.loc[df_affo["Source"].isna() & df_affo["value"].notna(), "Source"] = "avg nuts2 near"

    # ---- UK-specific fallback (lack of data for UK and islands without direct neighbours) ----
    df_affo["NUTS0"] = df_affo.index.str[:2]
    df_affo["NUTS1"] = df_affo.index.str[:3]

    uk_mask = df_affo["NUTS0"].eq("UK")
    if uk_mask.any():
        # A) within-UK NUTS1 group mean
        uk = df_affo.loc[uk_mask].copy()
        nuts1_mean = uk.groupby("NUTS1")["value"].mean()

        needs_uk = uk["value"].isna()
        fill_from_nuts1 = needs_uk & uk["NUTS1"].isin(nuts1_mean.dropna().index)
        df_affo.loc[uk.index[fill_from_nuts1], "value"] = uk.loc[fill_from_nuts1, "NUTS1"].map(nuts1_mean)
        new_filled_idx = uk.index[fill_from_nuts1 & df_affo.loc[uk.index, "Source"].isna()]
        df_affo.loc[new_filled_idx, "Source"] = "avg UK NUTS1"

        # refresh mask for still-missing UK rows
        uk = df_affo.loc[uk_mask].copy()
        needs_uk = uk["value"].isna()

        # B) nearest within-UK (project to meters for distance)
        if needs_uk.any():
            guk = gpd.GeoDataFrame(uk.join(nuts2[["geometry"]]), geometry="geometry", crs=nuts2.crs).to_crs(3035)
            guk["centroid"] = guk.geometry.centroid

            known = guk.loc[~guk["value"].isna(), ["centroid", "value"]]
            unknown = guk.loc[needs_uk, ["centroid"]]

            if not known.empty:
                known_coords = np.vstack([known["centroid"].x.values, known["centroid"].y.values]).T
                unk_coords = np.vstack([unknown["centroid"].x.values, unknown["centroid"].y.values]).T

                best_idx = []
                for ux, uy in unk_coords:
                    dx = known_coords[:, 0] - ux
                    dy = known_coords[:, 1] - uy
                    j = np.argmin(dx * dx + dy * dy)
                    best_idx.append(known.index[j])

                assign_idx = unknown.index
                nearest_vals = known.loc[best_idx, "value"].values
                df_affo.loc[assign_idx, "value"] = nearest_vals

                tag_mask = df_affo.loc[assign_idx, "Source"].isna()
                df_affo.loc[assign_idx[tag_mask], "Source"] = "nearest within UK"

    # ---- Fallback 3: Country fallback (two-stage): mean of NUTS2 -> Excel national ----
    df_affo["NUTS0"] = df_affo.index.str[:2]

    nuts2_mean_by_country = df_affo.groupby("NUTS0")["value"].mean()

    cbm_nuts0 = pd.Series(dtype=float)
    for c in (colN, colO):
        t = df_nai[[c, value_col]].copy()
        t[c] = t[c].astype(str).str.strip()
        t[value_col] = pd.to_numeric(t[value_col], errors="coerce")
        t = t[t[c].str.fullmatch(r"^[A-Z]{2}$")]
        if not t.empty:
            t = t.drop_duplicates(c).set_index(c)[value_col]
            cbm_nuts0 = cbm_nuts0.combine_first(t)

    country_avg = nuts2_mean_by_country.combine_first(cbm_nuts0)

    needs = df_affo["value"].isna()
    df_affo.loc[needs, "value"] = df_affo.loc[needs, "NUTS0"].map(country_avg)

    from_nuts2_mean = needs & df_affo["value"].notna() & df_affo["NUTS0"].isin(nuts2_mean_by_country.dropna().index)
    df_affo.loc[from_nuts2_mean & df_affo["Source"].isna(), "Source"] = "avg nuts0 (from NUTS2)"

    from_excel_nat = needs & df_affo["value"].notna() & ~df_affo["NUTS0"].isin(nuts2_mean_by_country.dropna().index)
    df_affo.loc[from_excel_nat & df_affo["Source"].isna(), "Source"] = "avg nuts0 (from excel)"

    # ---- Special case: copy Crete (EL43) to Malta (MT00) and Cyprus (CY00) ----
    for code in ["EL43", "MT00", "CY00"]:
        if code not in df_affo.index:
            break
    else:
        ref_val = df_affo.at["EL43", "value"]
        if pd.notna(ref_val):
            targets = [c for c in ["MT00", "CY00"] if c in df_affo.index]
            if targets:
                df_affo.loc[targets, "value"] = ref_val
                needs_tag = df_affo.loc[targets, "Source"].isna()
                df_affo.loc[[t for t, n in zip(targets, needs_tag) if n], "Source"] = "EL43 copy"

    # Report + save
    left_nan = df_affo["value"].isna().sum()
    if left_nan:
        print(f"[warn] {left_nan} NUTS-2 still missing after all fills.")
    else:
        print("[ok] No missing NUTS-2 values.")

    out = df_affo.drop(columns=["NUTS0", "NUTS1"])
    out.loc[:, "value"] = out.loc[:, "value"] / avg_life_young_forest
    out.rename(columns={"value": "affo rate (t/ha/y)"}, inplace=True)
    out.index.name = "NUTS2"
    out.sort_index(inplace=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV)
    print(f"[ok] wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
