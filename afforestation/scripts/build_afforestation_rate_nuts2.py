# SPDX-FileCopyrightText: 2024 Alberto Alamia <alamia@mpe.au.dk>
# SPDX-License-Identifier: GPL-3.0-or-later


"""
Compute afforestation growth rate and stock per NUTS2 region
Reference: 10.2760/222407

Calculates the Net Annual Increment (NAI) for afforestation per NUTS2 region based on
stock-weighted average NAI for forests under 20 years of age.

UNITS (stored in documentation, not in variable names):
- Afforestation growth rate: m3 ha-1 y-1
- Stock: m3 ha-1

Mapping logic from CBM dataset:

NUTS2 afforestation rate:
 a.1) Use NUTS2 value directly if available
 a.2) Else, fallback to NUTS0 value if available
 a.3) Else, use the value from the nearest NUTS2 region (in-country, then cross-border)

NUTS2 stock:
 s.1) If NUTS2 value exists in CBM dataset, use it
 s.2) If coarse region (e.g., BE20, DE1), distribute total stock evenly to detailed NUTS2s
 s.3) If not mapped, leave stock as NaN

NUTS0 afforestation rate:
 b.1) Use weighted average of NUTS2 values (if available)
 b.2) Else, fallback to NUTS0 value if available
 b.3) Else, use global average from NUTS2 results

NUTS0 stock:
 s.1) Use sum of all available CBM stocks mapped to the country

Author: Alberto Alamia
Date: 2025-04-07
"""

import pandas as pd
import geopandas as gpd
import os
import numpy as np

#### Inputs ####
# read csv with CBM dataset
nai_file = "resources/forests/NAI_evenaged_stands.csv"  # data with Net Annual Increment for age class: NAI  (m3 ha-1 yr-1)
stock_file = "resources/forests/Standing_stock_evenaged.csv"  # data with standing stock for age class: stock  (m3 ha-1)
nuts2_nai = "resources/forests/NAI_regions_codes.csv"  # Regional codes mapping available from CBM

# shape file nuts2
nuts2_file = "data/nuts/NUTS_RG_03M_2013_4326_LEVL_2.geojson"

#### Outputs ####
output_file_nuts2 = "data/afforestation/afforestation_nuts2.csv"
output_file_nuts0 = "data/afforestation/afforestation_nuts0.csv"


######
def get_direct_region_mappings():
    """
    Returns one-to-one region mappings (region → NUTS2) for countries like PL and FI.
    """
    return {
        "PL": {
            "BT": "PL43",
            "CP": "PL61",
            "MP": "PL21",
            "MS": "PL91",
            "MZ": "PL71",
            "SD": "PL72",
            "SL": "PL22",
            "WP": "PL41",
        },
        "FI": {
            "FI_N": "FI1E",
            "FI_S": "FI1C",
        },
        "SE": "region",  # SE uses region name directly
    }


def get_manual_region_mappings():
    """
    Returns coarse-to-fine NUTS2 mappings for countries like BE, DE, FR.
    """
    return {
        "BE": {
            "BE20": ["BE21", "BE22", "BE23", "BE24", "BE25"],
            "BE30": ["BE31", "BE32", "BE33", "BE34", "BE35"],
        },
        "DE": {
            "DE1": ["DE11", "DE12", "DE13", "DE14"],
            "DE2": ["DE21", "DE22", "DE23", "DE24", "DE25", "DE26", "DE27"],
            "DE3": ["DE30"],
            "DE4": ["DE40"],
            "DE5": ["DE50"],
            "DE6": ["DE60"],
            "DE7": ["DE71", "DE72", "DE73"],
            "DE8": ["DE80"],
            "DE9": ["DE91", "DE92", "DE93", "DE94"],
            "DEA": ["DEA1", "DEA2", "DEA3", "DEA4", "DEA5"],
            "DEB": ["DEB1", "DEB2", "DEB3"],
            "DEC": ["DEC0"],
            "DED": ["DED2", "DED4", "DED5"],
            "DEE": ["DEE0"],
            "DEF": ["DEF0"],
            "DEG": ["DEG0"],
        },
        "FR": {
            "FRF1": ["FRF11", "FRF12", "FRF13"],
            "FRF3": ["FRF31", "FRF32"],
            "FRJ2": ["FRJ21", "FRJ22"],
            "FRH0": ["FRH01", "FRH02"],
            "FR10": ["FR101", "FR102"],
        },
    }


def expand_coarse_nuts2(df, mapping_dict, country_code):
    """
    Expands coarse NUTS2 codes (e.g. DE1, BE20) into multiple finer NUTS2 codes
    and distributes afforestation stock evenly across them.

    Parameters:
        df (pd.DataFrame): DataFrame with a 'NUTS2' column and afforestation data
        mapping_dict (dict): Mapping from coarse NUTS2 to list of finer NUTS2 codes
        country_code (str): ISO country code to filter for expansion (e.g., "DE")

    Returns:
        pd.DataFrame: Expanded DataFrame with updated NUTS2 and stock
    """
    df_expanded = []
    for _, row in df.iterrows():
        if row["country"] == country_code and row["NUTS2"] in mapping_dict:
            target_nuts2_list = mapping_dict[row["NUTS2"]]
            n_targets = len(target_nuts2_list)

            # Divide stock evenly across all target NUTS2 regions
            affo_stock = row["afforestation_stock"]
            divided_stock = affo_stock / n_targets if pd.notna(affo_stock) else np.nan

            for target_nuts2 in target_nuts2_list:
                row_copy = row.copy()
                row_copy["NUTS2"] = target_nuts2
                row_copy["afforestation_stock"] = divided_stock
                df_expanded.append(row_copy)
        else:
            df_expanded.append(row)

    return pd.DataFrame(df_expanded)


def map_afforestation_rate(df_valid, nuts2_gdf, nuts_lookup):
    # Weighted average by NUTS2
    nuts2_weighted_avg = (
        df_valid.groupby("NUTS2")[["affo_rate", "affo_stock"]]
        .apply(lambda g: np.average(g["affo_rate"], weights=g["affo_stock"]))
        .to_dict()
    )

    # Weighted average by NUTS0
    nuts0_weighted_avg = (
        df_valid.groupby("country")[["affo_rate", "affo_stock"]]
        .apply(lambda g: np.average(g["affo_rate"], weights=g["affo_stock"]))
        .to_dict()
    )

    return nuts2_weighted_avg, nuts0_weighted_avg


def map_stock(df_valid, df_aff_rate):
    # Total stock per NUTS2
    nuts2_stock = df_valid.groupby("NUTS2")["affo_stock"].sum().to_dict()

    # Total stock per NUTS0
    nuts0_stock = (
        df_aff_rate.dropna(subset=["country", "affo_stock"])
        .groupby("country")["affo_stock"]
        .sum()
        .to_dict()
    )

    return nuts2_stock, nuts0_stock


def calculate_afforestation_rate(nai_file, stock_file, nuts2_nai):
    # read csv with CBM dataset
    df_nai = pd.read_csv(nai_file)
    df_stock = pd.read_csv(stock_file)
    df_nuts2_nai = pd.read_csv(nuts2_nai)

    # Filter for forest age classes <= 20 years (AgeCL_1, AgeCL_2)
    columns_to_keep = [
        "country",
        "status",
        "forest_type",
        "region",
        "mgmt_type",
        "mgmt_strategy",
        "con_broad",
        "AgeCL_1",
        "AgeCL_2",
    ]

    # clean up the df
    df_nai = df_nai[columns_to_keep]
    df_nai_clean = df_nai.dropna(subset=["country", "region"])
    df_nai = df_nai.reset_index(drop=True)

    df_stock = df_stock[columns_to_keep]
    df_stock = df_stock.dropna(subset=["country", "region"])
    df_stock = df_stock.reset_index(drop=True)

    # check that df_nai and df_stock can be matched:
    df_nai_key_counts = df_nai.groupby(
        ["country", "status", "forest_type", "region"]
    ).size()
    df_stock_key_counts = df_stock.groupby(
        ["country", "status", "forest_type", "region"]
    ).size()

    # check dataframes compatibility
    # if len(df_nai_key_counts) == len(df_stock_key_counts):
    #    print("df_nai and df_stock can be matched")
    # else:
    #    print("WARNING: df_nai and df_stock can be matched")

    ####### Calculate afforestation growth rate average stock-weighted (m3 ha-1 y-1)
    # Step 1: Prepare reduced df_stock
    df_stock_reduced = df_stock[
        ["country", "status", "forest_type", "region", "AgeCL_1", "AgeCL_2"]
    ].copy()

    # Step 2: Rename stock columns for clarity
    df_stock_reduced = df_stock_reduced.rename(
        columns={"AgeCL_1": "Stock_AgeCL_1", "AgeCL_2": "Stock_AgeCL_2"}
    )

    # Step 3: Merge with df_nai (keeping df_nai’s AgeCLs intact)
    df_aff_rate = df_nai.merge(
        df_stock_reduced, on=["country", "status", "forest_type", "region"], how="left"
    )

    # Step 4: Convert all involved columns to numeric
    for col in ["AgeCL_1", "AgeCL_2", "Stock_AgeCL_1", "Stock_AgeCL_2"]:
        df_aff_rate[col] = pd.to_numeric(df_aff_rate[col], errors="coerce")

    # Step 5: Calculate afforestation_rate
    denominator = df_aff_rate["Stock_AgeCL_1"] + df_aff_rate["Stock_AgeCL_2"]
    df_aff_rate["afforestation_stock"] = denominator

    df_aff_rate["afforestation_rate"] = (
        df_aff_rate["AgeCL_1"] * df_aff_rate["Stock_AgeCL_1"] / denominator
        + df_aff_rate["AgeCL_2"] * df_aff_rate["Stock_AgeCL_2"] / denominator
    )

    ######## Add NUTS2 from the existing CBM mapping
    df_aff_rate["NUTS2"] = None

    # Drop rows in df_nai with missing country/region
    df_aff_rate = df_aff_rate.dropna(subset=["country", "region"])

    # Build lookup table from df_nuts2_nai (and sort index for performance)
    nuts2_lookup = df_nuts2_nai.set_index(["Country", "Regions code"]).sort_index()

    # Iterate through the cleaned DataFrame
    for idx, row in df_aff_rate.iterrows():
        country = row["country"]
        region = row["region"]

        try:
            matched_rows = nuts2_lookup.loc[(country, region)]

            # If multiple rows are returned, take the first one
            if isinstance(matched_rows, pd.DataFrame):
                matched_row = matched_rows.iloc[0]
            else:
                matched_row = matched_rows

            note = matched_row["Note"]
            correspondence = matched_row[
                "Correspondence with NUTS classification system (if possible)"
            ]

            if note == "NUTS 2 level":
                df_aff_rate.at[idx, "NUTS2"] = region
            elif note == "Classification system adapted from NFI data" and pd.notna(
                correspondence
            ):
                df_aff_rate.at[idx, "NUTS2"] = correspondence
            else:
                pass  # Leave as None for manual handling

        except KeyError:
            continue  # Skip rows with no match

    # --- Apply one-to-one region mappings (PL, FI, SE) ---
    direct_mappings = get_direct_region_mappings()
    for country_code, mapping in direct_mappings.items():
        mask = df_aff_rate["country"] == country_code
        if mapping == "region":
            df_aff_rate.loc[mask, "NUTS2"] = df_aff_rate.loc[mask, "region"]
        else:
            df_aff_rate.loc[mask, "NUTS2"] = df_aff_rate.loc[mask, "region"].map(
                mapping
            )

    # --- Apply coarse-to-fine expansions (BE, DE, FR) ---
    coarse_mappings = get_manual_region_mappings()
    for country_code, mapping_dict in coarse_mappings.items():
        df_aff_rate = expand_coarse_nuts2(df_aff_rate, mapping_dict, country_code)

    return df_aff_rate


def map_afforestation_to_nuts2(df_aff_rate, nuts2_file):
    # Mapping to pypsa-eur NUTS2
    # Load NUTS2 geometry and set index
    nuts2_gdf = gpd.read_file(nuts2_file)
    nuts2_gdf = nuts2_gdf[["NUTS_ID", "NUTS_NAME", "CNTR_CODE", "geometry"]]
    nuts2_gdf = nuts2_gdf.set_index("NUTS_ID")
    nuts_lookup = nuts2_gdf[["NUTS_NAME", "CNTR_CODE"]]

    cols_affo_nuts2 = [
        "Country",
        "afforestation growth rate",
        "AgeCL (max)",
        "Forest type",
        "stock",
    ]
    df_affo_nuts2 = pd.DataFrame(
        data=None, columns=cols_affo_nuts2, index=nuts_lookup.index
    )
    df_affo_nuts2["AgeCL (max)"] = 20
    df_affo_nuts2["Forest type"] = "all"
    df_affo_nuts2["Country"] = nuts_lookup["CNTR_CODE"]

    # ------------------------------
    # Step 1: Weighted averages
    # ------------------------------

    # Remove rows with missing key data
    df_valid = df_aff_rate.dropna(
        subset=["NUTS2", "country", "afforestation_rate", "afforestation_stock"]
    )

    # Map afforestation rates to regions
    nuts2_weighted_avg, nuts0_weighted_avg = map_afforestation_rate(
        df_valid, nuts2_gdf, nuts_lookup
    )

    # Map stock to regions
    nuts2_stock, nuts0_stock = map_stock(df_valid, df_aff_rate)

    # ------------------------------
    # Step 2: Assign values nuts2 regions
    # ------------------------------

    for nuts2_code in df_affo_nuts2.index:
        # STEP 1: Direct match
        if nuts2_code in nuts2_weighted_avg:
            df_affo_nuts2.at[nuts2_code, "afforestation growth rate"] = (
                nuts2_weighted_avg[nuts2_code]
            )
            df_affo_nuts2.at[nuts2_code, "stock"] = nuts2_stock.get(nuts2_code, np.nan)
            df_affo_nuts2.at[nuts2_code, "Country"] = nuts_lookup.loc[
                nuts2_code, "CNTR_CODE"
            ]
            continue

        # STEP 2: Country-level fallback
        country_code = nuts_lookup.loc[nuts2_code, "CNTR_CODE"]
        if country_code in nuts0_weighted_avg:
            df_affo_nuts2.at[nuts2_code, "afforestation growth rate"] = (
                nuts0_weighted_avg[country_code]
            )
            df_affo_nuts2.at[nuts2_code, "Country"] = country_code
            continue

    # ------------------------------
    # Step 3: Spatial fallback for missing regions
    # ------------------------------

    # Find missing entries
    missing_idx = df_affo_nuts2[df_affo_nuts2["afforestation growth rate"].isna()].index

    # Prepare spatial GeoDataFrame with valid data
    nuts2_with_rates = nuts2_gdf.copy()
    nuts2_with_rates["affo_rate"] = nuts2_with_rates.index.map(nuts2_weighted_avg)
    nuts2_with_data = nuts2_with_rates.dropna(subset=["affo_rate"])

    # use UTM metric CRS
    nuts2_gdf = nuts2_gdf.to_crs(epsg=3035)
    nuts2_with_data = nuts2_with_data.to_crs(epsg=3035)

    for idx in missing_idx:
        target_geom = nuts2_gdf.loc[idx, "geometry"]
        target_country = nuts2_gdf.loc[idx, "CNTR_CODE"]

        # Try nearest within country
        subset = nuts2_with_data[nuts2_with_data["CNTR_CODE"] == target_country]
        if subset.empty:
            subset = nuts2_with_data  # fallback to any in Europe

        # Compute distance to all candidates
        subset = subset.copy()
        subset["distance"] = subset.geometry.distance(target_geom)
        nearest_row = subset.loc[subset["distance"].idxmin()]

        # Assign from nearest
        df_affo_nuts2.at[idx, "afforestation growth rate"] = nearest_row["affo_rate"]
        df_affo_nuts2.at[idx, "stock"] = np.nan
        df_affo_nuts2.at[idx, "Country"] = target_country

    # Sort by NUTS2
    df_affo_nuts2 = df_affo_nuts2.sort_index()

    # ------------------------------
    # Step 4: Create df_affo_nuts0
    # ------------------------------
    nuts0_codes = nuts2_gdf["CNTR_CODE"].unique()
    df_affo_nuts0 = pd.DataFrame(index=nuts0_codes)
    df_affo_nuts0["afforestation growth rate"] = np.nan
    df_affo_nuts0["AgeCL (max)"] = 20
    df_affo_nuts0["Forest type"] = "all"

    # Global fallback: average of all filled-in NUTS2 values (after spatial fallback)
    fallback_avg = df_affo_nuts2["afforestation growth rate"].dropna().mean()

    for country in df_affo_nuts0.index:
        # b.1) If the country has valid NUTS2-level entries in df_affo_nuts2 (i.e., from CBM)
        valid_nuts2 = df_affo_nuts2[
            (df_affo_nuts2["Country"] == country)
            & (df_affo_nuts2["stock"].notna())
            & (df_affo_nuts2["afforestation growth rate"].notna())
        ]

        if not valid_nuts2.empty:
            # Weighted average of NUTS2 growth rates, weighted by their stock
            rate = np.average(
                valid_nuts2["afforestation growth rate"],
                weights=valid_nuts2["stock"],
            )
            df_affo_nuts0.at[country, "afforestation growth rate"] = rate
        elif country in nuts0_weighted_avg:
            # b.2) Fallback to NUTS0-level CBM data if available
            df_affo_nuts0.at[country, "afforestation growth rate"] = nuts0_weighted_avg[
                country
            ]
        else:
            # b.3) Global fallback
            df_affo_nuts0.at[country, "afforestation growth rate"] = fallback_avg

        # For the stock value: always take it from df_aff_rate summary (nuts0_stock), if available
        df_affo_nuts0.at[country, "stock"] = nuts0_stock.get(country, np.nan)

    # Sort by NUTS0
    df_affo_nuts0 = df_affo_nuts0.sort_index()

    return df_affo_nuts2, df_affo_nuts0


###### Main ######

df_aff_rate = calculate_afforestation_rate(nai_file, stock_file, nuts2_nai)
df_affo_nuts2, df_affo_nuts0 = map_afforestation_to_nuts2(df_aff_rate, nuts2_file)

# Ensure the parent folder exists
os.makedirs(os.path.dirname(output_file_nuts2), exist_ok=True)

# Save the DataFrame
df_affo_nuts2.to_csv(output_file_nuts2)
df_affo_nuts0.to_csv(output_file_nuts0)
df_aff_rate.to_csv("data/afforestation/afforestation_cbm.csv")
