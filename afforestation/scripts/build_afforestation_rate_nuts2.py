# SPDX-FileCopyrightText: 2024 Alberto Alamia <your.email@domain.com>
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Compute afforestation growth rate per NUTS2 region
reference: 10.2760/222407

Author: Alberto Alamia
Date: 2025-04-07
"""


import pandas as pd
import geopandas as gpd
import os
import numpy as np

# function that
# Automatically download the afforestaion database from CBM JRC - DOI:  10.2760/222407
# calcualtes NAI for afforestation per NUTS2 region from: based on stock-weighted average NAI for forests with <20 years
#
# saves the file NAI_evenaged_stands.csv to resources

# read csv with CBM
df_nai = pd.read_csv(
    "resources/forests/NAI_evenaged_stands.csv"
)  # data with Net Annual Increment for age class: NAI  (m3 ha-1 yr-1)
df_stock = pd.read_csv(
    "resources/forests/Standing_stock_evenaged.csv"
)  # data with standing stock for age class: stock  (m3 ha-1)
df_nuts2_nai = pd.read_csv(
    "resources/forests/NAI_regions_codes.csv"
)  # Regional codes mapping available from CBM

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
#if len(df_nai_key_counts) == len(df_stock_key_counts):
#    print("df_nai and df_stock can be matched")
#else:
#    print("WARNING: df_nai and df_stock can be matched")

####### Calculate afforestation rate stock weighted average (m3 ha-1 y-1)
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

# manual mapping
pl_region_mapping = {
    "BT": "PL43",
    "CP": "PL61",
    "MP": "PL21",
    "MS": "PL91",  # or 'PL92' for Warsaw
    "MZ": "PL71",
    "SD": "PL72",
    "SL": "PL22",
    "WP": "PL41",
}

fi_region_mapping = {"FI_N": "FI1E", "FI_S": "FI1C"}

# adjust data mask for PL and SE
mask = df_aff_rate["country"] == "PL"
df_aff_rate.loc[mask, "NUTS2"] = df_aff_rate.loc[mask, "region"].map(pl_region_mapping)
mask = df_aff_rate["country"] == "FI"
df_aff_rate.loc[mask, "NUTS2"] = df_aff_rate.loc[mask, "region"].map(fi_region_mapping)
mask = df_aff_rate["country"] == "SE"
df_aff_rate.loc[mask, "NUTS2"] = df_aff_rate.loc[mask, "region"]

# created nuts2 df with afforestation rates in all nuts2 regions
# criterias:
# for each NUTS2 region:
# if NUTS2 region exists in CBM database use that value
# if NUTS2 does not exist but NUTS0 exist use the NUTS0 value for all NUTS2 regions in a country
# if NUTS2 and NUTS0 do not exist take the value from the closest NUTS2 region within teh coutnry and then to teh closest overall

# Mapping to pypsa-eur NUTS2
# Load NUTS2 geometry and set index
nuts2_gdf = gpd.read_file("data/nuts/NUTS_RG_01M_2021_4326_LEVL_2.geojson")
nuts2_gdf = nuts2_gdf[["NUTS_ID", "NUTS_NAME", "CNTR_CODE", "geometry"]]
nuts2_gdf = nuts2_gdf.set_index("NUTS_ID")
nuts_lookup = nuts2_gdf[["NUTS_NAME", "CNTR_CODE"]]

cols_affo_nuts2 = [
    "Country",
    "afforestation growth rate (m3 ha-1 y-1)",
    "AgeCL (max)",
    "Forest type",
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

# Weighted average by NUTS2
nuts2_weighted_avg = (
    df_valid.groupby("NUTS2")[["afforestation_rate", "afforestation_stock"]]
    .apply(
        lambda g: np.average(g["afforestation_rate"], weights=g["afforestation_stock"])
    )
    .to_dict()
)


# Weighted average by country (NUTS0)
nuts0_weighted_avg = (
    df_valid.groupby("country")[["afforestation_rate", "afforestation_stock"]]
    .apply(
        lambda g: np.average(g["afforestation_rate"], weights=g["afforestation_stock"])
    )
    .to_dict()
)

# ------------------------------
# Step 2: Assign values
# ------------------------------

for nuts2_code in df_affo_nuts2.index:
    # STEP 1: Direct match
    if nuts2_code in nuts2_weighted_avg:
        df_affo_nuts2.at[nuts2_code, "afforestation growth rate (m3 ha-1 y-1)"] = (
            nuts2_weighted_avg[nuts2_code]
        )
        df_affo_nuts2.at[nuts2_code, "Country"] = nuts_lookup.loc[
            nuts2_code, "CNTR_CODE"
        ]
        continue

    # STEP 2: Country-level fallback
    country_code = nuts_lookup.loc[nuts2_code, "CNTR_CODE"]
    if country_code in nuts0_weighted_avg:
        df_affo_nuts2.at[nuts2_code, "afforestation growth rate (m3 ha-1 y-1)"] = (
            nuts0_weighted_avg[country_code]
        )
        df_affo_nuts2.at[nuts2_code, "Country"] = country_code
        continue

# ------------------------------
# Step 3: Spatial fallback for missing regions
# ------------------------------

# Find missing entries
missing_idx = df_affo_nuts2[
    df_affo_nuts2["afforestation growth rate (m3 ha-1 y-1)"].isna()
].index

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
    df_affo_nuts2.at[idx, "afforestation growth rate (m3 ha-1 y-1)"] = nearest_row[
        "affo_rate"
    ]
    df_affo_nuts2.at[idx, "Country"] = target_country

# Sort by NUTS2
df_affo_nuts2 = df_affo_nuts2.sort_index()

# save to csv
output_path = "data/afforestation/afforestation_nuts2.csv"

# Ensure the parent folder exists
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Save the DataFrame
df_affo_nuts2.to_csv(output_path)
