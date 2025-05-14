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

# local functions
# Function to fill remaining NaNs with average of neighbours NUTs2
def fill_from_neighbors(row):
    if pd.isna(row['affo rate (t/ha)']):
        neighbors = neighbors_dict[row.name]
        neighbor_vals = df_affo_nuts2.loc[neighbors, 'affo rate (t/ha)'].dropna()
        if not neighbor_vals.empty:
            filled_indices.append(row.name)
            return neighbor_vals.mean()
    return row['affo rate (t/ha)']

# Function to fill remaining NaNs with country average
def fill_with_country_average(row):
    if pd.isna(row['affo rate (t/ha)']):
        return country_avg.get(row['NUTS0'], pd.NA)
    return row['affo rate (t/ha)']


#### Inputs ####
# read xlsx with CBM dataset
file_path = "resources/forests/Biomass_calculations.xlsx"  # data with Net Annual Increment for age class: NAI  (m3 ha-1 yr-1)

# shape file nuts2
nuts2_file = "data/nuts/NUTS_RG_03M_2013_4326_LEVL_2.geojson"

#### Outputs ####
output_file_nuts2 = "data/afforestation/afforestation_nuts2.csv"


# Read the desired columns (M:P) from the sheet 'INPUT CBM', skip the first row (0-indexed)
df_nai = pd.read_excel(
    file_path,
    sheet_name='INPUT CBM',
    usecols='M:P',
    skiprows=1
)

# Set the 'ISO' column (which should be the second column) as the index
df_nai.rename(columns={'ISO.1' : 'NUTS2'}, inplace=True)
df_nai.set_index('NUTS2', inplace=True)


# Load NUTS2 regions in PyPSA-eur
nuts2_gdf = gpd.read_file(nuts2_file)
nuts2_gdf = nuts2_gdf[["NUTS_ID", "NUTS_NAME", "CNTR_CODE", "geometry"]]
nuts2_gdf = nuts2_gdf.set_index("NUTS_ID")
nuts_lookup = nuts2_gdf[["NUTS_NAME", "CNTR_CODE"]]

# create afforetstation rate df
cols_aff = ['affo rate (t/ha)', 'Source']
df_affo_nuts2 = pd.DataFrame(index = nuts2_gdf.index, columns=cols_aff, data=0)

# copy
df_expanded = df_nai.copy()

# Split the 'CBM Code' entries by commas and explode
df_expanded['CBM Code'] = df_expanded['CBM Code'].astype(str)
df_expanded = df_expanded.assign(
    NUTS2_codes=df_expanded['CBM Code'].str.split(',')
)
df_expanded = df_expanded.explode('NUTS2_codes')

# Strip whitespace from codes
df_expanded['NUTS2_codes'] = df_expanded['NUTS2_codes'].str.strip()

# Create mappings for AGB and Source from both index and exploded codes
agb_map_index = df_expanded.set_index(df_expanded.index)['AGB (t/ha)']
agb_map_code  = df_expanded.set_index('NUTS2_codes')['AGB (t/ha)']
agb_map = pd.concat([agb_map_index, agb_map_code])
agb_map = agb_map[~agb_map.index.duplicated(keep='first')]

source_map_index = df_expanded.set_index(df_expanded.index)['Source']
source_map_code  = df_expanded.set_index('NUTS2_codes')['Source']
source_map = pd.concat([source_map_index, source_map_code])
source_map = source_map[~source_map.index.duplicated(keep='first')]

# Map to df_affo_nuts2
df_affo_nuts2['affo rate (t/ha)'] = df_affo_nuts2.index.map(agb_map)
df_affo_nuts2['Source'] = df_affo_nuts2.index.map(source_map)


# Filling values in NUTS2 region with the average of neighbouring nuts2

# Ensure both GeoDataFrame and afforestation DataFrame have the same index
assert (df_affo_nuts2.index == nuts2_gdf.index).all()

# Attach geometry to df_affo_nuts2 (creating a GeoDataFrame)
df_geo = df_affo_nuts2.join(nuts2_gdf[['geometry']])
gdf = gpd.GeoDataFrame(df_geo, geometry='geometry', crs=nuts2_gdf.crs)

# Build neighbors using spatial relationships (touches)
neighbors_dict = {}
for i, geom in gdf.geometry.items():
    neighbors = gdf[gdf.geometry.touches(geom)].index.tolist()
    neighbors_dict[i] = neighbors

# Function to fill missing values with average of neighbors
filled_indices = []

# Apply filling
df_affo_nuts2['affo rate (t/ha)'] = df_affo_nuts2.apply(fill_from_neighbors, axis=1)

# Fill 'affo rate (t/ha)' with neighbor averages
df_affo_nuts2['affo rate (t/ha)'] = df_affo_nuts2.apply(fill_from_neighbors, axis=1)
df_affo_nuts2.loc[filled_indices, 'Source'] = 'avg nuts2 near'

# Extract NUTS0 code (first 2 letters) and store in a new column
df_affo_nuts2['NUTS0'] = df_affo_nuts2.index.str[:2]

# Compute average per NUTS0, ignoring NaNs
country_avg = df_affo_nuts2.groupby('NUTS0')['affo rate (t/ha)'].mean()


df_affo_nuts2['affo rate (t/ha)'] = df_affo_nuts2.apply(fill_with_country_average, axis=1)

# Set 'Source' to 'AVG' for these fallback values
df_affo_nuts2.loc[df_affo_nuts2['Source'].isna(), 'Source'] = 'avg nuts0'

# Use Crete (EL43) as reference
reference_value = df_affo_nuts2.loc['EL43', 'affo rate (t/ha)']
reference_source = df_affo_nuts2.loc['EL43', 'Source']

# Apply to Malta and Cyprus
df_affo_nuts2.loc[['MT00', 'CY00'], 'affo rate (t/ha)'] = reference_value
df_affo_nuts2.loc[['MT00', 'CY00'], 'Source'] = 'EL43 copy'

# Sort index
df_affo_nuts2.sort_index(inplace=True)

#SAVE to csv. Ensure the parent folder exists
os.makedirs(os.path.dirname(output_file_nuts2), exist_ok=True)

# Save the DataFrame
df_affo_nuts2.to_csv(output_file_nuts2)
