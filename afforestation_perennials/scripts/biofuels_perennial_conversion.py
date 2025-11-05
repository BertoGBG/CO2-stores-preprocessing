import os
import pandas as pd
import requests
import numpy as np
from pathlib import Path
import geopandas as gpd


# ----helpers -----
def harmonize_to_nuts2021(df, keep_col, nuts2021_n2):
    """
    df : DataFrame indexed by ['geo', 'TIME_PERIOD', 'mapping']
         contains both NUTS2 and NUTS0 rows
    keep_col : column to harmonize (e.g. 'weighted_YL_(t/ha)')
    nuts2021_n2 : GeoDataFrame with index=NUTS2_ID and geometry
    """

    # NUTS2 target regions
    target = nuts2021_n2.index.sort_values()

    # Split df
    df = df.copy()

    # Identify NUTS2 and NUTS0 by index length
    df['is_nuts2'] = df.index.str.len() == 4

    # Split into NUTS2 and NUTS0
    df_nuts2 = df[df['is_nuts2']]
    df_nuts0 = df[~df['is_nuts2']]

    # Prepare working layer for only NUTS2
    df_work = df_nuts2[[keep_col]].reindex(target)

    # Country code extraction (first 2 chars)
    df_work['country'] = df_work.index.str[:2]

    # ✅ Fallback 2 — Use NUTS-0 values
    df_work = df_work.join(
        df_nuts0[[keep_col]].rename(columns={keep_col: 'fallback'}),
        on='country'
    )
    df_work[keep_col] = df_work[keep_col].fillna(df_work['fallback'])
    df_work.drop(columns=['fallback'], inplace=True)

    # Join geometry
    nuts_proj = nuts2021_n2.to_crs(epsg=3035)
    gdf = nuts_proj.join(df_work)[[keep_col, 'country', 'geometry']]

    # ✅ Fallback 3 — Spatial neighbors mean
    mask_missing = gdf[keep_col].isna()
    if mask_missing.any():
        missing_regions = gdf[mask_missing]
        print(f"⚠️ Neighbor fallback: {len(missing_regions)} regions")

        # Build neighbors list
        neighbors = {
            idx: gdf[gdf.geometry.touches(row.geometry)].index.tolist()
            for idx, row in gdf.iterrows()
        }

        for idx, row in missing_regions.iterrows():
            neigh_vals = gdf.loc[neighbors[idx], keep_col].dropna()

            if len(neigh_vals) > 0:
                gdf.at[idx, keep_col] = neigh_vals.mean()
            else:
                # If no touching neighbors are valid → nearest non-missing
                nearest = gdf[gdf[keep_col].notna()].distance(row.geometry).idxmin()
                gdf.at[idx, keep_col] = gdf.at[nearest, keep_col]

    # Final output tidy
    result = gdf[[keep_col]]
    result.index.name = 'NUTS2'
    return result


def download_database_nuts2(filepath):

    url = (
        "https://ec.europa.eu/eurostat/api/dissemination/sdmx/3.0/data/dataflow/ESTAT/apro_cpshr/1.0/*.*.*.*?"
        "c[freq]=A"
        "&c[crops]=C0000,C1000,C1210,C1300,C1310,C1320,P0000,R2000,I1110-1130,I0000,I1100,I1110,I1120,I1130,I9000,G0000,G1000,G2000,G2100,G2900,G2910,J0000,PECR,PECR9"
        "&c[strucpro]=AR,PR_HU_EU,YI_HU_EU,MA"
        "&c[TIME_PERIOD]=2020,2019,2018,2017"
        "&compress=false&format=csvdata&formatVersion=2.0&lang=en&labels=name"
        "&c[geo]=BE10,BE21,BE22,BE23,BE24,BE25,BE31,BE32,BE33,BE34,BE35,BEZZ,"
        "BG31,BG32,BG33,BG34,BG41,BG42,BGZZ,"
        "CZ01,CZ02,CZ03,CZ04,CZ05,CZ06,CZ07,CZ08,CZZZ,"
        "DK01,DK02,DK03,DK04,DK05,DKZZ,"
        "DE11,DE12,DE13,DE14,DE21,DE22,DE23,DE24,DE25,DE26,DE27,DE30,DE40,DE50,DE60,DE71,DE72,DE73,DE80,DE91,DE92,DE93,DE94,DEA1,DEA2,DEA3,DEA4,DEA5,DEB1,DEB2,DEB3,DEC0,DED2,DED4,DED5,DEE0,DEF0,DEG0,DEZZ,"
        "EE00,EEZZ,"
        "IE04,IE05,IE06,IE01,IE02,IEZZ,"
        "EL11,EL12,EL13,EL14,EL21,EL22,EL23,EL24,EL25,EL30,EL41,EL42,EL43,EL51,EL52,EL53,EL54,EL61,EL62,EL63,EL64,EL65,ELZZ,"
        "ES11,ES12,ES13,ES21,ES22,ES23,ES24,ES30,ES41,ES42,ES43,ES51,ES52,ES53,ES61,ES62,ES63,ES64,ES70,ESZZ,"
        "FR10,FRB0,FRC1,FRC2,FRD1,FRD2,FRE1,FRE2,FRF1,FRF2,FRF3,FRG0,FRH0,FRI1,FRI2,FRI3,FRJ1,FRJ2,FRK1,FRK2,FRL0,FRM0,FRY1,FRY2,FRY3,FRY4,FRY5,FR21,FR22,FR23,FR24,FR25,FR26,FR30,FR41,FR42,FR43,FR51,FR52,FR53,FR61,FR62,FR63,FR71,FR72,FR81,FR82,FR83,FRA1,FRA2,FRA3,FRA4,FRA5,FR91,FR92,FR93,FR94,FRZZ,"
        "HR02,HR03,HR04,HR05,HR06,HR01,"
        "ITC1,ITC2,ITC3,ITC4,ITD1,ITD2,ITD3,ITD4,ITD5,ITE1,ITE2,ITE3,ITE4,ITF1,ITF2,ITF3,ITF4,ITF5,ITF6,ITG1,ITG2,ITH1,ITH2,ITH3,ITH4,ITH5,ITI1,ITI2,ITI3,ITI4,ITZZ,"
        "CY00,CYZZ,"
        "LV00,LVZZ,"
        "LT01,LT02,LTZZ,"
        "LU00,LUZZ,"
        "HU11,HU12,HU10,HU21,HU22,HU23,HU31,HU32,HU33,HUZZ,"
        "MT00,MTZZ,"
        "NL11,NL12,NL13,NL21,NL22,NL23,NL31,NL32,NL33,NL34,NL35,NL36,NL41,NL42,NLZZ,"
        "AT11,AT12,AT13,AT21,AT22,AT31,AT32,AT33,AT34,ATZZ,"
        "PL11,PL12,PL21,PL22,PL31,PL32,PL33,PL34,PL41,PL42,PL43,PL51,PL52,PL61,PL62,PL63,PL71,PL72,PL81,PL82,PL84,PL91,PL92,PLZZ,"
        "PT11,PT15,PT16,PT17,PT18,PT19,PT1A,PT1B,PT1C,PT1D,PT20,PT30,PTZZ,"
        "RO11,RO12,RO21,RO22,RO31,RO32,RO41,RO42,ROZZ,"
        "SI03,SI01,SI04,SI02,SIZZ,"
        "SK01,SK02,SK03,SK04,SKZZ,"
        "FI13,FI18,FI19,FI1A,FI1B,FI1C,FI1D,FI20,FIZZ,"
        "SE11,SE12,SE21,SE22,SE23,SE31,SE32,SE33,SEZZ,"
        "IS00,"
        "NO01,NO02,NO03,NO04,NO05,NO06,NO07,NO08,NO09,NO0A,NO0B,"
        "CH01,CH02,CH03,CH04,CH05,CH06,CH07,"
        "UKC1,UKC2,UKD1,UKD3,UKD4,UKD6,UKD7,UKE1,UKE2,UKE3,UKE4,UKF1,UKF2,UKF3,UKG1,UKG2,UKG3,UKH1,UKH2,UKH3,UKI1,UKI2,UKI3,UKI4,UKI5,UKI6,UKI7,UKJ1,UKJ2,UKJ3,UKJ4,UKK1,UKK2,UKK3,UKK4,UKL1,UKL2,UKM2,UKM3,UKM5,UKM6,UKM7,UKM8,UKM9,UKN0,UKZZ,"
        "ME00,MK00,AL01,AL02,AL03,RS11,RS12,RS21,RS22,"
        "TR10,TR21,TR22,TR31,TR32,TR33,TR41,TR42,TR51,TR52,TR61,TR62,TR63,TR71,TR72,TR81,TR82,TR83,TR90,TRA1,TRA2,TRB1,TRB2,TRC1,TRC2,TRC3"
    )

    print("📡 Downloading data from Eurostat...")
    response = requests.get(url)

    if response.status_code == 200:
        with open(filepath, "wb") as f:
            f.write(response.content)
        print(f"✅ File downloaded successfully → {os.path.abspath(filepath)}")
    else:
        print(f"❌ Failed → HTTP {response.status_code}")
        print("Server says:", response.text[:250])


def download_database_nuts0(filepath):

    url = (
        "https://ec.europa.eu/eurostat/api/dissemination/sdmx/3.0/data/dataflow/ESTAT/apro_cpshr/1.0/*.*.*.*?"
        "c[freq]=A"
        "&c[crops]=C0000,C1000,C1210,C1300,C1310,C1320,P0000,R2000,I1110-1130,I0000,I1100,I1110,I1120,I1130,I9000,G0000,G1000,G2000,G2100,G2900,G2910,J0000,PECR,PECR9"
        "&c[strucpro]=AR,PR_HU_EU,YI_HU_EU,MA"
        "&c[TIME_PERIOD]=2020,2019,2018,2017"
        "&c[geo]=BE,BG,CZ,DK,DE,EE,IE,EL,ES,FR,HR,IT,CY,LV,LT,LU,HU,MT,NL,AT,PL,PT,RO,SI,SK,FI,SE,NO,CH,IS,ME,MK,AL,RS,TR,UK,EU"
        "&compress=false&format=csvdata&formatVersion=2.0&lang=en&labels=name"
    )

    print("📡 Downloading NATIONAL (NUTS-0) data from Eurostat…")
    response = requests.get(url)

    if response.status_code == 200:
        with open(filepath, "wb") as f:
            f.write(response.content)
        print(f"✅ Downloaded successfully → {os.path.abspath(filepath)}")
    else:
        print(f"❌ HTTP {response.status_code}")
        print("Server response:", response.text[:500])


def calculate_yields(filepath_nuts2, filepath_nuts0, crops_sel, crops_mapping, biofuel_yields):
    # filter columns - keep only relevant
    df_crops_raw_nuts2 = pd.read_csv(filepath_nuts2)
    df_crops_raw_nuts2['TIME_PERIOD'] = df_crops_raw_nuts2['TIME_PERIOD'].astype(int)  #

    df_crops_raw_nuts0 = pd.read_csv(filepath_nuts0)
    df_crops_raw_nuts0['TIME_PERIOD'] = df_crops_raw_nuts0['TIME_PERIOD'].astype(int)  #

    df_crops_raw = pd.concat([df_crops_raw_nuts0, df_crops_raw_nuts2], ignore_index=True)

    # drop empty and irrelevant columns
    columns_to_drop = [
        'Observation value',
        'OBS_FLAG',
        'Observation status (Flag) V2 structure',
        'CONF_STATUS',
        'Confidentiality status (flag)',
        'Time',
        'STRUCTURE_ID',
        'STRUCTURE',
        'STRUCTURE_NAME',
        'Geopolitical entity (reporting)',
        'Time frequency'
    ]

    # useful columns:
    # Index(['freq', 'crops', 'Crops', 'strucpro', 'Structure of production', 'geo','TIME_PERIOD', 'OBS_VALUE'],
    # freq : 'A' meaning annual
    # 'crops' : code of the crops e.g. 'G0000', 'G1000', 'G2000', 'G2100', 'G2900'
    # 'Crops' : name of the corp e.g. Sugar beet (excluding seed
    # 'strucpro' : type of data ['AR', 'MA', 'PR_HU_EU'], where AR is  cultivated area, MA is  and PR_HU_EU is production at standard EU humidity
    # 'OBS_VALUE' : numerical value

    df_crops = df_crops_raw.drop(columns=columns_to_drop, errors='ignore')
    df_crops['OBS_VALUE'] = df_crops['OBS_VALUE'].fillna(0)

    # Step 1: Filter to relevant rows for 2023 and strucpro of interest
    # Keep only relevant structure and crops (all years)
    df_sub = df_crops[
        (df_crops['strucpro'].isin(['AR', 'PR_HU_EU'])) &
        (df_crops['crops'].isin(crops_sel))
        ][['crops', 'geo', 'TIME_PERIOD', 'strucpro', 'OBS_VALUE']]

    # Pivot so AR and PR_HU_EU are columns for each (crop, geo, year)
    df_pivot = (
        df_sub.pivot_table(
            index=['crops', 'geo', 'TIME_PERIOD'],
            columns='strucpro',
            values='OBS_VALUE'
        )
        .dropna(subset=['AR', 'PR_HU_EU'])
        .reset_index()
    )

    # Compute yield per year (PR_HU_EU / AR)
    df_pivot['YL'] = np.divide(
        df_pivot['PR_HU_EU'],
        df_pivot['AR'],
        out=np.zeros_like(df_pivot['PR_HU_EU'], dtype=float),
        where=df_pivot['AR'] != 0
    )

    # Average data across years
    df_avg_yield = (
        df_pivot
        .groupby(['crops', 'geo'], as_index=False)[['AR', 'PR_HU_EU', 'YL']]
        .mean()
    )

    min_year = df_pivot['TIME_PERIOD'].min()
    max_year = df_pivot['TIME_PERIOD'].max()
    df_avg_yield['TIME_PERIOD'] = f"{min_year}-{max_year}"

    # map crops to categories unsustainable biofuels in
    rev_map = {
        code: key
        for key, val in crops_mapping.items()
        for code in (val if isinstance(val, list) else [val])
    }
    df_avg_yield['mapping'] = df_avg_yield['crops'].map(rev_map)

    # calculate weighted production per crop within mapping classes
    df_avg_yield['PR_share'] = (df_avg_yield['PR_HU_EU'] / df_avg_yield.groupby(['geo', 'TIME_PERIOD', 'mapping'])[
        'PR_HU_EU'].transform('sum'))
    df_avg_yield['PR_share'] = df_avg_yield['PR_share'].fillna(0)

    # calculated average weighted yield
    df_avg_yield['weighted_YL_(t/ha)'] = df_avg_yield['YL'] * df_avg_yield['PR_share']

    # sanity check for very low yields due to small productions
    thresholds = {
        'MINBIOCRP11': 2.0,  # cereals
        'MINBIOCRP21': 50.0,  # sugar beet
        'MINBIORPS1': 1.5,  # rapeseed
        'PERENNIALS': 5.0  # perennial grasses
    }

    # Apply crop-specific minimum threshold
    df_avg_yield['weighted_YL_(t/ha)'] = df_avg_yield.apply(
        lambda row: row['weighted_YL_(t/ha)'] if row['weighted_YL_(t/ha)'] >= thresholds.get(row['mapping'], 0) else 0,
        axis=1
    )

    # weighted yields from current production : applies to unsustainable biofuels
    weighted_yields = (
        df_avg_yield.groupby(['geo', 'TIME_PERIOD', 'mapping'])['weighted_YL_(t/ha)'].sum()
    )

    unsustainalbe_biofuels_yields = pd.DataFrame(weighted_yields)

    # unsustainable biomass yield units from t/ha to MWh/ha
    unsustainalbe_biofuels_yields = unsustainalbe_biofuels_yields[
        unsustainalbe_biofuels_yields.index.get_level_values('mapping') != 'PERENNIALS'
        ]

    unsustainalbe_biofuels_yields['energy_yields_(MWh/ha)'] = (
            unsustainalbe_biofuels_yields['weighted_YL_(t/ha)'] *
            unsustainalbe_biofuels_yields.index.get_level_values('mapping').map(biofuel_yields)
    )

    # yields of perennials per hectar in ton/ha
    perennial_yields = pd.DataFrame(weighted_yields)
    perennial_yields = perennial_yields[
        perennial_yields.index.get_level_values('mapping') == 'PERENNIALS'
        ]

    # max yields from current production : applies to perennials for greeen biorefining
    max_yields = df_avg_yield.groupby(['geo', 'TIME_PERIOD', 'mapping'])['YL'].max()

    perennial_yields_max = pd.DataFrame(max_yields)
    perennial_yields_max = perennial_yields_max[
        perennial_yields_max.index.get_level_values('mapping') == 'PERENNIALS'
        ]
    return unsustainalbe_biofuels_yields, perennial_yields


def main():
    PROJECT_ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd()
    print(PROJECT_ROOT)
    print(PROJECT_ROOT)
    CROPS_CSV_NUTS2 = PROJECT_ROOT / "resources" / "eurostat_crops" / "estat_apro_cpshr_filtered_en_nuts2.csv"
    CROPS_CSV_NUTS0 = PROJECT_ROOT / "resources" / "eurostat_crops" / "estat_apro_cpshr_filtered_en_nuts0.csv"

    NUTS2_2021_GEOJSON = PROJECT_ROOT / "data" / "nuts" / "NUTS_RG_01M_2021_4326_LEVL_2.geojson"
    OUT_CSV_perennials = PROJECT_ROOT / 'data' / 'crops' / "yields_perennials_nuts2.csv"
    OUT_CSV_MINBIOCRP21 = PROJECT_ROOT / 'data' / 'crops' / "yields_MINBIOCRP21_nuts2.csv"
    OUT_CSV_MINBIORPS1 = PROJECT_ROOT / 'data' / 'crops' / "yields_MINBIORPS1_nuts2.csv"
    OUT_CSV_MINBIOCRP11 = PROJECT_ROOT / 'data' / 'crops' / "yields_MINBIOCRP11_nuts2.csv"

    OUT_CSV_perennials.parent.mkdir(parents=True, exist_ok=True)

    #################################
    #           INPUT DATA

    download_dataset_flag = True

    # mapping perennial crops in eurostat
    perennial_codes = ['G0000', 'G1000', 'G2000', 'G2100', 'G2900']

    # mapping of 1st generation biofuels eurostat to enspreso
    crops_mapping = dict(MINBIOCRP11 = ['C0000','C1000','C1210','C1300','C1310','C1320',], # bioethanol from cereals,
                   MINBIOCRP21 = 'R2000', # bioethanol from sugar beet
                   MINBIORPS1 =  ['I1110', 'I1120', 'I1130', 'I1110-1130', 'I0000'], # biodiesel from rape seed, sunflower seed, soy
                   PERENNIALS = perennial_codes)

    LHV_fuels = dict(ethanol = 26.81/3.6, # MWh/t
                     biodiesel = 36.7/3.6) # MWh/t

    # biofuel conversion process from (doi:10.2760/69179) - NOTE: same inputs used in CAPRI in 2015 (MJ biofuel/MJ feedstock)
    biofuel_yields = dict(MINBIOCRP11 = 0.295 * LHV_fuels['ethanol'] ,  # t_ethanol/t_wheat ( wheat grain @ 13.5 % H2O) * LHV (MWh/t)- TABLE 93
                   MINBIOCRP21 = 0.07777 * LHV_fuels['ethanol'], # t_ethanol/t_SB (SB at 16% sugar content) LHV (MWh/t) - TABLE 133
                   MINBIORPS1 = 0.420 / 1.0063 * LHV_fuels['biodiesel']) # kg crude oil/(kg rapeseed @ 9 %H2O) * crude to FAME * LHV (MWh/t) - TABLE 155 and 159

    # mapping of crops
    other_crops_codes = [item for v in crops_mapping.values() for item in (v if isinstance(v, list) else [v])]
    crops_sel = perennial_codes + other_crops_codes

    #################################
    #      DOWNLOAD DATASETS

    # download file if not present
    if download_dataset_flag:
        download_database_nuts2(CROPS_CSV_NUTS2)
        download_database_nuts0(CROPS_CSV_NUTS0)

    #################################
    #    PROCESSING OF DATASETS
    unsustainalbe_biofuels_yields, perennial_yields  = calculate_yields(filepath_nuts0=CROPS_CSV_NUTS0,filepath_nuts2=CROPS_CSV_NUTS2, crops_sel=crops_sel, crops_mapping=crops_mapping, biofuel_yields=biofuel_yields)

    # create yield conversion for each
    yield_MINBIOCRP11 = unsustainalbe_biofuels_yields[unsustainalbe_biofuels_yields.index.get_level_values('mapping') == 'MINBIOCRP11']
    yield_MINBIOCRP21 = unsustainalbe_biofuels_yields[unsustainalbe_biofuels_yields.index.get_level_values('mapping') == 'MINBIOCRP21']
    yield_MINBIORPS1 = unsustainalbe_biofuels_yields[unsustainalbe_biofuels_yields.index.get_level_values('mapping') == 'MINBIORPS1']

    # Drop index levels and set index to NUTS2
    yield_MINBIOCRP11 = yield_MINBIOCRP11.droplevel(['TIME_PERIOD', 'mapping'])
    yield_MINBIOCRP11.index.name = 'NUTS2'
    yield_MINBIOCRP21 = yield_MINBIOCRP21.droplevel(['TIME_PERIOD', 'mapping'])
    yield_MINBIOCRP21.index.name = 'NUTS2'
    yield_MINBIORPS1 = yield_MINBIORPS1.droplevel(['TIME_PERIOD', 'mapping'])
    yield_MINBIORPS1.index.name = 'NUTS2'
    perennial_yields = perennial_yields.droplevel(['TIME_PERIOD', 'mapping'])
    perennial_yields.index.name = 'NUTS2'

    #################################
    # MAP DATA TO ALL NUTS2 REGIONS
    nuts2021_n2 = gpd.read_file(NUTS2_2021_GEOJSON) \
               .loc[:, ["NUTS_ID", "NUTS_NAME", "CNTR_CODE", "geometry"]] \
               .set_index("NUTS_ID")

    yield_MINBIOCRP11_full = harmonize_to_nuts2021(yield_MINBIOCRP11, 'energy_yields_(MWh/ha)', nuts2021_n2)
    yield_MINBIOCRP21_full = harmonize_to_nuts2021(yield_MINBIOCRP21, 'energy_yields_(MWh/ha)', nuts2021_n2)
    yield_MINBIORPS1_full = harmonize_to_nuts2021(yield_MINBIORPS1, 'energy_yields_(MWh/ha)', nuts2021_n2)
    yields_perennials_full = harmonize_to_nuts2021(perennial_yields, 'weighted_YL_(t/ha)', nuts2021_n2)

    """ save to csv"""
    # Save the DataFrame
    yield_MINBIOCRP11_full.to_csv(OUT_CSV_MINBIOCRP11, index=True)
    yield_MINBIOCRP21_full.to_csv(OUT_CSV_MINBIOCRP21, index=True)
    yield_MINBIORPS1_full.to_csv(OUT_CSV_MINBIORPS1, index=True)
    yields_perennials_full.to_csv(OUT_CSV_perennials, index=True)

if __name__ == "__main__":
    main()