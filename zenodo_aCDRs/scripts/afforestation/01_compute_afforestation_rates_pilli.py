"""
Compute afforestation CO₂ sequestration rates per NUTS-2 region
using Proposal A: Rotation-Averaged MAI from Pilli et al. (2024).

Pipeline:
  1. Load standing stock (m³/ha) and BCEF (t_biomass/m³) by age class
  2. Filter to even-aged, high-forest productive stands (ForAWS)
  3. For each forest-type × region combination:
       a) Determine rotation age T as the age where MAI_vol = stock(T)/T peaks
       b) Compute MAI_vol = standing_stock(T) / T  [m³/ha/yr]
       c) Compute MAI_biomass = MAI_vol × BCEF(T)  [t_DM/ha/yr]
       d) Compute MAI_CO2 = MAI_biomass × CF × CO2_C × (1 + RSR)  [tCO₂/ha/yr]
  4. Average across forest types per region (equal-weight)
  5. Map Pilli regions → NUTS-2 codes
  6. Expand to ALL NUTS2 regions in the reference GeoJSON via fallback cascade
  7. Output CSVs

Outputs:
  output/afforestation/afforestation_rates_per_forest_type.csv  — detailed per forest-type
  output/afforestation/afforestation_rates_nuts2.csv            — Pilli regions only (as computed)
  output/afforestation/afforestation_rates_nuts2_full.csv       — all 320 NUTS2 regions (with fallbacks)

Usage:
    python afforestation_proposal/01_compute_afforestation_rates_pilli.py
"""

from pathlib import Path

import geopandas as gpd
import pandas as pd
import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parents[2]
INPUT_DIR = ROOT_DIR / "data" /"zenodo_Pilli"
VOL_INC = INPUT_DIR / "Volume_increment_database"
VOL_BIO = INPUT_DIR / "Volume_biomass_bcef_database"

# Reference NUTS2 2013 file (all 320 regions used in pypsa-eur)
NUTS2_GEOJSON = ROOT_DIR / "data" / "nuts" / "NUTS_RG_03M_2013_4326_LEVL_2.geojson"

# ── Constants ────────────────────────────────────────────────────────────────
CF = 0.5  # carbon fraction of dry biomass (IPCC default)
CO2_C = 44 / 12  # molecular ratio CO₂ / C  ≈ 3.667
RSR = 0.25  # root-to-shoot ratio (belowground / aboveground)

# Age class midpoints (years) — 10-year classes
AGE_COLS = [f"AgeCL_{i}" for i in range(1, 21)]
AGE_MIDPOINTS = np.array([5 + 10 * i for i in range(20)])  # 5, 15, 25, ..., 195

# Management types to include (productive high forests)
# Country-specific: AT has HP/HS, most others have H, CZ has MAN, FR has ?
PRODUCTIVE_MT = {"HP", "HS", "H", "MAN", "?"}

# France: Pilli uses 2021 NUTS2 codes; reference GeoJSON uses 2013 codes.
# Boundaries are identical — only the codes changed after the 2015 regional reform.
FR_NUTS2021_TO_2013 = {
    "FRB0": "FR24",  # Centre-Val de Loire       → Centre
    "FRC1": "FR26",  # Bourgogne                 → Bourgogne
    "FRC2": "FR43",  # Franche-Comté             → Franche-Comté
    "FRD1": "FR25",  # Basse-Normandie           → Basse-Normandie
    "FRD2": "FR23",  # Haute-Normandie           → Haute-Normandie
    "FRE1": "FR30",  # Nord-Pas-de-Calais        → Nord-Pas-de-Calais
    "FRE2": "FR22",  # Picardie                  → Picardie
    "FRF1": "FR42",  # Alsace                    → Alsace
    "FRF2": "FR21",  # Champagne-Ardenne         → Champagne-Ardenne
    "FRF3": "FR41",  # Lorraine                  → Lorraine
    "FRG0": "FR51",  # Pays de la Loire          → Pays de la Loire
    "FRH0": "FR52",  # Bretagne                  → Bretagne
    "FRI1": "FR61",  # Aquitaine                 → Aquitaine
    "FRI2": "FR63",  # Limousin                  → Limousin
    "FRI3": "FR53",  # Poitou-Charentes          → Poitou-Charentes
    "FRJ1": "FR81",  # Languedoc-Roussillon      → Languedoc-Roussillon
    "FRJ2": "FR62",  # Midi-Pyrénées             → Midi-Pyrénées
    "FRK1": "FR72",  # Auvergne                  → Auvergne
    "FRK2": "FR71",  # Rhône-Alpes               → Rhône-Alpes
    "FRL0": "FR82",  # Provence-Alpes-Côte d'Azur → PACA
    "FRM0": "FR83",  # Corse                     → Corse
}

# Pilli country codes that differ from the NUTS2 prefix (GR data → EL regions)
COUNTRY_TO_NUTS_PREFIX = {"GR": "EL"}

# Pilli 4-char codes that are NUTS1 proxies (BE20 covers BE21-BE25, etc.)
PILLI_NUTS1_PROXIES = {"BE20": "BE2", "BE30": "BE3"}


def load_standing_stock():
    """Load standing stock (m³/ha) for even-aged stands."""
    df = pd.read_csv(VOL_INC / "Standing_stock_evenaged.csv")
    return df


def load_nai():
    """Load net annual increment (m³/ha/yr) for even-aged stands."""
    df = pd.read_csv(VOL_INC / "NAI_evenaged_stands.csv")
    return df


def load_bcef():
    """Load BCEF database (volume, total_agb, bcef by age class)."""
    df = pd.read_csv(VOL_BIO / "Vol_to_biomass_bcef_database.csv")
    return df


def load_regions_mapping():
    """Load region code → NUTS-2 mapping."""
    df = pd.read_csv(VOL_INC / "Regions_codes.csv")
    # Build mapping: (country, region_code) → list of NUTS-2 codes
    mapping = {}
    for _, row in df.iterrows():
        country = row["Country"]
        region = row["Regions code"]
        nuts2 = str(row.get("Correspondence with NUTS classification system (if possible)", "")).strip()
        if region == "?":
            # Country-level data: assign to all NUTS-2 of that country
            mapping[(country, region)] = "NUTS0"
        elif nuts2 and nuts2 != "nan":
            mapping[(country, region)] = nuts2
        else:
            mapping[(country, region)] = PILLI_NUTS1_PROXIES.get(region, region)
    return mapping


def load_forest_codes():
    """Load forest type → species group mapping."""
    df = pd.read_csv(VOL_INC / "Forest_codes.csv")
    return df


def filter_productive_evenaged(df):
    """
    Filter to even-aged (E), productive high forests.

    Status: include "ForAWS" and "?" (many countries don't distinguish);
            exclude "ForNAWS" (not available for wood supply).
    Management: include productive types (HP, HS, H, MAN, ?);
                exclude coppice (C), non-productive (HNP, NP, PRO, SPE),
                reconstruction (R), and special (S, T).
    """
    excluded_status = {"ForNAWS"}
    mask = (
        (~df["status"].isin(excluded_status))
        & (df["mgmt_strategy"] == "E")
        & (df["mgmt_type"].isin(PRODUCTIVE_MT))
    )
    return df[mask].copy()


def parse_age_values(row, age_cols):
    """
    Extract numeric values from age class columns.
    Returns array of floats, with NaN where data is missing ("-" or empty).
    """
    vals = []
    for col in age_cols:
        v = row[col]
        if isinstance(v, str) and v.strip() == "-":
            vals.append(np.nan)
        else:
            try:
                vals.append(float(v))
            except (ValueError, TypeError):
                vals.append(np.nan)
    return np.array(vals)


MIN_ROTATION_AGE = 25  # years — skip AgeCL_1 and AgeCL_2 (ages 5, 15)
MIN_VALID_AGE_CLASSES = 3  # require ≥3 age classes with data


def compute_rotation_age_and_mai(standing_stock_values):
    """
    Determine optimal rotation age and MAI from standing stock curve.

    Rotation age T is where MAI_vol = stock(T) / T is maximised,
    subject to T ≥ MIN_ROTATION_AGE (to avoid artifacts at very young ages).
    Returns (rotation_age_years, mai_vol_m3_ha_yr, rotation_age_class_idx).
    """
    # Compute MAI at each age class midpoint
    mai = standing_stock_values / AGE_MIDPOINTS

    # Only consider age classes with valid data AND age ≥ minimum
    valid = ~np.isnan(mai) & (AGE_MIDPOINTS >= MIN_ROTATION_AGE)
    if valid.sum() < 1:
        # Fall back: use any valid age class if at least MIN_VALID_AGE_CLASSES exist
        valid_any = ~np.isnan(standing_stock_values)
        if valid_any.sum() < MIN_VALID_AGE_CLASSES:
            return np.nan, np.nan, -1
        valid = valid_any & (AGE_MIDPOINTS > 0)

    # Find peak MAI
    mai_valid = np.where(valid, mai, -np.inf)
    best_idx = np.argmax(mai_valid)
    rotation_age = AGE_MIDPOINTS[best_idx]
    mai_vol = mai_valid[best_idx]

    return rotation_age, mai_vol, best_idx


def build_bcef_lookup(bcef_df):
    """
    Build a lookup: (country, forest_type, region, mgmt_type) → dict of age_class → bcef.
    """
    lookup = {}
    for _, row in bcef_df.iterrows():
        key = (row["country"], row["forest_type"], row["region"], row["mgmt_type"])
        age = row["age_class"]
        bcef_val = row["bcef"]
        try:
            bcef_val = float(bcef_val)
        except (ValueError, TypeError):
            continue
        if key not in lookup:
            lookup[key] = {}
        lookup[key][age] = bcef_val
    return lookup


def get_bcef_for_age(bcef_lookup, country, forest_type, region, mgmt_type, age_class_idx):
    """
    Get BCEF for a specific forest type at a given age class.
    Falls back through progressively broader keys.
    """
    age_label = f"AgeCL_{age_class_idx + 1}"

    # Try exact match
    key = (country, forest_type, region, mgmt_type)
    if key in bcef_lookup and age_label in bcef_lookup[key]:
        return bcef_lookup[key][age_label]

    # Try any mgmt_type for same country/forest_type/region
    for k, ages in bcef_lookup.items():
        if k[0] == country and k[1] == forest_type and k[2] == region:
            if age_label in ages:
                return ages[age_label]

    # Try any region for same country/forest_type
    for k, ages in bcef_lookup.items():
        if k[0] == country and k[1] == forest_type:
            if age_label in ages:
                return ages[age_label]

    # Default BCEF (IPCC default for temperate forests)
    return 0.7  # conservative default


def expand_to_all_nuts2(agg: pd.DataFrame, geojson_path: Path) -> pd.DataFrame:
    """
    Map Pilli-computed rates to ALL NUTS2 regions in the reference GeoJSON.

    Pilli data covers only countries with forest inventory data and uses a mix
    of NUTS2, NUTS1, country-level, and internal region codes. This function
    resolves all codes and fills gaps with a cascade of fallbacks.

    Fallback cascade (in order):
      1. Direct NUTS2 match  — Pilli nuts2 code found in GeoJSON index
      2. NUTS1 propagation   — 3-char Pilli code → all children NUTS2
      3. NUTS0 propagation   — Pilli region == "NUTS0" → all NUTS2 of that country
      4. Distance neighbour  — mean of NUTS2 regions within 100 km (iterated until
                               stable); the 100 km threshold captures meaningful sea
                               crossings (Channel ~34 km, Irish Sea ~70 km) without
                               linking e.g. Scotland to Norway (~300 km)
      5. Country mean        — mean of all filled NUTS2 in same country (handles
                               islands and other isolated regions)
      6. Malta / Cyprus      — copy from Crete (EL43), closest Mediterranean analog
    """
    # Load full NUTS2 geometry
    nuts = gpd.read_file(str(geojson_path))
    if "LEVL_CODE" in nuts.columns:
        nuts = nuts[nuts["LEVL_CODE"] == 2].copy()
    nuts = nuts[["NUTS_ID", "geometry"]].set_index("NUTS_ID").to_crs("EPSG:4326")

    rates = pd.Series(np.nan, index=nuts.index, name="mai_co2_mean")
    source = pd.Series(pd.NA, index=nuts.index, dtype="string", name="source")

    # Translate FR 2021 NUTS2 codes → 2013 before matching against the reference GeoJSON
    agg = agg.copy()
    agg["nuts2"] = agg["nuts2"].map(lambda x: FR_NUTS2021_TO_2013.get(x, x))

    # ── Step 1: Direct NUTS2 match ────────────────────────────────────────────
    for _, row in agg.iterrows():
        nid = row["nuts2"]
        if nid in rates.index and pd.isna(rates[nid]):
            rates[nid] = row["mai_co2_mean"]
            source[nid] = "direct"

    # ── Step 2: NUTS1 propagation (3-char Pilli codes) ───────────────────────
    nuts1_rows = agg[~agg["nuts2"].isin(["NUTS0"]) & (agg["nuts2"].str.len() == 3)]
    for _, row in nuts1_rows.iterrows():
        parent = row["nuts2"]
        children = [rid for rid in rates.index if rid.startswith(parent) and pd.isna(rates[rid])]
        for child in children:
            rates[child] = row["mai_co2_mean"]
            source[child] = f"{parent}→NUTS1"

    # ── Step 3: Country-level propagation ("NUTS0") ───────────────────────────
    # COUNTRY_TO_NUTS_PREFIX handles cases where Pilli country code differs from
    # the NUTS2 prefix (e.g. Pilli uses "GR" but NUTS2 codes start with "EL").
    for _, row in agg[agg["nuts2"] == "NUTS0"].iterrows():
        country = row["country"]
        prefix = COUNTRY_TO_NUTS_PREFIX.get(country, country)
        children = [rid for rid in rates.index if rid[:2] == prefix and pd.isna(rates[rid])]
        for child in children:
            rates[child] = row["mai_co2_mean"]
            source[child] = f"{country}→NUTS0"

    n_missing = int(rates.isna().sum())
    print(f"  After Pilli direct/NUTS1/NUTS0 mapping: {len(rates) - n_missing} filled, "
          f"{n_missing} still missing — applying fallbacks ...")

    # ── Step 4: Distance-based neighbour mean, iterated ──────────────────────
    # Uses a 100 km buffer in EPSG:3035 so sea-crossing neighbours are included:
    # southern England ← N France/Belgium, Scotland ← N Ireland via NI ← Ireland,
    # Adriatic islands ← mainland, Baltic islands ← mainland, etc.
    # Then iterative propagation fills inland from these coastal anchors.
    if rates.isna().any():
        SEA_THRESHOLD_M = 100_000
        gdf = gpd.GeoDataFrame(
            {"geometry": nuts["geometry"]}, crs=nuts.crs
        ).to_crs(3035)
        neighbors_dict = {
            rid: (
                []
                if (geom is None or geom.is_empty)
                else [
                    n for n in gdf[gdf.geometry.intersects(geom.buffer(SEA_THRESHOLD_M))].index
                    if n != rid
                ]
            )
            for rid, geom in gdf.geometry.items()
        }
        changed = True
        while changed:
            changed = False
            for rid in list(rates.index[rates.isna()]):
                valid = [n for n in neighbors_dict.get(rid, []) if pd.notna(rates[n])]
                if valid:
                    rates[rid] = rates[valid].mean()
                    source[rid] = "avg neighbours (100km)"
                    changed = True
        prev = n_missing
        n_missing = int(rates.isna().sum())
        if n_missing < prev:
            print(f"    Distance neighbour: filled {prev - n_missing} ({n_missing} left).")

    # ── Step 5: Country mean of filled regions ────────────────────────────────
    if rates.isna().any():
        nuts0 = rates.index.str[:2]
        country_mean = rates.groupby(nuts0).mean()
        for rid in rates.index[rates.isna()]:
            key = rid[:2]
            if key in country_mean.index and pd.notna(country_mean[key]):
                rates[rid] = country_mean[key]
                source[rid] = f"avg country {key}"
        prev = n_missing
        n_missing = int(rates.isna().sum())
        if n_missing < prev:
            print(f"    Country mean: filled {prev - n_missing} ({n_missing} left).")

    # ── Step 6: Malta / Cyprus → Crete (EL43) ────────────────────────────────
    for code in ["MT00", "CY00"]:
        if code in rates.index and pd.isna(rates[code]):
            if "EL43" in rates.index and pd.notna(rates["EL43"]):
                rates[code] = rates["EL43"]
                source[code] = "EL43 copy (Crete)"

    # Final report
    n_missing = int(rates.isna().sum())
    if n_missing:
        print(f"[warn] {n_missing} NUTS2 still missing after all fallbacks: "
              f"{list(rates.index[rates.isna()])}")
    else:
        print(f"[ok] All {len(rates)} NUTS2 regions covered.")

    out = pd.DataFrame({"CO2 seq rate tCO2/(ha y)": rates, "Source": source})
    out.index.name = "NUTS2"
    return out


def compute_rates():
    """Main computation: rotation-averaged CO₂ sequestration rates per NUTS-2."""

    print("Loading data...")
    stock_df = load_standing_stock()
    bcef_df = load_bcef()
    regions_map = load_regions_mapping()
    forest_codes = load_forest_codes()

    print(f"  Standing stock: {len(stock_df)} rows")
    print(f"  BCEF database:  {len(bcef_df)} rows")

    # Filter to productive even-aged high forests
    stock_filt = filter_productive_evenaged(stock_df)
    print(f"  After filtering (ForAWS, even-aged, productive): {len(stock_filt)} rows")

    # Build BCEF lookup
    bcef_filt = bcef_df[
        (bcef_df["status"] == "ForAWS")
        & (bcef_df["mgmt_strategy"] == "E")
    ].copy()
    bcef_lookup = build_bcef_lookup(bcef_filt)
    print(f"  BCEF lookup keys: {len(bcef_lookup)}")

    # Compute per forest-type × region
    results = []

    for _, row in stock_filt.iterrows():
        country = row["country"]
        forest_type = row["forest_type"]
        region = row["region"]
        mgmt_type = row["mgmt_type"]
        con_broad = row["con_broad"]

        # Parse standing stock values
        stock_vals = parse_age_values(row, AGE_COLS)

        # Find rotation age and volumetric MAI
        rotation_age, mai_vol, rot_idx = compute_rotation_age_and_mai(stock_vals)
        if np.isnan(mai_vol) or rot_idx < 0:
            continue

        # Get BCEF at rotation age
        bcef_val = get_bcef_for_age(
            bcef_lookup, country, forest_type, region, mgmt_type, rot_idx
        )

        # Convert: m³/ha/yr → tCO₂/ha/yr
        #   MAI_biomass = MAI_vol × BCEF  [t_DM/ha/yr]
        #   MAI_CO2 = MAI_biomass × CF × CO2_C × (1 + RSR)
        mai_biomass = mai_vol * bcef_val
        mai_co2 = mai_biomass * CF * CO2_C * (1 + RSR)

        # Map region to NUTS-2
        nuts2_code = regions_map.get((country, region), region)

        results.append({
            "country": country,
            "region": region,
            "nuts2": nuts2_code,
            "forest_type": forest_type,
            "con_broad": con_broad,
            "mgmt_type": mgmt_type,
            "rotation_age": rotation_age,
            "mai_vol_m3_ha_yr": mai_vol,
            "bcef": bcef_val,
            "mai_co2_tCO2_ha_yr": mai_co2,
        })

    results_df = pd.DataFrame(results)
    print(f"\n  Computed rates for {len(results_df)} forest-type × region combinations")

    # ── Summary statistics ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  PER FOREST-TYPE RESULTS (sample)")
    print("=" * 70)
    cols = ["country", "region", "forest_type", "con_broad", "rotation_age",
            "mai_vol_m3_ha_yr", "bcef", "mai_co2_tCO2_ha_yr"]
    print(results_df[cols].head(20).to_string(index=False))

    # ── Aggregate to NUTS-2: equal-weight average across forest types ────────
    # For regions with "NUTS0" (country-level data), use country code
    agg = (
        results_df
        .groupby(["country", "nuts2"])
        .agg(
            mai_co2_mean=("mai_co2_tCO2_ha_yr", "mean"),
            mai_co2_min=("mai_co2_tCO2_ha_yr", "min"),
            mai_co2_max=("mai_co2_tCO2_ha_yr", "max"),
            n_forest_types=("forest_type", "nunique"),
            rotation_age_mean=("rotation_age", "mean"),
            mai_vol_mean=("mai_vol_m3_ha_yr", "mean"),
        )
        .reset_index()
    )

    print("\n" + "=" * 70)
    print("  NUTS-2 AGGREGATED RATES  [tCO₂/ha/yr]")
    print("=" * 70)
    print(agg.to_string(index=False))

    # ── Overall statistics ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  OVERALL STATISTICS")
    print("=" * 70)
    print(f"  Countries:           {results_df['country'].nunique()}")
    print(f"  NUTS-2 regions:      {agg['nuts2'].nunique()}")
    print(f"  Forest types total:  {len(results_df)}")
    print(f"  Mean rate:           {agg['mai_co2_mean'].mean():.2f} tCO₂/ha/yr")
    print(f"  Median rate:         {agg['mai_co2_mean'].median():.2f} tCO₂/ha/yr")
    print(f"  Min region rate:     {agg['mai_co2_mean'].min():.2f} tCO₂/ha/yr")
    print(f"  Max region rate:     {agg['mai_co2_mean'].max():.2f} tCO₂/ha/yr")
    print(f"  Expected range:      5–15 tCO₂/ha/yr (per methodology doc)")

    # ── Save outputs ─────────────────────────────────────────────────────────
    out_dir = ROOT_DIR / "outputs" / "afforestation"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Detailed per forest-type results
    detail_path = out_dir / "afforestation_rates_per_forest_type.csv"
    results_df.to_csv(detail_path, index=False)
    print(f"\n  Saved detailed results: {detail_path}")

    # Aggregated NUTS-2 rates (Pilli regions only)
    nuts2_path = out_dir / "afforestation_rates_nuts2.csv"
    agg.to_csv(nuts2_path, index=False)
    print(f"  Saved NUTS-2 rates:     {nuts2_path}")

    # Full coverage: all 320 NUTS2 regions with fallback cascade
    if NUTS2_GEOJSON.exists():
        print("\nExpanding to all NUTS2 regions in reference GeoJSON ...")
        full_df = expand_to_all_nuts2(agg, NUTS2_GEOJSON)
        full_path = out_dir / "afforestation_rates_nuts2_full.csv"
        full_df.to_csv(full_path)
        print(f"  Saved full NUTS-2 rates: {full_path}")
    else:
        print(f"[warn] NUTS2 GeoJSON not found at {NUTS2_GEOJSON} — skipping full expansion.")

    return results_df, agg


if __name__ == "__main__":
    results_df, agg = compute_rates()
