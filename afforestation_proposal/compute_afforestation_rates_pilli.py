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
  6. Output CSV: NUTS2, rate_tCO2_ha_yr

Usage:
    python afforestation_proposal/compute_afforestation_rates_pilli.py
"""

from pathlib import Path

import pandas as pd
import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent / "zenodo_Pilli"
VOL_INC = BASE / "Volume_increment_database"
VOL_BIO = BASE / "Volume_biomass_bcef_database"

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
            mapping[(country, region)] = region  # already NUTS-2 or custom
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
    out_dir = Path(__file__).parent / "output"
    out_dir.mkdir(exist_ok=True)

    # Detailed per forest-type results
    detail_path = out_dir / "afforestation_rates_per_forest_type.csv"
    results_df.to_csv(detail_path, index=False)
    print(f"\n  Saved detailed results: {detail_path}")

    # Aggregated NUTS-2 rates
    nuts2_path = out_dir / "afforestation_rates_nuts2.csv"
    agg.to_csv(nuts2_path, index=False)
    print(f"  Saved NUTS-2 rates:     {nuts2_path}")

    # ── Denmark case study ───────────────────────────────────────────────────
    dk_rows = agg[agg["country"] == "DK"]
    if not dk_rows.empty:
        print("\n" + "=" * 70)
        print("  DENMARK CASE STUDY")
        print("=" * 70)
        print(dk_rows.to_string(index=False))
        dk_detail = results_df[results_df["country"] == "DK"]
        print("\n  Forest-type detail:")
        print(dk_detail[cols].to_string(index=False))
        print(f"\n  Expected (from doc 02): weighted mix ~6.5 tCO₂/ha/yr")
        print(f"  After 15% reversal buffer: ~5.5 tCO₂/ha/yr")

    return results_df, agg


if __name__ == "__main__":
    results_df, agg = compute_rates()
