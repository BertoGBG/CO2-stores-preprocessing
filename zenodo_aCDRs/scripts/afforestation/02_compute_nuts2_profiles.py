"""
Compute monthly GPP seasonal profiles per NUTS2 and merge with growth rates.

Pipeline:
  1. Load FluxCom daily GPP NetCDF files (gC m⁻² day⁻¹)
  2. Clip to Europe bounding box
  3. Aggregate: daily → monthly climatology (mean across all years and days per month)
  4. Zonal mean per NUTS2 polygon for each month
  5. Normalise: each region's 12-month profile sums to 1.0
     Partial-NaN handling: NaN winter months (GPP ≈ 0) → treated as 0, row re-normalised.
     Fallback for all-NaN regions (in order):
       a) mean profile of NUTS2 regions within 100 km (iterated, handles sea crossings)
       b) mean profile of all resolved regions in the same country
       c) uniform 1/12
  6. Merge with afforestation_nuts2_growth_rates.csv (annual tCO₂/ha/yr)
     → monthly rates in tCO₂ ha⁻¹ month⁻¹ per NUTS2

Uses NUTS 2013 regions by default (consistent with zenodo_aCDRs growth rates).

Outputs:
  output/afforestation_nuts2_monthly_weights.csv   — NUTS2 × 12 weights (sum = 1 per row)
  output/afforestation_nuts2_monthly_rates.csv     — NUTS2 × 12 monthly tCO₂ ha⁻¹ month⁻¹

Reference:
  Jung, M. et al. (2020). Biogeosciences, 17(5), 1343–1365.
  https://doi.org/10.5194/bg-17-1343-2020

Usage:
    python 02_compute_nuts2_profiles.py
    python 02_compute_nuts2_profiles.py --gpp-dir data/fluxcom_raw/ --rates ../../zenodo_aCDRs/outputs/afforestation_nuts2_growth_rates.csv
    python 02_compute_nuts2_profiles.py --nuts-path ../../zenodo_aCDRs/data/nuts/NUTS_RG_03M_2013_4326_LEVL_2.geojson
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from shapely.geometry import Point

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parents[2]
GPP_DIR    = ROOT_DIR / "data" / "fluxcom_raw"
OUTPUT_DIR = ROOT_DIR / "outputs"/ "afforestation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Default: zenodo_aCDRs growth rates (NUTS 2013, all regions covered)
RATES_CSV = ROOT_DIR / "outputs" / "afforestation" / "afforestation_rates_nuts2_full.csv"

# Default NUTS: local NUTS 2013 file (consistent with zenodo_aCDRs growth rates)
NUTS_2013_PATH = ROOT_DIR / "data" / "nuts" / "NUTS_RG_03M_2013_4326_LEVL_2.geojson"

# Europe bounding box
EUROPE_BBOX = (-25.0, 34.0, 45.0, 72.0)

MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]

# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Compute NUTS2 monthly GPP profiles from FluxCom")
    p.add_argument("--gpp-dir",   type=Path, default=GPP_DIR)
    p.add_argument("--rates",     type=Path, default=RATES_CSV)
    p.add_argument("--nuts-path", type=Path, default=NUTS_2013_PATH,
                   help="Path to NUTS2 GeoJSON file (default: local NUTS 2013)")
    return p.parse_args()

# ── Step 1–3: Load and aggregate to monthly climatology ───────────────────────

def load_monthly_climatology(gpp_dir: Path) -> xr.DataArray:
    """
    Load FluxCom daily GPP files, clip to Europe, return monthly climatology.
    Output dims: (month=12, lat, lon), values in gC m⁻² day⁻¹.
    """
    nc_files = sorted(gpp_dir.glob("GPP.RS_METEO*.nc"))
    if not nc_files:
        raise FileNotFoundError(
            f"No FluxCom files found in {gpp_dir}\n"
            "Run download_fluxcom.py first."
        )
    print(f"Loading {len(nc_files)} file(s): {[f.name for f in nc_files]}")

    # FluxCom uses reference date 1582-10-15 which overflows pandas/cftime.
    # Fix: decode_times=False + drop time_bnds, then assign a clean date range
    # from the year embedded in each filename.
    def open_one(path):
        year = int(path.stem.split(".")[-1])
        ds = xr.open_dataset(
            path, engine="netcdf4",
            decode_times=False,
            drop_variables=["time_bnds"],
        )
        n_days = ds.sizes["time"]
        ds = ds.assign_coords(
            time=pd.date_range(f"{year}-01-01", periods=n_days, freq="D")
        )
        return ds

    ds = xr.concat([open_one(f) for f in nc_files], dim="time")

    # variable name varies slightly between versions
    gpp_var = next((v for v in ["GPP", "gpp"] if v in ds), None)
    if gpp_var is None:
        raise KeyError(f"GPP variable not found. Available: {list(ds.data_vars)}")
    gpp = ds[gpp_var]

    # normalise coordinate names
    rename = {}
    for dim in gpp.dims:
        if dim.lower() in ("latitude", "lat"):   rename[dim] = "lat"
        elif dim.lower() in ("longitude", "lon"): rename[dim] = "lon"
    if rename:
        gpp = gpp.rename(rename)

    # clip to Europe (lat stored N→S in FluxCom)
    lon_min, lat_min, lon_max, lat_max = EUROPE_BBOX
    gpp_eu = gpp.sel(lat=slice(lat_max, lat_min), lon=slice(lon_min, lon_max))
    if gpp_eu.lat.size == 0:   # try S→N order
        gpp_eu = gpp.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))

    print(f"  Europe subset: {gpp_eu.lat.size} lat × {gpp_eu.lon.size} lon "
          f"× {gpp_eu.time.size} days")

    # daily → monthly climatology (mean across all years and all days in that month)
    print("  Aggregating daily → monthly climatology ...")
    clim = gpp_eu.groupby("time.month").mean(dim="time")
    print(f"  Climatology shape: {dict(clim.sizes)}")
    return clim   # dims: (month=12, lat, lon)

# ── Step 4: Zonal mean per NUTS2 ──────────────────────────────────────────────

def load_nuts2(source: Path | str) -> gpd.GeoDataFrame:
    print(f"Loading NUTS2 polygons from: {source}")
    nuts = gpd.read_file(str(source))
    if "LEVL_CODE" in nuts.columns:
        nuts = nuts[nuts["LEVL_CODE"] == 2].copy()
    nuts = nuts[["NUTS_ID", "geometry"]].to_crs("EPSG:4326")
    print(f"  {len(nuts)} NUTS2 regions.")
    return nuts


def zonal_mean(clim: xr.DataArray, nuts: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Point-in-polygon: assign each 0.5° grid cell centre to a NUTS2 region,
    then average GPP over all cells in that region per month.
    Falls back to nearest cell for very small regions (e.g. city-states).
    """
    lons = clim.lon.values
    lats = clim.lat.values
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    lon_flat = lon_grid.ravel()
    lat_flat = lat_grid.ravel()

    cells = gpd.GeoDataFrame(
        {"li": np.repeat(np.arange(len(lats)), len(lons)),
         "lo": np.tile(np.arange(len(lons)), len(lats))},
        geometry=[Point(x, y) for x, y in zip(lon_flat, lat_flat)],
        crs="EPSG:4326",
    )

    print("  Spatial join (grid cells → NUTS2) ...")
    joined = gpd.sjoin(cells, nuts[["NUTS_ID", "geometry"]], how="left", predicate="within")

    records = []
    for nuts_id in nuts["NUTS_ID"]:
        sel = joined[joined["NUTS_ID"] == nuts_id]

        if sel.empty:
            # nearest cell fallback for tiny regions (e.g. city-states, small islands)
            ctr = nuts.loc[nuts["NUTS_ID"] == nuts_id, "geometry"].values[0].centroid
            dist = (lon_flat - ctr.x)**2 + (lat_flat - ctr.y)**2
            idx = int(np.argmin(dist))
            li, lo = idx // len(lons), idx % len(lons)
            monthly = {m: float(clim.sel(month=m).values[li, lo]) for m in range(1, 13)}
        else:
            lis = sel["li"].astype(int).values
            los = sel["lo"].astype(int).values
            monthly = {}
            for m in range(1, 13):
                vals = clim.sel(month=m).values[lis, los]
                vals = vals[np.isfinite(vals) & (vals > 0)]
                monthly[m] = float(np.mean(vals)) if len(vals) > 0 else np.nan

        records.append({"NUTS_ID": nuts_id, **monthly})

    df = pd.DataFrame(records).set_index("NUTS_ID")
    df.columns = pd.Index(range(1, 13), name="month")
    print(f"  Zonal stats done: {df.shape}")
    return df

# ── Step 5: Normalise with fallback ───────────────────────────────────────────

def normalise_with_fallback(df: pd.DataFrame, nuts: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Normalise GPP monthly values to weights that sum to 1.0 per region.

    Partial-NaN handling: some regions have NaN for winter months (GPP ≈ 0
    filtered out in zonal_mean). These NaN months are treated as zero growth,
    and the row is re-normalised.

    Fallback strategy for all-NaN regions:
      1. Distance-based neighbour mean — mean profile of NUTS2 regions whose
         boundary lies within 100 km (EPSG:3035); iterated until stable.
         The 100 km threshold captures sea crossings (Channel, Irish Sea, etc.)
         so that e.g. southern England can draw on northern France.
      2. Country mean — mean profile of all resolved regions in the same country.
      3. Uniform 1/12 — last resort for fully isolated regions.
    """
    M = list(range(1, 13))  # integer month columns

    # Identify all-NaN rows (no GPP data at all) before any processing
    all_nan_mask = df[M].isna().all(axis=1)

    # For partial-NaN rows: NaN means GPP ≈ 0 was filtered → treat as 0 and re-normalise
    df_filled = df[M].copy()
    partial_nan_mask = df[M].isna().any(axis=1) & ~all_nan_mask
    if partial_nan_mask.any():
        df_filled.loc[partial_nan_mask] = df_filled.loc[partial_nan_mask].fillna(0.0)
        print(f"  {partial_nan_mask.sum()} regions had partial NaN (winter months → 0).")

    # Normalise: divide each row by its annual sum
    row_sums = df_filled.sum(axis=1)
    weights = df_filled.div(row_sums, axis=0)

    # Reset all-NaN rows to NaN (to be filled by fallbacks below)
    weights.loc[all_nan_mask] = np.nan

    n_missing = int(all_nan_mask.sum())
    if n_missing > 0:
        print(f"  {n_missing} regions have no GPP data — applying fallbacks ...")

    # ── Fallback 1: Distance-based neighbour mean, iterated (100 km) ─────────
    if n_missing > 0:
        SEA_THRESHOLD_M = 100_000
        nuts_idx = nuts.set_index("NUTS_ID")
        gdf = nuts_idx[["geometry"]].to_crs(3035)
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
            for rid in list(weights.index[weights[M].isna().all(axis=1)]):
                valid_nbrs = [
                    n for n in neighbors_dict.get(rid, [])
                    if n in weights.index and not weights.loc[n, M].isna().all()
                ]
                if valid_nbrs:
                    weights.loc[rid, M] = weights.loc[valid_nbrs, M].mean().values
                    changed = True

        prev = n_missing
        n_missing = int(weights[M].isna().all(axis=1).sum())
        if n_missing < prev:
            print(f"    Distance neighbour fallback resolved {prev - n_missing} regions "
                  f"({n_missing} still missing).")

    # ── Fallback 2: Country (NUTS0) mean ──────────────────────────────────────
    if n_missing > 0:
        known = weights[~weights[M].isna().all(axis=1)]
        cntr_mean = known.groupby(known.index.str[:2])[M].mean()
        for rid in list(weights.index[weights[M].isna().all(axis=1)]):
            key = rid[:2]
            if key in cntr_mean.index:
                weights.loc[rid, M] = cntr_mean.loc[key].values

        prev = n_missing
        n_missing = int(weights[M].isna().all(axis=1).sum())
        if n_missing < prev:
            print(f"    Country fallback resolved {prev - n_missing} regions "
                  f"({n_missing} still missing).")

    # ── Fallback 3: Uniform 1/12 (last resort) ────────────────────────────────
    if n_missing > 0:
        for rid in list(weights.index[weights[M].isna().all(axis=1)]):
            weights.loc[rid, M] = [1.0 / 12] * 12
        print(f"    Applied uniform 1/12 to {n_missing} remaining regions.")

    weights = weights[M]
    assert np.allclose(weights.sum(axis=1), 1.0, atol=1e-6), \
        "Row-sum check failed after normalisation + fallbacks"
    return weights

# ── Step 6: Merge with annual growth rates ────────────────────────────────────

def compute_monthly_rates(weights: pd.DataFrame, rates_csv: Path) -> pd.DataFrame | None:
    """
    monthly_rate [tCO₂ ha⁻¹ month⁻¹] = weight × annual_rate [tCO₂ ha⁻¹ yr⁻¹]

    Handles both column-name conventions:
      - zenodo_aCDRs output: NUTS2 column = 'NUTS2', value column = 'affo rate (t/ha/y)'
      - legacy Pilli output: NUTS2 column = 'nuts2', value column = 'mai_co2_mean'
    """
    if not rates_csv.exists():
        print(f"  Rates CSV not found at {rates_csv} — skipping monthly rates.")
        return None

    rates = pd.read_csv(rates_csv)

    # Determine ID column
    if "NUTS2" in rates.columns:
        id_col = "NUTS2"
    elif "nuts2" in rates.columns:
        id_col = "nuts2"
    else:
        id_col = rates.columns[0]

    # Determine value column
    if "CO2 seq rate tCO2/(ha y)" in rates.columns:
        val_col = "CO2 seq rate tCO2/(ha y)"
    elif "affo rate (t/ha/y)" in rates.columns:
        val_col = "affo rate (t/ha/y)"
    elif "mai_co2_mean" in rates.columns:
        val_col = "mai_co2_mean"
    else:
        # last resort: pick first numeric column after the ID column
        numeric_cols = rates.select_dtypes(include="number").columns.tolist()
        if numeric_cols:
            val_col = numeric_cols[0]
            print(f"  Warning: guessing value column as '{val_col}'")
        else:
            print(f"  ERROR: no numeric value column found in {rates_csv}. Skipping.")
            return None

    rates = rates.set_index(id_col)[val_col]
    rates.index.name = "NUTS_ID"

    common = weights.index.intersection(rates.index)
    only_in_weights = weights.index.difference(rates.index)
    only_in_rates   = rates.index.difference(weights.index)

    print(f"  Rates CSV: {len(rates)} regions  |  Weights: {len(weights)}  |  Common: {len(common)}")
    if len(only_in_weights):
        print(f"  In weights but NOT in rates ({len(only_in_weights)}): "
              f"{sorted(only_in_weights)[:10]}")
    if len(only_in_rates):
        print(f"  In rates but NOT in weights ({len(only_in_rates)}): "
              f"{sorted(only_in_rates)[:10]}")

    monthly_rates = weights.loc[common].mul(rates.loc[common], axis=0)
    monthly_rates.columns = MONTH_NAMES
    return monthly_rates

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if not args.nuts_path.exists():
        raise FileNotFoundError(
            f"NUTS2 GeoJSON not found: {args.nuts_path}\n"
            "Pass --nuts-path to the correct NUTS 2013 GeoJSON file."
        )

    # 1–3. Load + climatology
    clim = load_monthly_climatology(args.gpp_dir)

    # 4. NUTS2 zonal stats
    nuts = load_nuts2(args.nuts_path)
    df_gpp = zonal_mean(clim, nuts)

    # 5. Normalise with fallback strategy (integer month columns 1–12)
    print("Normalising to monthly weights (with spatial fallbacks) ...")
    weights_int = normalise_with_fallback(df_gpp, nuts)

    # Rename integer month columns to month names for CSV output
    weights = weights_int.rename(columns={i: n for i, n in enumerate(MONTH_NAMES, 1)})

    out_w = OUTPUT_DIR / "afforestation_nuts2_monthly_weights.csv"
    weights.to_csv(out_w)
    print(f"  Saved: {out_w}")

    # 6. Monthly rates
    print("Computing monthly tCO₂ rates ...")
    monthly_rates = compute_monthly_rates(weights_int.copy(), args.rates)
    if monthly_rates is not None:
        out_r = OUTPUT_DIR / "afforestation_nuts2_monthly_rates.csv"
        monthly_rates.to_csv(out_r)
        print(f"  Saved: {out_r}")

    # ── Step 7: Western Balkans pseudo-NUTS2 (RS/AL/BA/XK absent from Eurostat GeoJSON) ─
    # These countries have no Eurostat NUTS2 regions, but are present in PyPSA-EUR.
    # Use neighbour-country mean monthly weight profile (same cascade as script 01 Step 7).
    EXTRA_PYPSA = {
        "RS00": ["HR", "HU", "RO", "BG"],
        "AL00": ["EL", "MK"],
        "BA00": ["HR", "ME"],
        "XK00": ["MK", "RS00"],  # RS00 processed first → available in extra_w_rows
    }

    # Load annual rates for pseudo-codes from the full rates CSV
    extra_annual = {}
    if args.rates.exists():
        annual_df = pd.read_csv(args.rates)
        id_col_a = "NUTS2" if "NUTS2" in annual_df.columns else annual_df.columns[0]
        if "CO2 seq rate tCO2/(ha y)" in annual_df.columns:
            val_col_a = "CO2 seq rate tCO2/(ha y)"
        elif "affo rate (t/ha/y)" in annual_df.columns:
            val_col_a = "affo rate (t/ha/y)"
        else:
            numeric_a = annual_df.select_dtypes(include="number").columns
            val_col_a = numeric_a[0] if len(numeric_a) else None
        if val_col_a:
            annual_ser = annual_df.set_index(id_col_a)[val_col_a]
            for pseudo in EXTRA_PYPSA:
                if pseudo in annual_ser.index:
                    extra_annual[pseudo] = float(annual_ser[pseudo])

    extra_w_rows = {}  # pseudo_code → Series indexed by MONTH_NAMES
    extra_r_rows = {}  # pseudo_code → Series indexed by MONTH_NAMES

    for pseudo_code, nbr_countries in EXTRA_PYPSA.items():
        profiles = []
        for cc in nbr_countries:
            if cc in extra_w_rows:
                # Already-computed pseudo-code (e.g. RS00 used for XK00)
                profiles.append(extra_w_rows[cc])
            else:
                mask = weights.index.str[:2] == cc
                if mask.any():
                    profiles.append(weights.loc[mask].mean())

        if profiles:
            profile = pd.concat(profiles, axis=1).mean(axis=1)
            profile = profile / profile.sum()  # ensure sums to 1
        else:
            profile = pd.Series([1.0 / 12] * 12, index=MONTH_NAMES)

        extra_w_rows[pseudo_code] = profile

        if pseudo_code in extra_annual:
            extra_r_rows[pseudo_code] = profile * extra_annual[pseudo_code]

    if extra_w_rows:
        extra_w_df = pd.DataFrame(extra_w_rows).T
        extra_w_df.index.name = "NUTS_ID"
        weights_full = pd.concat([weights, extra_w_df])
        weights_full.to_csv(out_w)
        print(f"  Step 7: appended {len(extra_w_rows)} Western Balkans weights → {out_w} "
              f"({len(weights_full)} total rows)")

    if extra_r_rows and monthly_rates is not None:
        extra_r_df = pd.DataFrame(extra_r_rows).T
        extra_r_df.index.name = "NUTS_ID"
        monthly_rates_full = pd.concat([monthly_rates, extra_r_df])
        monthly_rates_full.to_csv(out_r)
        print(f"  Step 7: appended {len(extra_r_rows)} Western Balkans rates → {out_r} "
              f"({len(monthly_rates_full)} total rows)")

    # Sanity check
    print("\n── Row-sum check (all should be 1.0) ──")
    print(weights.sum(axis=1).describe().to_string())
    print("\n── Sample weights (5 regions) ──")
    print(weights.head())
    if monthly_rates is not None:
        print("\n── Sample monthly rates [tCO₂/ha/month] (5 regions) ──")
        print(monthly_rates.head())


if __name__ == "__main__":
    main()
