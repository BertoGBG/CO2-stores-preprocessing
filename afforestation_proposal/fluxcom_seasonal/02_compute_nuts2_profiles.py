"""
Compute monthly GPP seasonal profiles per NUTS2 and merge with Pilli et al. rates.

Pipeline:
  1. Load FluxCom daily GPP NetCDF files (gC m⁻² day⁻¹)
  2. Clip to Europe bounding box
  3. Aggregate: daily → monthly climatology (mean across all years and days per month)
  4. Zonal mean per NUTS2 polygon for each month
  5. Normalise: each region's 12-month profile sums to 1.0
  6. Merge with afforestation_rates_nuts2.csv (Pilli et al. MAI_CO2_mean)
     → monthly rates in tCO₂ ha⁻¹ month⁻¹ per NUTS2

Outputs:
  output/nuts2_monthly_weights.csv   — NUTS2 × 12 weights (sum = 1 per row)
  output/nuts2_monthly_rates.csv     — NUTS2 × 12 monthly tCO₂ ha⁻¹ month⁻¹

Reference:
  Jung, M. et al. (2020). Biogeosciences, 17(5), 1343–1365.
  https://doi.org/10.5194/bg-17-1343-2020

Usage:
    python 02_compute_nuts2_profiles.py
    python 02_compute_nuts2_profiles.py --gpp-dir data/fluxcom_raw/ --rates ../output/afforestation_rates_nuts2.csv
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

HERE       = Path(__file__).parent
GPP_DIR    = HERE / "data" / "fluxcom_raw"
OUTPUT_DIR = HERE / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

RATES_CSV = HERE.parent / "output" / "afforestation_rates_nuts2.csv"

# Europe bounding box
EUROPE_BBOX = (-25.0, 34.0, 45.0, 72.0)

MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]

NUTS_URL = (
    "https://gisco-services.ec.europa.eu/distribution/v2/nuts/geojson/"
    "NUTS_RG_10M_2021_4326_LEVL_2.geojson"
)

# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Compute NUTS2 monthly GPP profiles from FluxCom")
    p.add_argument("--gpp-dir", type=Path, default=GPP_DIR)
    p.add_argument("--rates",   type=Path, default=RATES_CSV)
    p.add_argument("--nuts-url", default=NUTS_URL)
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
            "Run 01_download_fluxcom.py first."
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

def load_nuts2(source: str) -> gpd.GeoDataFrame:
    print(f"Loading NUTS2 polygons ...")
    nuts = gpd.read_file(source)
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
            # nearest cell fallback for tiny regions
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

# ── Step 5: Normalise ─────────────────────────────────────────────────────────

def normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Divide each row by its annual sum → weights sum to 1.0 per region."""
    weights = df.div(df.sum(axis=1), axis=0)
    # regions with no data (islands/sea) → uniform 1/12
    uniform = pd.Series([1/12]*12, index=df.columns)
    weights = weights.apply(lambda r: uniform if r.isna().all() else r, axis=1)
    assert np.allclose(weights.sum(axis=1).dropna(), 1.0, atol=1e-6)
    return weights

# ── Step 6: Merge with Pilli annual rates ─────────────────────────────────────

def compute_monthly_rates(weights: pd.DataFrame, rates_csv: Path) -> pd.DataFrame | None:
    """
    monthly_rate [tCO₂ ha⁻¹ month⁻¹] = weight × MAI_CO2_mean [tCO₂ ha⁻¹ yr⁻¹]
    """
    if not rates_csv.exists():
        print(f"  Rates CSV not found at {rates_csv} — skipping monthly rates.")
        return None

    rates = pd.read_csv(rates_csv)

    # handle both 'nuts2' and 'NUTS_ID' column names
    id_col = "nuts2" if "nuts2" in rates.columns else "NUTS_ID"
    rates = rates.set_index(id_col)["mai_co2_mean"]
    rates.index.name = "NUTS_ID"

    common = weights.index.intersection(rates.index)
    print(f"  Merging {len(common)} common NUTS2 regions with Pilli rates.")

    monthly_rates = weights.loc[common].mul(rates.loc[common], axis=0)
    monthly_rates.columns = MONTH_NAMES
    return monthly_rates

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # 1–3. Load + climatology
    clim = load_monthly_climatology(args.gpp_dir)

    # 4. NUTS2 zonal stats
    nuts = load_nuts2(args.nuts_url)
    df_gpp = zonal_mean(clim, nuts)

    # 5. Normalise
    print("Normalising to monthly weights ...")
    weights = normalise(df_gpp)
    weights.columns = MONTH_NAMES

    out_w = OUTPUT_DIR / "nuts2_monthly_weights.csv"
    weights.to_csv(out_w)
    print(f"  Saved: {out_w}")

    # 6. Monthly rates
    print("Computing monthly tCO₂ rates ...")
    monthly_rates = compute_monthly_rates(
        weights.copy().rename(columns={n: i+1 for i, n in enumerate(MONTH_NAMES)}),
        args.rates,
    )
    if monthly_rates is not None:
        out_r = OUTPUT_DIR / "nuts2_monthly_rates.csv"
        monthly_rates.to_csv(out_r)
        print(f"  Saved: {out_r}")

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
