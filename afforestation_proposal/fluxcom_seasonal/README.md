# FluxCom Seasonal GPP Profiles for NUTS2 Afforestation

Derives monthly carbon sequestration weights per NUTS2 region from the
FluxCom RS ensemble (Jung et al. 2020), to capture within-year seasonality
of forest carbon uptake for use alongside the Pilli et al. annual rates.

## Outputs

| File | Description |
|------|-------------|
| `output/nuts2_monthly_weights.csv` | 12 monthly weights per NUTS2, each row sums to 1.0 |
| `output/nuts2_monthly_rates.csv` | Monthly tCO₂ ha⁻¹ month⁻¹ (weights × Pilli annual rate) |
| `output/fig_monthly_profiles_sample.png` | Line plot for 6 representative regions |
| `output/fig_seasonal_map.png` | 12-panel map across Europe |

## Steps

### 1. Get FluxCom data

Register at http://www.fluxcom.org/CF-Download/ and download the RS V006
monthly GPP files (one NetCDF per year, ~150 MB each). Place them in
`data/fluxcom_raw/` named as `GPP_fluxcom_monthly_YYYY.nc`.

Or run the download script:
```bash
export FLUXCOM_USER="your_username"
export FLUXCOM_PASS="your_password"
python 01_download_fluxcom.py
```

### 2. Compute NUTS2 profiles

```bash
python 02_compute_nuts2_profiles.py
```

Requires: `geopandas`, `xarray`, `numpy`, `pandas`, `shapely`

### 3. Visual check

```bash
python 03_plot_check.py
```

Requires additionally: `matplotlib`

## Method summary (for Zenodo / paper)

> Monthly sequestration weights per NUTS2 were derived from the FluxCom
> RS V006 ensemble (Jung et al. 2020), which provides machine-learning
> upscaled gross primary production (GPP) at 0.5° resolution and monthly
> timestep, trained on FLUXNET eddy-covariance tower observations. A
> multi-year climatology (2008–2012) was computed and spatially averaged
> over each NUTS2 polygon. The resulting 12-month GPP profile was
> normalised to sum to unity, yielding dimensionless monthly weights that
> distribute the annual sequestration rate (Pilli et al. 2024) across the
> calendar year.

## Reference

Jung, M. et al. (2020). Scaling carbon fluxes from eddy covariance sites
to globe: synthesis and evaluation of the FLUX COM approach.
*Biogeosciences*, 17(5), 1343–1365.
https://doi.org/10.5194/bg-17-1343-2020
