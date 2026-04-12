"""
Download FluxCom RS+METEO daily GPP data (ERA5 forcing) via anonymous FTP.

Dataset: FluxCom RS_METEO ensemble, ERA5 forcing, GPP variable
  FTP:  ftp.bgc-jena.mpg.de (anonymous, no credentials needed)
  Path: /pub/outgoing/uweber/Fluxcom/tCarbonFluxes/RS_METEO/ensemble/ERA5/daily/
  File: GPP.RS_METEO.FP-ALL.MLM-ALL.METEO-ERA5.720_360.daily.YYYY.nc
  Unit: gC m⁻² day⁻¹  |  Resolution: 0.5° global  |  Period: 1979–2020
  Size: ~1.2 GB per year

We download 3 years by default — sufficient for a stable seasonal climatology.
Files are saved to data/fluxcom_raw/ for processing by 02_compute_nuts2_profiles.py,
and will be uploaded to Zenodo alongside the Pilli et al. afforestation rates.

Reference:
  Jung, M. et al. (2020). Scaling carbon fluxes from eddy covariance sites to
  globe: synthesis and evaluation of the FLUX COM approach.
  Biogeosciences, 17(5), 1343–1365. https://doi.org/10.5194/bg-17-1343-2020

Usage:
    python download_fluxcom.py            # downloads 2010, 2011, 2012
    python download_fluxcom.py --years 2008 2009 2010 2011 2012
"""

import argparse
import urllib.request
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────────

FTP_BASE = (
    "ftp://ftp.bgc-jena.mpg.de/pub/outgoing/uweber/Fluxcom/"
    "tCarbonFluxes/RS_METEO/ensemble/ERA5/daily/"
)
FILE_PATTERN = "GPP.RS_METEO.FP-ALL.MLM-ALL.METEO-ERA5.720_360.daily.{year}.nc"

DEFAULT_YEARS = [2010, 2011, 2012]   # 3 years → stable climatology, ~3.6 GB

ROOT_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT_DIR / "data" / "fluxcom_raw"
# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Download FluxCom GPP via anonymous FTP")
    p.add_argument(
        "--years", nargs="+", type=int, default=DEFAULT_YEARS,
        help=f"Years to download (default: {DEFAULT_YEARS}). Available: 1979–2020.",
    )
    return p.parse_args()

# ── Download ───────────────────────────────────────────────────────────────────

def _progress(block_count, block_size, total_size):
    downloaded = block_count * block_size
    if total_size > 0:
        pct = min(100, 100 * downloaded / total_size)
        mb = downloaded / 1e6
        total_mb = total_size / 1e6
        print(f"\r    {pct:5.1f}%  {mb:.0f} / {total_mb:.0f} MB", end="", flush=True)


MIN_FILE_SIZE_MB = 500  # FluxCom annual files are ~1.2 GB; treat <500 MB as truncated


def download_year(year: int) -> Path:
    filename = FILE_PATTERN.format(year=year)
    url = FTP_BASE + filename
    out_path = OUTPUT_DIR / filename

    if out_path.exists():
        size_mb = out_path.stat().st_size / 1e6
        if size_mb >= MIN_FILE_SIZE_MB:
            print(f"  {year}: already downloaded ({size_mb:.0f} MB), skipping.")
            return out_path
        else:
            print(f"  {year}: existing file looks truncated ({size_mb:.0f} MB < {MIN_FILE_SIZE_MB} MB) — re-downloading.")
            out_path.unlink()

    print(f"  {year}: downloading {filename}")
    print(f"         from {url}")
    urllib.request.urlretrieve(url, out_path, reporthook=_progress)
    print(f"\n         done → {out_path.stat().st_size / 1e6:.0f} MB")
    return out_path


def main():
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"FluxCom GPP download — {len(args.years)} year(s): {args.years}")
    print(f"Output directory: {OUTPUT_DIR.resolve()}\n")

    for year in args.years:
        download_year(year)

    print(f"\nDone. Next step: run 02_compute_nuts2_profiles.py")


if __name__ == "__main__":
    main()
