"""
Download and extract the Pilli et al. (2024) JRC forest growth library from Zenodo.

Source:  https://zenodo.org/records/11387301
Archive: https://zenodo.org/api/records/11387301/files-archive

The zip is downloaded into memory (no zip file saved to disk) and extracted
directly to the output directory, preserving the internal folder structure:
  data/zenodo_Pilli/
    Volume_increment_database/   — standing stock, NAI, region/forest codes
    Volume_biomass_bcef_database/ — BCEF lookup table
    Volume_biomass_selection/    — auxiliary biomass selection tables

Usage:
    python download_zenodo_pilli_afforestation.py
"""

import io
import zipfile
from pathlib import Path

import requests
import urllib3

# DTU (and many university) networks use an SSL-inspecting proxy whose certificate
# does not match external hostnames. Disable verification and suppress the warning.
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

ZENODO_ARCHIVE_URL = "https://zenodo.org/api/records/11387301/files-archive"

ROOT_DIR   = Path(__file__).resolve().parents[2]          # zenodo_aCDRs/
OUTPUT_DIR = ROOT_DIR / "data" / "zenodo_Pilli"


def download_bytes(url: str) -> bytes:
    """Stream-download url and return raw bytes, with a progress indicator."""
    print(f"Downloading {url}")
    response = requests.get(url, timeout=300, stream=True, allow_redirects=True, verify=False)
    response.raise_for_status()

    chunks = []
    downloaded = 0
    for chunk in response.iter_content(chunk_size=1 << 20):  # 1 MB chunks
        chunks.append(chunk)
        downloaded += len(chunk)
        print(f"  {downloaded / 1024**2:.1f} MB received ...", end="\r")
    print(f"  {downloaded / 1024**2:.1f} MB downloaded.       ")

    data = b"".join(chunks)
    if not data:
        raise RuntimeError(f"Download returned empty content from {url}")
    return data


def strip_top_level(name: str) -> str:
    """
    Remove a single common top-level directory from a zip member path.

    Zenodo archives often wrap everything in a '<record>-<version>/' prefix.
    E.g. 'pilli-2024-v2/Volume_increment_database/foo.csv'
         → 'Volume_increment_database/foo.csv'
    If the path has no directory component, it is returned unchanged.
    """
    parts = Path(name).parts
    return str(Path(*parts[1:])) if len(parts) > 1 else name


def extract_zip(data: bytes, output_dir: Path, _label: str = "outer") -> None:
    """
    Extract zip bytes to output_dir, stripping any top-level wrapper folder.
    Any .zip files found inside are extracted recursively (in memory) and then
    discarded — no zip files are written to disk.
    """
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        members = zf.namelist()
        print(f"  [{_label}] archive contains {len(members)} entries.")

        # Detect whether all members share a common top-level directory
        top_dirs = {Path(m).parts[0] for m in members if Path(m).parts}
        strip = len(top_dirs) == 1  # single top-level folder → strip it

        for member in members:
            if member.endswith("/"):
                continue  # skip directory entries

            dest_rel = strip_top_level(member) if strip else member

            with zf.open(member) as src:
                inner_bytes = src.read()

            # If the entry is itself a zip, extract it recursively into the
            # subdirectory that matches its name (without the .zip extension)
            if member.lower().endswith(".zip"):
                inner_name = Path(dest_rel).stem   # e.g. "Volume_increment_database"
                inner_dir  = output_dir / inner_name
                inner_dir.mkdir(parents=True, exist_ok=True)
                print(f"  Recursing into {Path(member).name} → {inner_dir.name}/")
                extract_zip(inner_bytes, inner_dir, _label=Path(member).stem)
            else:
                dest = output_dir / dest_rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(inner_bytes)

        print(f"  [{_label}] done → {output_dir}")


def main() -> None:
    print(f"Output directory: {OUTPUT_DIR}")

    data = download_bytes(ZENODO_ARCHIVE_URL)
    extract_zip(data, OUTPUT_DIR)

    # Quick sanity check
    expected = [
        OUTPUT_DIR / "Volume_increment_database" / "Standing_stock_evenaged.csv",
        OUTPUT_DIR / "Volume_increment_database" / "Regions_codes.csv",
        OUTPUT_DIR / "Volume_biomass_bcef_database" / "Vol_to_biomass_bcef_database.csv",
    ]
    missing = [p for p in expected if not p.exists()]
    if missing:
        print(f"[warn] Expected files not found after extraction:")
        for p in missing:
            print(f"  {p}")
    else:
        print("[ok] Key files present — download complete.")


if __name__ == "__main__":
    main()
