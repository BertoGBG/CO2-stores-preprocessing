# SPDX-FileCopyrightText: 2024 Alberto Alamia <alamia@mpe.au.dk>
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Retrieve CBM - JRC data about forest stocks
reference: 10.2760/222407

Author: Alberto Alamia
Date: 2025-04-07
"""

import os
import requests
import zipfile
import io

# Download URL
zenodo_url = "https://zenodo.org/record/11387301/files/Volume_increment_database.zip"

# Desired files and custom save names
files_to_extract = {
    "NAI_evenaged_stands.csv": "NAI_evenaged_stands.csv",
    "Regions_codes.csv": "NAI_regions_codes.csv",
    "Standing_stock_evenaged.csv": "Standing_stock_evenaged.csv",
}

# Target save directory
resources_folder = os.path.join(os.getcwd(), "resources/forests")
os.makedirs(resources_folder, exist_ok=True)

# Download the ZIP
print("Downloading ZIP file...")
response = requests.get(zenodo_url)
response.raise_for_status()

# Open ZIP from memory
with zipfile.ZipFile(io.BytesIO(response.content)) as z:
    zip_file_list = z.namelist()
    print("Files in ZIP:")
    for name in zip_file_list:
        print(f" - {name}")

    for original_name, save_as in files_to_extract.items():
        matched_file = next(
            (f for f in zip_file_list if f.endswith(original_name)), None
        )

        if matched_file:
            output_path = os.path.join(resources_folder, save_as)
            print(f"Extracting '{matched_file}' and saving as '{save_as}'...")
            with (
                z.open(matched_file) as source_file,
                open(output_path, "wb") as target_file,
            ):
                target_file.write(source_file.read())
        else:
            print(f"Could not find '{original_name}' in ZIP!")

print("forest data extracted and saved.")
