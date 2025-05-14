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

# Target save directory
resources_folder = os.path.join(os.getcwd(), "resources/forests")
os.makedirs(resources_folder, exist_ok=True)

# File name and path
file_name = "Biomass_calculations.xlsx"
file_path = os.path.join(resources_folder, file_name)

# Figshare file URL
url = "https://figshare.com/ndownloader/files/43678533"

# Start a session to handle redirects
session = requests.Session()
response = session.get(url, allow_redirects=True)

# Follow redirects to final download
final_url = response.url
file_response = session.get(final_url)

# Save to the specified path
with open(file_path, "wb") as f:
    f.write(file_response.content)

print(f"Downloaded to: {file_path}")

