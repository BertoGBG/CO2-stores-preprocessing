# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Compute conversion factor from 1G biofuels (MWh/y) to perennials (tDM/y) potentials for each clustered model region
using data from JRC ENSPRESO.
"""

import logging

import geopandas as gpd
import pandas as pd
from _helpers import configure_logging, set_scenario_config

logger = logging.getLogger(__name__)


def build_nuts2_shapes():
    """
    - load NUTS2 geometries
    - add RS, AL, BA country shapes (not covered in NUTS 2013)
    - consistently name ME, MK
    """
    nuts2 = gpd.GeoDataFrame(
        gpd.read_file(snakemake.input.nuts2).set_index("NUTS_ID").geometry
    )

    countries = gpd.read_file(snakemake.input.country_shapes).set_index("name")
    missing_iso2 = countries.index.intersection(["AL", "RS", "XK", "BA"])
    missing = countries.loc[missing_iso2]

    nuts2.rename(index={"ME00": "ME", "MK00": "MK"}, inplace=True)

    return pd.concat([nuts2, missing])


def area(gdf):
    return gdf.to_crs(epsg=3035).area.div(1e6)


def convert_nuts2_to_regions(bio_nuts2, regions):
    return bio_regions


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "build_perennials_potentials",
            clusters="39",
            planning_horizons=2050,
        )

    configure_logging(snakemake)
    set_scenario_config(snakemake)

    nuts2 = build_nuts2_shapes()

    df_nuts2 = gpd.GeoDataFrame(nuts2.geometry).join(yields_perennials_1G_biofuels)

    regions = gpd.read_file(snakemake.input.regions_onshore)

    df = convert_nuts2_to_regions(df_nuts2, regions)

    df.to_csv(snakemake.output.perennials_potentials)



######### RULES ########

rule build_perennial_potentials:
    params:
        biomass=config_provider("biomass"),
    input:
        yields_perennials_1G_biofuels = resources("yields_perennials_1G_biofuels.csv")
        regions_onshore = resources("regions_onshore_base_s_{clusters}.geojson"),
    output:
        csv_file = resources("conversion_1Gbiofuels_to_perennials.csv"),
    log:
        logs("build_perennial_potentials_s_{clusters}.log"),
    resources:
        mem_mb = 1000,
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/build_perennial_potentials.py"


rule get_perennial_potential_nuts_file:
    input:
    output:
        perennial_nuts_file = resources("yields_perennials_1G_biofuels.csv")
    log:
        logs("get_perennial_potential_nuts_file.log"),
    resources:
        mem_mb = 1000,
    shell:
        "wget https://raw.githubusercontent.com/BertoGBG/CO2-stores-preprocessing/refs/heads/main/afforestation_perennials/data/crops/yields_perennials_1G_biofuels.csv -O resources/yields_perennials_1G_biofuels.csv"

