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
        "../scripts/build_perennials_potentials.py"


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



rule prepare_sector_network:
    params:
        time_resolution=config_provider("clustering", "temporal", "resolution_sector"),
        co2_budget=config_provider("co2_budget"),
        conventional_carriers=config_provider(
            "existing_capacities", "conventional_carriers"
        ),
        foresight=config_provider("foresight"),
        costs=config_provider("costs"),
        sector=config_provider("sector"),
        industry=config_provider("industry"),
        renewable=config_provider("renewable"),
        lines=config_provider("lines"),
        pypsa_eur=config_provider("pypsa_eur"),
        length_factor=config_provider("lines", "length_factor"),
        planning_horizons=config_provider("scenario", "planning_horizons"),
        countries=config_provider("countries"),
        adjustments=config_provider("adjustments", "sector"),
        emissions_scope=config_provider("energy", "emissions"),
        biomass=config_provider("biomass"),
        RDIR=RDIR,
        heat_pump_sources=config_provider("sector", "heat_pump_sources"),
        heat_systems=config_provider("sector", "heat_systems"),
        energy_totals_year=config_provider("energy", "energy_totals_year"),
        heat_utilisation_potentials=config_provider(
            "sector", "district_heating", "heat_utilisation_potentials"
        ),
        direct_utilisation_heat_sources=config_provider(
            "sector", "district_heating", "direct_utilisation_heat_sources"
        ),
    input:
        unpack(input_profile_offwind),
        unpack(input_heat_source_potentials),
        **rules.cluster_gas_network.output,
        **rules.build_gas_input_locations.output,
        snapshot_weightings=resources(
            "snapshot_weightings_base_s_{clusters}_elec_{opts}_{sector_opts}.csv"
        ),
        retro_cost=lambda w: (
            resources("retro_cost_base_s_{clusters}.csv")
            if config_provider("sector", "retrofitting", "retro_endogen")(w)
            else []
        ),
        floor_area=lambda w: (
            resources("floor_area_base_s_{clusters}.csv")
            if config_provider("sector", "retrofitting", "retro_endogen")(w)
            else []
        ),
        biomass_transport_costs=lambda w: (
            resources("biomass_transport_costs.csv")
            if config_provider("sector", "biomass_transport")(w)
            or config_provider("sector", "biomass_spatial")(w)
            else []
        ),
        sequestration_potential=lambda w: (
            resources("co2_sequestration_potential_base_s_{clusters}.csv")
            if config_provider(
                "sector", "regional_co2_sequestration_potential", "enable"
            )(w)
            else []
        ),
        network=resources("networks/base_s_{clusters}_elec_{opts}.nc"),
        eurostat="data/eurostat/Balances-April2023",
        pop_weighted_energy_totals=resources(
            "pop_weighted_energy_totals_s_{clusters}.csv"
        ),
        pop_weighted_heat_totals=resources("pop_weighted_heat_totals_s_{clusters}.csv"),
        shipping_demand=resources("shipping_demand_s_{clusters}.csv"),
        transport_demand=resources("transport_demand_s_{clusters}.csv"),
        transport_data=resources("transport_data_s_{clusters}.csv"),
        avail_profile=resources("avail_profile_s_{clusters}.csv"),
        dsm_profile=resources("dsm_profile_s_{clusters}.csv"),
        co2_totals_name=resources("co2_totals.csv"),
        co2="data/bundle/eea/UNFCCC_v23.csv",
        biomass_potentials=resources(
            "biomass_potentials_s_{clusters}_{planning_horizons}.csv"
        ),
        costs=lambda w: (
            resources("costs_{}.csv".format(config_provider("costs", "year")(w)))
            if config_provider("foresight")(w) == "overnight"
            else resources("costs_{planning_horizons}.csv")
        ),
        biochar_potentials=lambda w: (
            resources("biochar_potentials_s_{clusters}.csv")
            if config_provider("sector", "biochar")(w)
            else []
        ),
        EW_potentials=lambda w: (
            resources("EW_potentials_s_{clusters}.csv")
            if config_provider("sector", "EW")(w)
            else []
        ),
        afforestation_potentials=lambda w: (
            resources("afforestation_potentials_s_{clusters}.csv")
	    if config_provider("sector", "afforestation")(w)
            else []
        ),
        yields_perennials_1G_biofuels = resources(
            "yields_perennials_1G_biofuels_s_{clusters}_{planning_horizons}.csv"
        ),
        h2_cavern=resources("salt_cavern_potentials_s_{clusters}.csv"),
        busmap_s=resources("busmap_base_s.csv"),
        busmap=resources("busmap_base_s_{clusters}.csv"),
        clustered_pop_layout=resources("pop_layout_base_s_{clusters}.csv"),
        industrial_demand=resources(
            "industrial_energy_demand_base_s_{clusters}_{planning_horizons}.csv"
        ),
        hourly_heat_demand_total=resources(
            "hourly_heat_demand_total_base_s_{clusters}.nc"
        ),
        industrial_production=resources(
            "industrial_production_base_s_{clusters}_{planning_horizons}.csv"
        ),
        district_heat_share=resources(
            "district_heat_share_base_s_{clusters}_{planning_horizons}.csv"
        ),
        heating_efficiencies=resources("heating_efficiencies.csv"),
        temp_soil_total=resources("temp_soil_total_base_s_{clusters}.nc"),
        temp_air_total=resources("temp_air_total_base_s_{clusters}.nc"),
        cop_profiles=resources("cop_profiles_base_s_{clusters}_{planning_horizons}.nc"),
        solar_thermal_total=lambda w: (
            resources("solar_thermal_total_base_s_{clusters}.nc")
            if config_provider("sector", "solar_thermal")(w)
            else []
        ),
        egs_potentials=lambda w: (
            resources("egs_potentials_{clusters}.csv")
            if config_provider("sector", "enhanced_geothermal", "enable")(w)
            else []
        ),
        egs_overlap=lambda w: (
            resources("egs_overlap_{clusters}.csv")
            if config_provider("sector", "enhanced_geothermal", "enable")(w)
            else []
        ),
        egs_capacity_factors=lambda w: (
            resources("egs_capacity_factors_{clusters}.csv")
            if config_provider("sector", "enhanced_geothermal", "enable")(w)
            else []
        ),
        direct_heat_source_utilisation_profiles=resources(
            "direct_heat_source_utilisation_profiles_base_s_{clusters}_{planning_horizons}.nc"
        ),
    output:
        resources(
            "networks/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.nc"
        ),
    threads: 1
    resources:
        mem_mb=2000,
    log:
        logs(
            "prepare_sector_network_base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.log"
        ),
    benchmark:
        benchmarks(
            "prepare_sector_network/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}"
        )
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/prepare_sector_network.py"

##### prepare_sector_network #######

def add_perennials(n, costs):
    logger.info("Adding perennials.")

    perennial_CO2_seq = (
            snakemake.config["perennials"]["yield"]
            / snakemake.config["perennials"]["potential_co2"]
    )  # tDM perennials / tCO2e sequestred

    nodes = pop_layout.index
    n.add("Carrier", "perennial")
    n.add("Carrier", "perennial store")

    n.add(
        "Bus",
        nodes + " perennials co2 store",
        location=nodes,
        carrier="perennial store",
        unit="t_co2",
    )

    df_gbr = pd.DataFrame(index=n.snapshots, columns=["harvest"])
    df_gbr["harvest"] = df_gbr.index.month.isin([4, 5, 6, 7, 8, 9, 10]).astype(int)
    p_max_pu = pd.DataFrame(index=n.snapshots, columns=nodes)

    for node in nodes:
        p_max_pu[node] = df_gbr["harvest"]

    n.add(
        "Link",
        nodes,
        suffix=" perennials GBR",
        bus0="co2 atmosphere",
        bus1=nodes + " perennials co2 store",
        bus2=nodes.values,
        bus3=spatial.gas.biogas,
        efficiency=1,
        efficiency2=-costs.at['perennials gbr', "electricity-input"] * perennial_CO2_seq,
        efficiency3=costs.at['perennials gbr', "biogas-output"] * perennial_CO2_seq,
        carrier="perennial",
        p_nom_extendable=True,
        p_max_pu=p_max_pu,
        capital_cost=costs.at['perennials gbr', "fixed"] * perennial_CO2_seq,
        marginal_cost=costs.at['perennials gbr', "VOM"] * perennial_CO2_seq,
        lifetime=costs.at['perennials gbr', "lifetime"],
    )

    biomass_potentials = pd.read_csv(snakemake.input.biomass_potentials, index_col=0)

    yields_perennials_1G_biofuels = pd.read_csv(snakemake.input.yields_perennials_1G_biofuels).set_index("node")

    perennials_area_spatial =  (
        (
            biomass_potentials.filter(regex='biofuels_1G')
            / yields_perennials_1G_biofuels.filter(regex='biofuels_1G')
        ).sum(axis=1)  # -> (MWh/y) / (MWh / ha / y) = (ha) returns the area used by sum of the 3 biofuels_1G classes, that is potentially assigned to perennials

    perennials_potentials_spatial = perennials_area_spatial * yields_perennials_1G_biofuels.filter(regex='perennials') # (ha) * (tCO2_seq_perennial/ha)

    #perennials_potentials_spatial = (
    #        (
    #                biomass_potentials.filter(regex='biofuels_1G')
    #                / snakemake.config["perennials"]["yield_biofuels_1G"]
    #        ).sum(axis=1)
    #        * snakemake.config["perennials"]["potential_co2"]
    #)  # potential tCO2e seq

    n.add(
        "Store",
        nodes,
        suffix=" CO2s_perennials",
        bus=nodes + " perennials co2 store",
        e_nom_extendable=True,
        e_nom_max=perennials_potentials_spatial,
        carrier="perennial store",
        e_cyclic=False,
    )

# TODO: review and test in pypsa-eur
