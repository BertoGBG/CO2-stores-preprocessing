
# import necessary Python modules
import geopandas
import pandas
import logging
import yaml



def build_afforestation_potentials(config_yaml, network_geojson, nuts2_geojson, afforestation_corine_potentials_csv_file, afforestation_nuts_file, output_csv_file, log):

    # load config yaml file representing the PyPSA-Eur configuration
    handle = open(config_yaml)
    config = yaml.safe_load(handle)
    handle.close()


    # configure log mechanism
    if log is True:
        logging.basicConfig(level = config["logging"]["level"])
        logger = logging.getLogger(__name__)
        if config["afforestation"]["potential_type"] == "density":
            logger.info("Calculate afforestation potentials based on biomass densities")
        else:   # growth
            logger.info("Calculate afforestation potentials based on growth rates")


    # load files
    network = geopandas.read_file(network_geojson)
    nuts2 = geopandas.read_file(nuts2_geojson)
    corine_potentials = pandas.read_csv(afforestation_corine_potentials_csv_file).set_index("node")

    if config["afforestation"]["potential_type"] == "density":
        nuts_biomass_density = pandas.read_excel(afforestation_nuts_file, sheet_name = "BIOMASS 2020", skiprows = 2).iloc[:, [2, 7]].set_index("NUTS")
        nuts_biomass_density = nuts_biomass_density[nuts_biomass_density.index.str.len() == 2]
    else:   # growth
        nuts2_growth_rates = pandas.read_csv(afforestation_nuts_file).set_index("NUTS2")

        # harmonize NUTS2 indexes against the network indexes
        nuts2["NUTS_ID"] = nuts2["NUTS_ID"].apply(lambda x: "GR%s" % x[2:] if x[:2] == "EL" else x)
        nuts2["NUTS_ID"] = nuts2["NUTS_ID"].apply(lambda x: "GB%s" % x[2:] if x[:2] == "UK" else x)
        nuts2_growth_rates = nuts2_growth_rates.rename(index = lambda x: "GR%s" % x[2:] if x[:2] == "EL" else x)
        nuts2_growth_rates = nuts2_growth_rates.rename(index = lambda x: "GB%s" % x[2:] if x[:2] == "UK" else x)


    # create data frame to store afforestation potential for each node
    data_frame = pandas.DataFrame(columns = ["node", 'growth rate [tCO2/ha/y]', "potential [tCO2/y]" ])

    # iterate through PyPSA-Eur network regions (nodes)
    if config["afforestation"]["potential_type"] == "density":

        for i in range(len(network)):

            # get node information
            node_row = network.iloc[i]
            node_name = node_row["name"]

            # calculate afforestation potential
            country = node_name[:2]
            if country in nuts_biomass_density.index:
                biomass_density = nuts_biomass_density.loc[country]["(Tons/ha)"]
            else:
                if log is True:
                    logger.warning("Set biomass density to 117 t/ha for node '%s' (given that country '%s' does not have information)" % (node_name, country))
                biomass_density = 117   # average value across Europe is taken as default biomass density (in t/ha) in case country does not exist in dataframe (e.g. Kosovo - XK)
            growth_rate = biomass_density / config["afforestation"]["number_years"] * config["afforestation"]["co2_per_tonne"]  # tCO2/y/ha
            node_afforestation_potential = (corine_potentials.loc[node_name]["potential [sqkm]"] * 100) * growth_rate * config["afforestation"]["max_land_usage"] # tCO2/y

            # add node afforestation potential into data frame
            if log is True:
                logger.info("Node '%s' has an afforestation potential of %d [t/y]" % (node_name, node_afforestation_potential))
            data_frame.loc[len(data_frame)] = [node_name, node_afforestation_potential]

    else:   # growth

        for i in range(len(network)):

            # get node information
            node_row = network.iloc[i]
            node_name = node_row["name"]
            node_geometry = geopandas.GeoSeries(node_row["geometry"], crs = 3035)

            # iterate through NUTS2 codes
            total_fraction = 0
            node_afforestation_potential = 0
            for j in range(len(nuts2)):

                # get NUTS2 information
                nuts2_row = nuts2.iloc[j]
                nuts2_name = nuts2_row["NUTS_ID"]

                # check that NUTS2 belongs to node country
                if nuts2_name[:2] != node_name[:2]:
                    continue

                # get NUTS2 geometry
                nuts2_geometry = geopandas.GeoSeries(nuts2_row["geometry"], crs = 3035)

                # calculate fraction of node geometry overlapped by NUTS2 geometry
                intersection = nuts2_geometry.intersection(node_geometry.iloc[0])
                fraction = float(intersection.area.iloc[0]) / float(node_geometry.area.iloc[0])
                total_fraction += fraction

                # calculate afforestation potential based on fraction and aggregate it to node afforestation potential
                growth_rate = nuts2_growth_rates.loc[nuts2_name]["affo rate (t/ha/y)"] * config["afforestation"]["co2_per_tonne"]  # tCO2/y/ha
                node_afforestation_potential += (corine_potentials.loc[node_name]["potential [sqkm]"] * 100) * fraction * config["afforestation"]["max_land_usage"]  * growth_rate # tCO2/y

            # check that total fraction is very close/near to 1
            if total_fraction > 0 and (total_fraction < 0.99 or total_fraction > 1.01):
                logger.error("Node '%s' has an unexpected/incorrect total fraction of %.4f (expected value should be very close/near to 1)" % (node_name, total_fraction))

            # add node afforestation potential into data frame
            if log is True:
                logger.info("Node '%s' has an afforestation potential of %d [t/y]" % (node_name, node_afforestation_potential))
            data_frame.loc[len(data_frame)] = [node_name, node_afforestation_potential]


    # save afforestation potentials into CSV file
    if log is True:
        logger.info("Save afforestation potentials into CSV file '%s'" % output_csv_file)
    data_frame.set_index("node", inplace = True)
    data_frame.to_csv(output_csv_file)



if __name__ == "__main__":

    # build and save afforestation potentials into CSV file
    if "snakemake" in globals():
        build_afforestation_potentials("config/config.yaml", snakemake.params["network_geojson"], snakemake.params["nuts2_geojson"], snakemake.input["afforestation_corine_potentials_csv_file"], snakemake.input["afforestation_nuts_file"], snakemake.output["csv_file"], True)
    else:
        build_afforestation_potentials("config.yaml", "regions_onshore_base_s_39.geojson", "NUTS_RG_03M_2013_4326_LEVL_2.geojson", "afforestation_corine_potentials_s_39.csv", "afforestation_nuts_biomass_densities.xlsx", "afforestation_potentials_s_39.csv", True)


