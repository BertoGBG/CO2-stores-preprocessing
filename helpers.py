import pandas as pd
import numpy as np
import pypsa
import parameters as p

# -------TECHNO-ECONOMIC DATA & ANNUITY
def annuity(n, r):
    """Calculate the annuity factor for an asset with lifetime n years and
    discount rate of r, e.g. annuity(20,0.05)*20 = 1.6"""

    if r > 0:
        return r / (1. - 1. / (1. + r) ** n)
    else:
        return 1 / n


def prepare_costs(cost_file, USD_to_EUR, discount_rate, Nyears, lifetime):
    """ This function uses, data retrived form the technology catalogue and other sources and compiles a DF used in the model
    input: cost_file # csv
    output: costs # DF with all cost used in the model"""

    # Nyear = nyear in the interval for myoptic optimization--> set to 1 for annual optimization
    # set all asset costs and other parameters
    costs = pd.read_csv(cost_file, index_col=[0, 1]).sort_index()

    # correct units to MW and EUR
    costs.loc[costs.unit.str.contains("/kW"), "value"] *= 1e3
    costs.loc[costs.unit.str.contains("USD"), "value"] *= USD_to_EUR

    # min_count=1 is important to generate NaNs which are then filled by fillna
    costs = costs.loc[:, "value"].unstack(level=1).groupby("technology").sum(min_count=1)
    costs = costs.fillna({"CO2 intensity": 0,
                          "FOM": 0,
                          "VOM": 0,
                          "discount rate": discount_rate,
                          "efficiency": 1,
                          "fuel": 0,
                          "investment": 0,
                          "lifetime": lifetime
                          })
    annuity_factor = lambda v: annuity(v["lifetime"], v["discount rate"]) + v["FOM"] / 100
    costs["fixed"] = [annuity_factor(v) * v["investment"] * Nyears for i, v in costs.iterrows()]
    return costs


def cost_add_technology(discount_rate, tech_costs, technology, investment, lifetime, FOM, VOM):
    '''function to calculate annualized fixed cost for any technology from custom inputs
    and adds it to the tech_costs dataframe '''
    annuity_factor = annuity(lifetime, discount_rate) + FOM / 100
    tech_costs.at[technology, "fixed"] = annuity_factor * investment
    tech_costs.at[technology, "lifetime"] = lifetime
    tech_costs.at[technology, "FOM"] = FOM
    tech_costs.at[technology, "investment"] = investment
    tech_costs.at[technology, "VOM"] = VOM
    return tech_costs


def potential_perennials_NUTS0(file_path):
    '''calculates the potential in tCO2e/y for perennials per each NUTS0 '''
    #file_path = p.file_path_NUTS0
    df_jrc_imp = pd.read_excel(file_path)
    df_jrc = df_jrc_imp[df_jrc_imp['Energy Commodity'].isin(p.jrc_yields_assumptions.keys())]
    df_jrc = df_jrc.assign(Yield_assumption=df_jrc['Energy Commodity'].map(p.jrc_yields_assumptions))
    df_jrc['Yield unit'] = 'PJ/Mha'
    df_jrc['Area (Mha)'] = df_jrc['Value'] / df_jrc['Yield_assumption']
    df_jrc['CO2_seq (MtCO2/y)'] = df_jrc['Area (Mha)']* p.CO2e_seq_ha

    # return potential per country in areas availble for perennials (Mha)
    df_NUTS0_potential = df_jrc.groupby('NUTS0')['CO2_seq (MtCO2/y)'].sum()
    df_NUTS0_potential_area = df_jrc.groupby('NUTS0')['Area (Mha)'].sum()

    return df_NUTS0_potential, df_NUTS0_potential_area
