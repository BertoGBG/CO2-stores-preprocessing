import pypsa
import pandas as pd
import numpy as np

n = pypsa.Network('Results_analysis/pypsa-eur/base_s_90__Co2L0-3H-T-H-B-I-A-solar+p3-dist1_2050.nc')

##########################
# CHECK PERENNIALS
##########################
per_lk = n.links.filter(like = 'perennials', axis=0).index
per_st = n.stores.filter(like = 'perennials', axis=0).index

# weights based on potential CO2 removal
p_per = n.stores.e_nom_max[per_st]
w_per = p_per / p_per.sum()

# buses
bus0_per = n.links.bus0[per_lk] # co2 from atm
bus1_per = n.links.bus1[per_lk] # co2 to store - potential
bus2_per = n.links.bus2[per_lk] # el
bus3_per = n.links.bus3[per_lk] # biogas

# efficiencies  unit/tDM (dry matter)
eff_per = n.links.efficiency[per_lk]
eff2_per = n.links.efficiency2[per_lk]
eff3_per = n.links.efficiency3[per_lk]

# weighted efficiencies:
eff_avg_per = sum(eff_per * w_per.values)
eff2_avg_per = sum(eff2_per * w_per.values)
eff3_avg_per = sum(eff3_per * w_per.values)
print('PERENNIALISATION: Electricity input (MWhel/tCO2)', eff2_avg_per)
print('PERENNIALISATION: Biogas Output (MWh_bg/tCO2)', eff3_avg_per)

# COSTS - based on inputs
# VOM potential weighted
vom_per = sum(n.links.marginal_cost[per_lk] * w_per.values)
print('PERENNIALISATION: avg VOM in €/tCO2 removed', vom_per)

# capital cost in €/(tCO2/h) - not annualized
inv_per_td = 1371168.1394 # € /tDM/h - from technology-data
VOM_per_td = 43.2317 # € /tDM - from technology data
perennial_CO2_seq  = n.links.marginal_cost[per_lk] / VOM_per_td #tDM/tCO2
inv_per = sum(inv_per_td * perennial_CO2_seq *  w_per.values)
inv_per_a = sum(inv_per_td * perennial_CO2_seq /4200 *  w_per.values)
print('PERENNIALISATION: avg capital cost M€/(tCO2/h)', inv_per*1e-6)

# COSTS as Results -  Cost of removing 1 tCO2 including energy consumption/sales at hourly shadow prices.
# annual sum of costs & revenues divided by the removed CO2
el_per_cost_node = pd.DataFrame(
    n.buses_t.marginal_price[bus2_per].values * n.links_t.p2[per_lk].values,
    columns=n.links_t.p2[per_lk].columns
).sum(axis=0) # € / a as electricity costs

bg_per_cost_node = pd.DataFrame(
    n.buses_t.marginal_price[bus3_per].values * n.links_t.p3[per_lk].values,
    columns=n.links_t.p3[per_lk].columns
).sum(axis=0)  # € / a as bigas revenues

# European average CO2 removal cost from annual balances
vom_per_node = n.links.marginal_cost[per_lk] * n.links_t.p0[per_lk].sum() # € / a as vom (including sales of protein)
cc_per_node = n.links.capital_cost[per_lk] * n.links.p_nom_opt[per_lk] # € / a annualized investmet cost
tot_per_cost = cc_per_node+ vom_per_node+ bg_per_cost_node+ el_per_cost_node # tot cost €/a per node
co2_per_node = pd.Series(n.stores_t.e[per_st].iloc[-1,:].values, index=per_lk) # CO2 removed per node tCO2/a
CO2_per_removal_cost = tot_per_cost / co2_per_node.values
w_per_results = co2_per_node / sum(co2_per_node)
avg_per_co2_rem_cost = np.nansum(CO2_per_removal_cost * w_per_results.values)
print('R: PERENNIALISATION total CO2 removed (MtCO2/a): ', co2_per_node.sum()*1e-6)
print('R: PERENNIALISATION AVG CO2 removal cost (€/tCO2)', avg_per_co2_rem_cost)

df_per_node = pd.DataFrame({
    'R el cost €/a': el_per_cost_node,
    'R biogas cost €/a': bg_per_cost_node,
    'R VOM €/a': vom_per_node,
    'R INV €/a': cc_per_node,
    'R CO2 removed tCO2/a': co2_per_node,
    'R CO2 removal cost €/tCO2' : CO2_per_removal_cost,
})

dict_per_avg = {
    'T1 el_input (MWh/tCO2)' : eff2_avg_per,
    'T1 bg_input (MWh/tCO2)' : eff3_avg_per,
    'T1 capital cost (M€/(tCO2/h)' : inv_per*1e-6,
    'T1 capital cost (k€/(tCO2/a)': inv_per_a * 1e-3,
    'T1 VOM (€/(tCO2)': vom_per,
    'R AVG CO2 removal cost €/tCO2' : avg_per_co2_rem_cost,
}


##########################
# CHECK AFFORESTATION
##########################
af_lk = n.links.filter(like = 'afforestation', axis=0).index
af_st = n.stores.filter(like = 'afforestation', axis=0).index

# weights based on potential CO2 removal
p_af = n.stores.e_nom_max[af_st]
w_af = p_af / p_af.sum()

# buses
bus0_af = n.links.bus0[af_lk] # co2 from atm
bus1_af = n.links.bus1[af_lk] # co2 to store - potential

# efficiencies  unit/tCO2
eff_af = n.links.efficiency[af_lk]

# weighted efficiencies:
eff_avg_af = sum(eff_af * w_af.values)
print('AFFORESTATION: CO2 capture rate (tCO2_atm/tCO2_biom)', eff_avg_af)

# COSTS - based on inputs
# capital cost weighted by potential
avg_af_co2_rem_cost = sum(n.stores.capital_cost[af_st] * w_af) # €/tCO2
print('AFFORESTATION: avg CO2 removal cost (€/tCO2)', avg_af_co2_rem_cost)

# COSTS as Results
# European average CO2 removal cost from annual balances
cc_af_node = n.stores.capital_cost[af_st] * n.stores.e_nom_opt[af_st] # total €/a annualized investment cost
co2_af_node = n.stores_t.e[af_st].iloc[-1,:] # CO2 removed per node tCO2/a
CO2_af_removal_cost = cc_af_node / co2_af_node
w_af_results = co2_af_node / co2_af_node.sum()
avg_af_co2_rem_cost_res = np.nansum(CO2_af_removal_cost * w_af_results)
print('R: AFFORESTATION total CO2 removed (MtCO2/a): ', co2_af_node.sum()*1e-6)
print('R: AFFORESTATION AVG CO2 removal cost (€/tCO2)', avg_af_co2_rem_cost_res)

df_af = pd.DataFrame({
    'R INV €/a': cc_af_node,
    'R annualized cc €/tCO2/a': CO2_af_removal_cost,
    'R CO2 removed tCO2/a': co2_af_node,
    'R CO2 removal cost €/tCO2' : CO2_af_removal_cost,
})

dict_af_avg = {
    'T1: AVG CO2 removal cost €/tCO2' : avg_per_co2_rem_cost,
}

##########################
# CHECK BIOCHAR
##########################
bc_lk = n.links.filter(like='biochar', axis=0)
bc_lk = bc_lk[~bc_lk.index.str.contains('co2')].index # filter off links for biochar heat
bc_st = n.stores.filter(like = 'co2 biochar', axis=0).index

# weights based on potential
p_bc = n.stores.e_nom_max[bc_st] # tCO2/a max
w_bc = p_bc / p_bc.sum()

# buses
bus0_bc = n.links.bus0[bc_lk] # CO2 atm
bus1_bc = n.links.bus1[bc_lk] # biochar
bus2_bc = n.links.bus2[bc_lk] # biomass
bus3_bc = n.links.bus3[bc_lk] # el
bus4_bc = n.links.bus4[bc_lk] # heat

# efficiencies  unit/MW_el
eff_bc = n.links.efficiency[bc_lk] # biochar(tCO2/h)/(tCO2/h)
eff2_bc = n.links.efficiency2[bc_lk] # biomass MW/ (tCO2/h)
eff3_bc = n.links.efficiency3[bc_lk] # el MW/(tCO2/h)
eff4_bc = n.links.efficiency4[bc_lk] # heat MW/(tCO2/h)

# weighted efficiencies:
eff_avg_bc = sum(eff_bc * w_bc.values)
eff2_avg_bc = sum(eff2_bc * w_bc.values)
eff3_avg_bc = sum(eff3_bc * w_bc.values)
eff4_avg_bc = sum(eff4_bc * w_bc.values)
print('BIOCHAR: Electricity input (MWhel/tCO2)', eff3_avg_bc)
print('BIOCHAR: Biomass input (MWh/tCO2)', eff2_avg_bc)
print('BIOCHAR: Heat output (MWh/tCO2)', eff4_avg_bc)

# COSTS - based on inputs
# VOM potential weighted
vom_bc = sum(n.links.marginal_cost[bc_lk] * w_bc.values)
print('BIOCHAR: avg VOM in €/tCO2 removed', vom_bc)

# cost for technolgy data:
inv_bc_td = 8939565.2069  # EUR/t_CO2/h
vom_bc_td = 47.6777 # EUR/t_CO2
fom_bc_td = 3.4167 # % inv / y

# capital cost in €/(tCO2/h) - not annualized
inv_bc = sum(n.links.capital_cost[bc_lk] * w_bc.values)
print('BIOCHAR: avg capital cost in M€/(tCO2/h) removed', inv_bc_td*1e-6)
print('BIOCHAR: avg FOM in % inv year removed', fom_bc_td)

# COSTS as Results
el_bc_cost_node = pd.DataFrame(
    n.buses_t.marginal_price[bus3_bc].values * n.links_t.p3[bc_lk].values,
    columns=n.links_t.p3[bc_lk].columns
).sum(axis=0) # € / a as electricity costs

bm_bc_bus = n.links.bus2[bc_lk]
bm_bc_cost_node = pd.DataFrame(
    n.buses_t.marginal_price[bm_bc_bus].values * n.links_t.p2[bc_lk].values,
    columns=n.links_t.p2[bc_lk].columns
).sum(axis=0) # € / a as biomass costs

if 'bus4' in n.links.columns and n.links.bus4[bc_lk].str.strip().ne('').all():
    ht_bc_bus = n.links.bus4[bc_lk]
    ht_bc_cost_node = pd.DataFrame(
        n.buses_t.marginal_price[ht_bc_bus].values * n.links_t.p4[bc_lk].values,
        columns=n.links_t.p4[bc_lk].columns
    ).sum(axis=0)
else:
    ht_bc_cost_node = pd.Series(0, index=bc_lk)

vom_bc_node = n.links.marginal_cost[bc_lk] * n.links_t.p0[bc_lk].sum() # € / a as vom (including sles of protein)
cc_bc_node = n.links.capital_cost[bc_lk] * n.links.p_nom_opt[bc_lk] # € / a annualized investment cost
tot_bc_cost = cc_bc_node + vom_bc_node + el_bc_cost_node + bm_bc_cost_node + ht_bc_cost_node# tot cost €/a per node

# CO2 removed
co2_bc_node = n.stores_t.e[bc_st].iloc[-1,:] # CO2 removed per node tCO2/a
co2_bc_node = pd.Series(co2_bc_node.values, index=bc_lk)
CO2_bc_removal_cost = tot_bc_cost / co2_bc_node.values
weights_bc_co2_removed = co2_bc_node / sum(co2_bc_node)
avg_bc_co2_rem_cost = np.nansum(CO2_bc_removal_cost * weights_bc_co2_removed.values)
print('R: BIOCHAR total CO2 removed (MtCO2/a): ', co2_bc_node.sum()*1e-6)
print('R: BIOCHAR AVG CO2 removal cost (€/tCO2)', avg_bc_co2_rem_cost)

df_bc_node = pd.DataFrame({
    'R el cost €/a': el_bc_cost_node,
    'R biomass cost €/a': bm_bc_cost_node,
    'R VOM €/a': vom_bc_node,
    'R INV €/a': cc_bc_node,
    'R CO2 removed tCO2/a': co2_bc_node,
    'R CO2 removal cost €/tCO2' : CO2_bc_removal_cost,
})


dict_bc_avg = {
    'T1 el_input (MWh/(tCO2/h))' : eff3_avg_bc,
    'T1 biomass_input (MWh/(tCO2/h))' : eff2_avg_bc,
    'T1 heat_output (MWh/(tCO2/h))': eff4_avg_bc,
    'T1 avg capital cost (M€/(tCO2/h))' : inv_bc_td*1e-6,
    'T1 VOM (€/tCO2)': vom_bc_td,
    'T1 FOM (% nv/a)': fom_bc_td,
    'R AVG CO2 removal cost €/tCO2' : avg_bc_co2_rem_cost,
}


##########################
# CHECK ERW
##########################
ew_lk = n.links.filter(like = 'EW', axis=0).index
ew_st = n.stores.filter(like = 'EW', axis=0).index

# weights based on potential
p_ew = n.stores.e_nom[ew_st] # NOTE not e_nom_max as in other CDRs
w_ew = p_ew / p_ew.sum()

# buses
bus0_ew = n.links.bus0[ew_lk] # el
bus1_ew = n.links.bus1[ew_lk] # co2 atm
bus2_ew = n.links.bus2[ew_lk] # co2 store

# efficiencies  unit/MW_el
eff_ew = n.links.efficiency[ew_lk]
eff2_ew = n.links.efficiency2[ew_lk]

# weighted efficiencies:
eff_avg_ew = sum(eff_ew * w_ew.values)
eff2_avg_ew = sum(eff2_ew * w_ew.values)
print('ERW: Electricity input (MWhel/tCO2)', 1/eff2_avg_ew)

# COSTS - based on inputs
inv_ew_td = 116625.6521 # (EUR/a)/tCO2 - from technology data
VOM_ew_td = 175.9993 # € /(tCO2/h) - from technology data

# VOM potential weighted
vom_ew = sum(n.links.marginal_cost[ew_lk] / eff2_ew * w_ew.values)
print('ERW: avg VOM in (€/tCO2) removed', vom_ew)

# capital cost in €/(tCO2) - already annualized
# TODO CHECK formulation because capital cost is not flow based (link)
# PV = PMT × [1 - (1 + r)^(-n)] / r
r = 0.07 # %/y
l = 30 # years
capital_cost_ew_recalc = inv_ew_td * (1-(1+r)**(-l)) / r
print('ERW: avg capital cost M€/tCO2', capital_cost_ew_recalc * 1e-6)

"""
FROM PYPSA-EUR
n.add(
    "Link",
    nodes,
    suffix=" EW",
    bus0=nodes.values,
    bus1="co2 atmosphere",
    bus2=nodes + " EW co2 store",
    carrier="EW",
    capital_cost=costs.at["Enhanced Weathering", "investment"] / costs.at["Enhanced Weathering", "electricity-input"],
    marginal_cost=costs.at["Enhanced Weathering", "VOM"] / costs.at["Enhanced Weathering", "electricity-input"],
    efficiency=-1 / costs.at["Enhanced Weathering", "electricity-input"],
    efficiency2=1 / costs.at["Enhanced Weathering", "electricity-input"],
    p_nom_extendable=True,
    lifetime=costs.at["Enhanced Weathering", "lifetime"],
)

FROM TECHNOLOGY - DATA
Enhanced Weathering,VOM,175.9993, EUR/h/tCO2," Jessica Strefler et al 2018 Environ. Res. Lett. 13 034010 https://doi.org/10.1088/1748-9326/aaa9c4, conversion 0.7541 EUR/US (2014)",,2014.0
Enhanced Weathering,electricity-input,0.1852,MWh/tCO2 , Jessica Strefler et al 2018 Environ. Res. Lett. 13 034010 https://doi.org/10.1088/1748-9326/aaa9c4,,
Enhanced Weathering,investment,116625.6521,EUR/tCO2," Jessica Strefler et al 2018 Environ. Res. Lett. 13 034010 https://doi.org/10.1088/1748-9326/aaa9c4, conversion 0.7541 EUR/US (2014)",,2014.0
Enhanced Weathering,lifetime,1.0,years,costs are given per year,,

NOTE POTENTIAL ISSUES about UNITS: 
1) capital cost  should be in €/MW_el (or MWh/h) on a energy flow basis (of CO2 flow for other techs)
capital_cost = (EUR/tCO2) / (MWh_el/(tCO2)) = EUR/MWh_el --> it should be EUR/MW ???? (or EUR/(MWh/h))

2) VOM should be per unit of energy (or tCO2) but they seem to be per flow. and have implemented per unit of flow. 
VOM = EUR/h/tCO2 --> what does it even mean ? (I assumed it to be (EUR * h / tCO2 ) or EUR

"""

# COSTS as Results
el_ew_cost_node = pd.DataFrame(
    n.buses_t.marginal_price[bus0_ew].values * n.links_t.p0[ew_lk].values,
    columns=n.links_t.p0[ew_lk].columns
).sum(axis=0) # € / a as electricity costs

vom_ew_node = n.links.marginal_cost[ew_lk] * n.links_t.p0[ew_lk].sum() # € / a as vom
cc_ew_node = n.links.capital_cost[ew_lk] * n.links.p_nom_opt[ew_lk] # € / a annualized investment cost
tot_ew_cost = cc_ew_node + vom_ew_node + el_ew_cost_node # tot cost €/a per node

# CO2 removed
co2_ew_node = n.stores_t.e[ew_st].iloc[-1,:] # CO2 removed per node tCO2/a
co2_ew_node = pd.Series(co2_ew_node.values, index=ew_lk)
CO2_ew_removal_cost = tot_ew_cost / co2_ew_node.values
weights_ew_co2_removed = co2_ew_node / sum(co2_ew_node)
avg_ew_co2_rem_cost = np.nansum(CO2_ew_removal_cost * weights_ew_co2_removed.values)
print('R: ERW total CO2 removed (MtCO2/a): ', co2_ew_node.sum()*1e-6)
print('R: ERW AVG CO2 removal cost (€/tCO2)', avg_ew_co2_rem_cost)

df_ew_node = pd.DataFrame({
    'R el cost €/a': el_ew_cost_node,
    'R VOM €/a': vom_ew_node,
    'R INV €/a': cc_ew_node,
    'R CO2 removed tCO2/a': co2_ew_node,
    'R CO2 removal cost €/tCO2' : CO2_ew_removal_cost,
})


dict_ew_avg = {
    'T1 el_input (MWh/(tCO2/h))' : 1/eff2_avg_ew,
    'T1 avg capital cost (M€/(tCO2/h))' : inv_ew_td*1e-6,
    'T1 VOM (€/tCO2)': vom_ew,
    'R AVG CO2 removal cost €/tCO2' : avg_ew_co2_rem_cost,
}
