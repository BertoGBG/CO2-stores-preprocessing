""" this file pre process the input data for perennials.
the pre processing is the same for each node, except that for the potential which is calcualted separately
The schematics of the multilink structure is attached in the document
"""

"""References
R1 : https://doi.org/10.1016/B978-0-323-95879-0.50147-8
R2: https://doi.org/10.1016/j.scitotenv.2023.167943
R3: https://dcapub.au.dk/djfpublikation/djfpdf/DCArapport193.pdf"""

""" LINKS
L1: multilink modelling perennial carbon sequestration and green biorefining
L2: link modelling biogas production from energy fibers
L3: link modelling dewatering, drying and pelletization of energy fibers to agro residues"""

#######

from helpers import *

""" L1 - perennials & green bio refining
Reference = R2 (tentative) 

bus 0 : CO2atm, unit :  tCO2
bus 1 : CO2seq, unit : tCO2 
bus 2 : el, unit : MWh_el
bus 3 : biogas, unit :  MWh
"""
"""Potential per node"""
CO2s_potential_perennials = p.arable_land_node * p.CO2e_seq_ha

""" add data to tech_cost DF (from technology data) """
peren_CO2atm = (p.Y_perennials/p.DM_perennials) / p.CO2e_seq_ha # t fresh biomass / CO2atm

"efficiencies" \
"NOTE: ONLY BIOGAS! NOT SEPARATED FROM CO2"
eff_co2= 1 # bus1/bus0, unit: tCO2/tCO2
eff_el = (p.el_cons / p.peren_input) * peren_CO2atm   # bus2/bus0, unit: MWh_el/tCO2atm
eff_biogas = (p.biogas_prod / p.peren_input) * peren_CO2atm   # bus3/bus0, unit: MWh_biogas/tCO2atm

"cost added per ton of "
tech_costs = prepare_costs(p.cost_file, p.USD_to_EUR, p.discount_rate, 1, p.lifetime)
GBR_inv_co2 = p.tot_capex / p.USD_to_EUR / p.peren_input_cost * peren_CO2atm *1e6 # M€/(tCO2/h)
GBR_VOM_co2 = p.tot_opex / p.USD_to_EUR / p.peren_input_cost * peren_CO2atm *1e6 # €/y

cost_add_technology(p.discount_rate, tech_costs, 'Perennials_GBR', GBR_inv_co2 * 1e6,
                    p.lifetime, 0, GBR_VOM_co2)

# TODO Adress :(GBR produces only April-October, Cseq during whole year but energy prod is not)

""" Code to add link in PyPSA,
NOTE: it requires an existing pypsa network"""
n.add('Bus','CO2s_perennials',carrier = 'CO2', unit = 't')

n.add('Link',
      'perennials_GBR',
      bus0="CO2_atm",
      bus1="CO2s_perennials",
      bus2="el",
      bus3 ="biogas",
      efficiency=1,
      efficiency2 = -eff_el,
      efficiency3 = eff_biogas,
      p_nom_extendable=True,
      capital_cost=tech_costs.at['Perennials_GBR', "fixed"],
      marginal_cost=tech_costs.at['Perennials_GBR', "VOM"])

n.add('Store',
      'CO2s_perennials',
      bus= 'CO2s_perennials',
      e_nom_extendable= False,
      e_nom_max = CO2s_potential_perennials,
      e_cyclic = False)











