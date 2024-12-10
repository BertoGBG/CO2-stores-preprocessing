'''ECONOMICS AND COST ASSUMPTIONS'''
'''Technology Data Economic Parameters'''
year_EU = 2030  # invesment year -> MUST be set > 2030
cost_file = "costs_2030.csv"
USD_to_EUR = 1
DKK_Euro = 7.46  # ratio avg 2019
discount_rate = 0.07  #
Nyears = 1  # for myoptic optimization (parameter not used but needed in some functions)
lifetime = 25  # of the plant

""" general paramaters"""
LHV_ch4 = 50/3600 # MWh/kg

'''PERENNIALS AND GREEN BIOREFINING'''

" Potential for perennials and green biorefining"
arable_land_node= 1e6 # ha --> EXAMPLE

"""Data about perennials, Ref: R3"""
CO2e_seq_ha = 2 # tCO2e/ha/y sequestred (R3 table 3.5)
Y_perennials  = 15 # t DM biomass/ha/y (R3 table 2.5)
DM_perennials = 0.18 # ref: R2

"""GBR input data from Ref: R2 scenario 2 - TENTATIVE!
NOTE Brown juice and press cake all to biogas"""
peren_input = 32.7 # tons, @ 18% DM
brown_juice = 18.74 # tons (very low DM)
press_cake = 12.125 # Tons (DM>50%)
biogas_prod = 2.256 # tons (40 % CH4)
biogas_prod = 2.256 * 0.4 * LHV_ch4 * 1e3 # MWh
el_cons = 0.389 + 0.726 # MWh el

"""GBR Cost estimation - Investment + OPEX. TENTATIVE 
ref: R1 """
peren_input_cost = 40 # t/h
tot_capex = 9.33 # M USD # based on Danish prices
tot_opex =  6.05 -0.11- 5.25 # M USD / y  # including cost of fresh biomass and wuick transport) and sales of proteins, but excluding Electricity
flh_y = 4200 # green crops harvest is only May-October --> need a time series for biogas and residues produciton
year_cost = 2020

