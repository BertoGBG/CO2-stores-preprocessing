Update	December 2023
Authors:	Viorel Blujdea, Roberto Pilli, Paul Rougieux

This folder includes detailed information on the parameters defining each allometric equation associated to each forest type and group of classifiers as defined within the companion databases Volume_increment_database and Volume_biomass_bcef_databse.
"vol_to_biomass_best_match" reports the RMSE and the proportion of stemwood to total boveground biomass associated to the allometric equation associated to each forest type and group of classifiers.
All data were classified according to the following items:

- countries: all EU-27 Memebr States, except Cyprus and Malta, where detailed information are missing.  See Country_codes.csv in Volume_increment_database.
- status: defined, when possible, as Forest Area Available for Wood Supply (FAWS) and Forest Area Not Available for Wood Supply (FNAWS). See Status_codes.csv in Volume_increment_database.
- forest types: defined according to the leading species identified at country level. See Forest_codes.csv in Volume_increment_database.
- administrative regions (when possible), generally identified at NUTS 2 level or according to other classification system derived from NFI data. See Regions_codes.csv in Volume_increment_database.
- management types: mostly including high forests (H) and coppices (C) or otherwise defined within the countries' sheets. See Management_codes.csv in Volume_increment_database.
- management strategies: mostly including even-aged (E) and uneven-aged (U) forests.  See Management_codes.csv in Volume_increment_database.
- species' group: distinguished between coniferous (con) and broadleaves (broad) species. See Forest_codes.csv in Volume_increment_database..

Detailed information on the classification system applied at country level are reported on the companion collection Volume_increment_databse

Each allometric equation includes the following parameters, reported on "vol_to_biomass_params" (see Boudewyn et al., 2007 for further details):

a
b
a_nonmerch
b_nonmerch
k_nonmerch
cap_nonmerch
a_sap
b_sap
k_sap
cap_sap
a1
a2
a3
b1
b2
b3
c1
c2
c3
min_volume
max_volume
low_stemwood_prop
high_stemwood_prop
low_stembark_prop
high_stembark_prop
low_branches_prop
high_branches_prop
low_foliage_prop
high_foliage_prop

References

Boudewyn, P., Song, X., Magnussen, S., Gillis, M.D. (2007). Model-based, Volume-to-Biomass.
Conversion for Forested and Vegetated Land in Canada. Canadian Forest Service, Victoria, Canada (Inf. Rep. BC-X-411).

