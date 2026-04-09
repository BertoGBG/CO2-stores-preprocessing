Update	December 2023
Authors:	Viorel Blujdea, Roberto Pilli, Paul Rougieux

This collection includes a library of data reporting the total forest aboveground biomass of the main forest species identified for EU-27 MS. All data were derived by species-specific allometric equations selected through a modelling approach, using as input National Forest Inventories and other data sources, collected and harmonized by the JRC (Pilli et al., 2024). Further details on the original data sources and the classification system applied at country level are reported on the companion collection Volume_increment_database.
Present collection also reports the following biomass' components: stem (i.e., merchantable wood biomass excluding bark), bark, branches (including both main and small branches) and foliages. All biomass data are reported in t dry biomass ha-1, further distinguished by age classes for even-aged stands.
The database also includes the biomass conversion and expansion factor (BCEF) associated to each species and age class, estimated as the ratio between the merchantable volume associated to each record (as reported within the companion collection Volume_increment_database) and the corresponding total aboveground biomass.

All data were classified according to the following items:
- countries: all EU-27 Memebr States, except Cyprus and Malta, where detailed information are missing.  See Country_codes.csv in Volume_increment_database.
- status: defined, when possible, as Forest Area Available for Wood Supply (FAWS) and Forest Area Not Available for Wood Supply (FNAWS). See Status_codes.csv in Volume_increment_database.
- forest types: defined according to the leading species identified at country level. See Forest_codes.csv in Volume_increment_database.
- administrative regions (when possible), generally identified at NUTS 2 level or according to other classification system derived from NFI data. See Regions_codes.csv in Volume_increment_database.
- management types: mostly including high forests (H) and coppices (C) or otherwise defined within the countries' sheets. See Management_codes.csv in Volume_increment_database.
- management strategies: mostly including even-aged (E) and uneven-aged (U) forests.  See Management_codes.csv in Volume_increment_database.
- species' group: distinguished between coniferous (con) and broadleaves (broad) species. See Forest_codes.csv in Volume_increment_database..
- age_class: age classes distribution applied to even-aged stands
- volume: merchantable standing volume (m3 ha-1 u.b.)
- stem_biomass: merchantable standing biomass under bark (t DM ha-1)
- bark_biomass: bark biomass (t DM ha-1)
- branches_biomass: branches biomass (t DM ha-1)
- foliage_biomass: biomass of foliages (t DM ha-1)
- foliage/stem ratio: stem_biomass/foliage_biomass. 
- total_agb*: total aboveground biomass (t DM ha-1)
- bcef: Biomass Conversion and Expansion factor = total_agb/volume (t DM m-3)

*foliage/stem ratio was used as a proxy to exclude outliers on foliage biomass computation (e.g., for HR, forest type QP):
For AgeCL_1: IF foliage/stem ratio >300% --> total_agb =stem_biomass+bark_biomass+branches_biomass, ELSE total_agb = stem_biomass+bark_biomass + branches_biomass+foliage_biomass
For age_classes > AgeCL_1: IF foliage/stem ratio >100% --> total_agb = stem_biomass+bark_biomass +branches_biomass, ELSE total_agb = stem_biomass+bark_biomass + branches_biomass+foliage_biomass

Detailed information on the classification system and data sources are reported on the companion collection Volume_increment_databse

Other information
Special character "-" is used for age classes where no volume/increment is reported on the library (e.g. for fast growing species and age classes > AgeCL_10).

Age classes considered for even-aged stands
Age_classes
Age_interval (yrs.)
AgeCL_1
0-10
AgeCL_2
11-20
AgeCL_3
21-30
AgeCL_4
31-40
AgeCL_5
41-50
AgeCL_6
51-60
AgeCL_7
61-70
AgeCL_8
71-80
AgeCL_9
81-90
AgeCL_10
91-100
AgeCL_11
101-110
AgeCL_12
111-120
AgeCL_13
121-130
AgeCL_14
131-140
AgeCL_15
141-150
AgeCL_16
151-160
AgeCL_17
161-170
AgeCL_18
171-180
AgeCL_19
181-190
AgeCL_20
>190
