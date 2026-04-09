Update	December 2023
Authors:	Roberto Pilli, Viorel Blujdea, Paul Rougieux

This collection includes a library of volume and increment forest growth curves for EU-27 MS. Data were derived by 
National Forest Inventories and other data sources, collected, modelled and harmonized by the Joint Research Centre. 
Original data sources are mostly referred to the period 2005 - 2015.

The collection includes the following libraries:
- Standing_stock_evenaged:  reports the net merchantable standing stock per ha (in m3 ha-1) and age class for even-aged forest stands.
- NAI_evenaged_stands: reports the merchantable net annual increment per ha (NAI, in m3 ha-1 yr-1) and by age classes for even-aged stands.
- Unevenaged_stands: reports the net merchantable standing stock (in m3 ha-1) and net annual stock change per ha (m3 ha-1 yr-1) for uneven-aged stands .

All data were further classified according to the following items:
- countries: all EU-27 Memebr States, except Cyprus and Malta, where detailed information are missing.  See Country_codes.csv for details.	
- status: defined, when possible, as Forest Area Available for Wood Supply (FAWS) and Forest Area Not Available for Wood Supply (FNAWS). See Status_codes.csv for details.
- forest types: defined according to the leading species identified at country level. See Forest_codes.csv for details.
- administrative regions (when possible), generally identified at NUTS 2 level or according to other classification system derived from NFI data. See Regions_codes.csv for details.
- management types: mostly including high forests (H) and coppices (C). See Management_codes.csv for details.
- management strategies: mostly including even-aged (E) and uneven-aged (U) forests. See Management_codes.csv for details.
- species' group: distinguished between coniferous (con) and broadleaves (broad) species. See Forest_codes.csv for details.

Standing stock data harmonization for even aged stands:
The standing (growing) volume (m3/ha) data were preliminary harmonized to a common definition as the merchantable volume under bark - excluding branches, top and stump - of all living trees with a diameter at 
breast hight (dbh)>= 9 cm. To harmonize original input data, we collected from NFI and literature (Gschwantner et al., 2019; Gschwantner et al., 2022) country specific information on bark, small and main 
branches, top and stump eventually included within the country s definition of the merchantable standing stock. Based on this assessment we applied country specific correction factors to the original volume data 
reported by countries, to exclude bark and other non-merchantable biomass components. Once corrected the original values, for the even-aged forest stands we estimated country specific growth curves by 
interpolating, through a Chapman-Richard or similar exponential functions  (Richards, 1959), the volume reported for each FT (forest type) and, when possible, MT (management type) and region, against constant 
age classes with 10 yrs span.
When the distribution of volume versus age classes was not available, we derived the functionally equivalent growth curves from a predetermined library, based on a literature review, selecting from this library, the 
growth curve with the average volume most similar to the one reported at country level within the same age class interval reported by original data sources (see Pilli et al., 2013). In few cases, where no data was 
available, we used growth curves derived from other conterminous countries or similar FTs.
This data can be used for the initialization of standing stocks in living biomass and other pools within the EU forest carbon model (Blujdea et al., 2022).

Increment data harmonization for even aged stands
Such as for volume's harmonization, we preliminarily assessed the various biomass components included within the definition of increment (in m3/ha/yr of the standing stock) applied at national level and, in most 
cases, we excluded Annual Natural Losses(ANL)*. When possible, we based our assessment on the information collected by countries or by literature (see also Avitabile et al., 2023) and we harmonized the 
original increment values by applying, to each country, specific correction factors, in order to exclude non-merchantable woody components, bark and ANL. 
Once corrected the original values, we estimated the evolution of the increment against time using a combined exponential and power function, applied to each FT, MT and (where possible) region. 
When the distribution of increment versus age classes was not available, we derived the growth curves from a predetermined library, with the same approach applied to the standing stock (see Pilli et al., 2013). In few 
cases, where no data was available, we used growth curves derived from other conterminous countries or similar forest types. This data can be used for the simulation of forest dynamics within the EU forest carbon model 
(Blujdea et al., 2022).
* Original country data are mostly referred to Gross Annual Increment = Net Annual Increment + Annual Natural Losses

Uneven-aged stands data
Volume and increment data referred to unven-aged stands (when relevant  at country level) were harmonized according to the same approach described for even-aged stands (i.e. excluding bark and non-
merchantable wood components). For uneven-aged stands we  report the harmonized average merchantable standing stock, in m3 per ha, and the average net annual increment of the merchantable 
stock, in m3 per ha per year, excluding natural losses and other disturbance events. These last values should be considered with some caution, since they could also include losses due to management activities 
(e.g., thinnings) or various adjustments (e.g., pure or mixed stands), depending from the approach applied at country level for data collection.

Other information
Across the tables, "?" reflects general applicability for all classification criteria
Special character "-" is used for age classes where no volume/increment is reported on the library (e.g. for 
fast growing species and age classes > AgeCL_10).

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

Countries 
NFI refef. year
Austria
2007-2009
Belgium
1994-2012
Bulgaria
2010
Croatia
2005
Cyprus
 
Czechia
2009-2017
Denmark
2013
Estonia
2010
Finland
2009-13/2014-18
France
2006-2010
Germany
2011-2012
Greece
1992
Hungary
2015
Ireland
2005
Italy
2005/2015
Latvia
2004-2008
Lithuania
2010-2014
Luxembourg
2010
Malta
 
Netherlands
1997
Poland
2010-2014
Portugal
2015
Romania
2010
Slovakia
2010
Slovenia
2009
Spain
2004-2018
Sweden
2010

Detailed information on data sources used for each country are reported in:
Pilli et al. (2024). The Calibration of the JRC Forest Carbon Model for the period 2010   2020. URL: 
https://publications.jrc.ec.europa.eu/repository/handle/JRC135639

General references:
Avitabile V., Pilli R., Migliavacca, M., Camia A., Mubareka S., 2023. Ch 6: Forest Biomass Production. In: 
Avitabile et al. Biomass production, supply, uses and flows in the European Union. Integrated assessment. 
Mubareka S, Migliavacca M, S nchez L pez J (Editors).Publications Office of the European Union, 
Luxembourg, 2023, doi:10.2760/484748, JRC132358
Blujdea, V.N.B., Rougieux, P., Sinclair, L., Morken, S., Pilli, R., Grassi, G., Mubareka, S. and Kurz, A., W., 
The JRC Forest Carbon Model: description of EU-CBM-HAT. Publications Office of the European Union, 
Luxembourg, 2022. doi:10.2760/244051. JRC130609.
Gschwantner, T., et al. (2019). Harmonisation of stem volume estimates in European National Forest 
Inventories. Annals of forest science, 76(1), 1-23.
Pilli R., Grassi G., Kurz W. A., Smyth C. E., Blujdea V., 2013. Application of the CBM-CFS3 model to 
estimate Italy s forest carbon budget, 1995 to 2020, Ecol. Model., 266, 144 171, 
doi:10.1016/j.ecolmodel.2013.07.007.
Richards, F.J., 1959. A flexible growth function for empirical use [J]. J. Exp. Botany,10(2): 290 300.

