### Description
This repo is used to pre-process data for CO2 negative stores project.
Perennials and green biorefining are modelled in PyPSA according to the following logic: 
![image](https://github.com/user-attachments/assets/48017eb5-f731-4ec2-8a74-bb7c6b60d3d1)


The repo contains .py files for pre processing of input data and examples of sections of pypsa netowrks
- parameters.py contains the input parameters
- helpers.py contains funcitons
- pre-processing.py contains the main calculations for efficiencies and costs in the multilink, and an example of the code in pypsa 
- cost-2030.csv is an output file from the technology-data repository to be updated with the cost for GBR and perennials 


### PERENNIALS 
Assumptions for modelling perennials and green biorefining implemented in the code: 
1) perennials can grow only on the arable land allocated to 1st generation biofiels in the ENSPRESSO DATABASE (medium scenario)
2) [#TODO add NUTS0 land for 1st gen biofuels]
3) perennials are assumed to have a GHG reducton of 2 tCO2/ha/y compared to seasonal crops (which includes C-storage in the soil and reduciton of N-related emission)
   Reference: https://dcapub.au.dk/djfpublikation/djfpdf/DCArapport193.pdf 
4) Yield of perennials is assumed to be 15 t/ha/y in dry matter
   Reference: https://dcapub.au.dk/djfpublikation/djfpdf/DCArapport193.pdf
5) Green crops (perennials) have a continous effect on reducing GHG, however the production of perennials (hence of energy products) is only during the April-Octobe

### Green biorefining
Green biorefining (GBR) is a process (moslty based on microfiltraiton) where fresh green biomass are separated into products.
Here only a simplified version of GBR is considered where fresh biomass is separated in protein for monogastric animals, and two energy streams: brown juice and press cake
Both energy streams are converted to biogas through an anaerobig digestor, maximizing energy production.
Assumptions for modelling of green biorefining
1) press cake and brown juice are both sent to biogas plant
2) currently the cost of the biogas plant is inlcude in the cost of the GBR (although it is stopped for 1/2 year)
3) mass and energy balance are from: https://doi.org/10.1016/j.scitotenv.2023.167943
4) cost and similar mass balance are from: https://doi.org/10.1016/B978-0-323-95879-0.50147-8
5) everything is recalculated per tCO2e removed from atmsphere compared to seasonal crops 

### PyPSA multilink 
1) all efficiencies and cost are calcualted base on bus0 (CO2e removed from atm)
2) p_max_pu of the multilink is == 1 in April -October and 0 during the rest of the year
3) the potential for CO2 reduction (in tCO2/y) is calcualted per node assuming that the node corresponds to NTSU0

