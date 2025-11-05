# SPDX-FileCopyrightText: 2024 Alberto Alamia <your.email@domain.com>
#
# SPDX-License-Identifier: GPL-3.0-or-later


rule retrieve_forest:
    output:
        NAI_data=resources("forests/NAI_evenaged_stands.csv"),
        NAI_regions=resources("forests/NAI_regions_codes.csv"),
        forest_stocks=resources("forests/Standing_stock_evenaged.csv"),

    resources:
        mem_mb=2000,
    log:
        logs("retrieve_forest.log"),
    benchmark:
        benchmarks("retrieve_forest")
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/retrieve_afforestation.py"


rule build_afforestation_inputs:
    input:
        NAI_data=resources("forests/NAI_evenaged_stands.csv"),
        NAI_regions=resources("forests/NAI_regions_codes.csv"),
        forest_stocks=resources("forests/Standing_stock_evenaged.csv"),
        nuts2=data("nuts/NUTS_RG_03M_2013_4326_LEVL_2.geojson")
    output:
        growth_rates_nuts2=data("afforestation_perennials/afforestation_nuts2.csv"),
    resources:
        mem_mb=4000,
    log:
        logs("build_afforestation_inputs.log"),
    benchmark:
        benchmarks("build_afforestation_inputs")
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/build_afforestation_rate_nuts2.py"

