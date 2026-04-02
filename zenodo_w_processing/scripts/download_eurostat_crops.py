# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
# SPDX-License-Identifier: MIT

"""
Download raw crop harvest data from the Eurostat API (dataset apro_cpshr).

This script retrieves the original CSV downloads used as input data for the
PyPSA-Eur workflow. It does not perform preprocessing, harmonization, yield
calculation, or spatial filling.

Outputs
-------
- eurostat_apro_cpshr_nuts2_raw.csv
- eurostat_apro_cpshr_nuts0_raw.csv
"""

import logging
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
logger = logging.getLogger(__name__)


def download_file(url: str, filepath: Path) -> None:
    logger.info("Downloading %s", filepath.name)
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.write_bytes(response.content)
    logger.info("Saved to %s", filepath.resolve())


def download_database_nuts2(filepath: Path) -> None:
    url = (
        "https://ec.europa.eu/eurostat/api/dissemination/sdmx/3.0/data/dataflow/ESTAT/apro_cpshr/1.0/*.*.*.*?"
        "c[freq]=A"
        "&c[crops]=C0000,C1000,C1210,C1300,C1310,C1320,P0000,R2000,I1110-1130,I0000,I1100,I1110,I1120,I1130,I9000,G0000,G1000,G2000,G2100,G2900,G2910,J0000,PECR,PECR9"
        "&c[strucpro]=AR,PR_HU_EU,YI_HU_EU,MA"
        "&c[TIME_PERIOD]=2020,2019,2018,2017"
        "&compress=false&format=csvdata&formatVersion=2.0&lang=en&labels=name"
        "&c[geo]=BE10,BE21,BE22,BE23,BE24,BE25,BE31,BE32,BE33,BE34,BE35,BEZZ,"
        "BG31,BG32,BG33,BG34,BG41,BG42,BGZZ,"
        "CZ01,CZ02,CZ03,CZ04,CZ05,CZ06,CZ07,CZ08,CZZZ,"
        "DK01,DK02,DK03,DK04,DK05,DKZZ,"
        "DE11,DE12,DE13,DE14,DE21,DE22,DE23,DE24,DE25,DE26,DE27,DE30,DE40,DE50,DE60,DE71,DE72,DE73,DE80,DE91,DE92,DE93,DE94,DEA1,DEA2,DEA3,DEA4,DEA5,DEB1,DEB2,DEB3,DEC0,DED2,DED4,DED5,DEE0,DEF0,DEG0,DEZZ,"
        "EE00,EEZZ,"
        "IE04,IE05,IE06,IE01,IE02,IEZZ,"
        "EL11,EL12,EL13,EL14,EL21,EL22,EL23,EL24,EL25,EL30,EL41,EL42,EL43,EL51,EL52,EL53,EL54,EL61,EL62,EL63,EL64,EL65,ELZZ,"
        "ES11,ES12,ES13,ES21,ES22,ES23,ES24,ES30,ES41,ES42,ES43,ES51,ES52,ES53,ES61,ES62,ES63,ES64,ES70,ESZZ,"
        "FR10,FRB0,FRC1,FRC2,FRD1,FRD2,FRE1,FRE2,FRF1,FRF2,FRF3,FRG0,FRH0,FRI1,FRI2,FRI3,FRJ1,FRJ2,FRK1,FRK2,FRL0,FRM0,FRY1,FRY2,FRY3,FRY4,FRY5,FR21,FR22,FR23,FR24,FR25,FR26,FR30,FR41,FR42,FR43,FR51,FR52,FR53,FR61,FR62,FR63,FR71,FR72,FR81,FR82,FR83,FRA1,FRA2,FRA3,FRA4,FRA5,FR91,FR92,FR93,FR94,FRZZ,"
        "HR02,HR03,HR04,HR05,HR06,HR01,"
        "ITC1,ITC2,ITC3,ITC4,ITD1,ITD2,ITD3,ITD4,ITD5,ITE1,ITE2,ITE3,ITE4,ITF1,ITF2,ITF3,ITF4,ITF5,ITF6,ITG1,ITG2,ITH1,ITH2,ITH3,ITH4,ITH5,ITI1,ITI2,ITI3,ITI4,ITZZ,"
        "CY00,CYZZ,"
        "LV00,LVZZ,"
        "LT01,LT02,LTZZ,"
        "LU00,LUZZ,"
        "HU11,HU12,HU10,HU21,HU22,HU23,HU31,HU32,HU33,HUZZ,"
        "MT00,MTZZ,"
        "NL11,NL12,NL13,NL21,NL22,NL23,NL31,NL32,NL33,NL34,NL35,NL36,NL41,NL42,NLZZ,"
        "AT11,AT12,AT13,AT21,AT22,AT31,AT32,AT33,AT34,ATZZ,"
        "PL11,PL12,PL21,PL22,PL31,PL32,PL33,PL34,PL41,PL42,PL43,PL51,PL52,PL61,PL62,PL63,PL71,PL72,PL81,PL82,PL84,PL91,PL92,PLZZ,"
        "PT11,PT15,PT16,PT17,PT18,PT19,PT1A,PT1B,PT1C,PT1D,PT20,PT30,PTZZ,"
        "RO11,RO12,RO21,RO22,RO31,RO32,RO41,RO42,ROZZ,"
        "SI03,SI01,SI04,SI02,SIZZ,"
        "SK01,SK02,SK03,SK04,SKZZ,"
        "FI13,FI18,FI19,FI1A,FI1B,FI1C,FI1D,FI20,FIZZ,"
        "SE11,SE12,SE21,SE22,SE23,SE31,SE32,SE33,SEZZ,"
        "IS00,"
        "NO01,NO02,NO03,NO04,NO05,NO06,NO07,NO08,NO09,NO0A,NO0B,"
        "CH01,CH02,CH03,CH04,CH05,CH06,CH07,"
        "UKC1,UKC2,UKD1,UKD3,UKD4,UKD6,UKD7,UKE1,UKE2,UKE3,UKE4,UKF1,UKF2,UKF3,UKG1,UKG2,UKG3,UKH1,UKH2,UKH3,UKI1,UKI2,UKI3,UKI4,UKI5,UKI6,UKI7,UKJ1,UKJ2,UKJ3,UKJ4,UKK1,UKK2,UKK3,UKK4,UKL1,UKL2,UKM2,UKM3,UKM5,UKM6,UKM7,UKM8,UKM9,UKN0,UKZZ,"
        "ME00,MK00,AL01,AL02,AL03,RS11,RS12,RS21,RS22,"
        "TR10,TR21,TR22,TR31,TR32,TR33,TR41,TR42,TR51,TR52,TR61,TR62,TR63,TR71,TR72,TR81,TR82,TR83,TR90,TRA1,TRA2,TRB1,TRB2,TRC1,TRC2,TRC3"
    )
    download_file(url, filepath)


def download_database_nuts0(filepath: Path) -> None:
    url = (
        "https://ec.europa.eu/eurostat/api/dissemination/sdmx/3.0/data/dataflow/ESTAT/apro_cpshr/1.0/*.*.*.*?"
        "c[freq]=A"
        "&c[crops]=C0000,C1000,C1210,C1300,C1310,C1320,P0000,R2000,I1110-1130,I0000,I1100,I1110,I1120,I1130,I9000,G0000,G1000,G2000,G2100,G2900,G2910,J0000,PECR,PECR9"
        "&c[strucpro]=AR,PR_HU_EU,YI_HU_EU,MA"
        "&c[TIME_PERIOD]=2020,2019,2018,2017"
        "&c[geo]=BE,BG,CZ,DK,DE,EE,IE,EL,ES,FR,HR,IT,CY,LV,LT,LU,HU,MT,NL,AT,PL,PT,RO,SI,SK,FI,SE,NO,CH,IS,ME,MK,AL,RS,TR,UK,EU"
        "&compress=false&format=csvdata&formatVersion=2.0&lang=en&labels=name"
    )
    download_file(url, filepath)


def main():
    ROOT_DIR = Path(__file__).resolve().parents[1]
    output_dir = ROOT_DIR / "downloads"
    download_database_nuts2(output_dir / "eurostat_apro_cpshr_nuts2_raw.csv")
    download_database_nuts0(output_dir / "eurostat_apro_cpshr_nuts0_raw.csv")
    logger.info("Raw downloads complete.")


if __name__ == "__main__":
    main()
