import pandas as pd

CLIMATE_ZONES = {
    "hot dry"   : {"base": 0.5, "season_amp": 0.3},
    "hot humid" : {"base": 0.7, "season_amp": 0.25},
    "mixed dry" : {"base": 0.6, "season_amp": 0.20},
    "cold"      : {"base": 0.4, "season_amp": 0.35},
    "very cold" : {"base": 0.3, "season_amp": 0.40},
    "marine"    : {"base": 0.5, "season_amp": 0.20},
}

VENTILATION_MODES = {
    "typical [1/h]"     : 1.0,
    "efficient  [1/h]"  : 0.7,
    "optimal [1/h]"     : 0.4
}

DT_INDEX = pd.date_range(
    start   =   "2018-01-01 00:00",
    end     =   "2018-12-31 23:45",
    freq    =   "15min"
)
