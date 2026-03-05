import logging
import os

import pandas as pd
from entise.constants.columns import Columns
from entise.constants import Types
from entise.constants.objects import Objects
from entise.core.generator import Generator
from entise.methods.auxiliary.internal.strategies import InternalOccupancy
from entise.methods.hvac                       import R1C1
from entise.methods.auxiliary.ventilation.strategies       import VentilationTimeSeries
from teaser.project import Project

# Load data
cwd = f""
objects = pd.read_csv(os.path.join(cwd, "object.csv"))
data = {}

data_folder = "data"
for file in os.listdir(os.path.join(cwd, data_folder)):
    if file.endswith(".csv"):
        name = file.split(".")[0]
        data[name] = pd.read_csv(os.path.join(os.path.join(cwd, data_folder, file)), parse_dates=True)

print("Loaded data keys:", list(data.keys()))
print(objects)

gen = Generator(logging_level=logging.WARNING)
gen.add_objects(objects)

internal_occupancy_gen  =   InternalOccupancy()
hvac_r1c1_gen           =   R1C1()

summary, dfs = gen.generate(data, workers=1)

for obj_id,ts in dfs.items():
    
    obj                         =   objects[objects[Objects.ID]==obj_id].iloc[0]
    
    ts[Types.OCCUPANCY]         =   ts[Types.OCCUPANCY].squeeze("columns")
    ts[Objects.GAINS_INTERNAL]  =   internal_occupancy_gen.generate(obj,ts)
    
    prj                         =   Project()
    prj.add_residential(
            construction_data                   =   'tabula_de_standard',
            geometry_data                       =   'tabula_de_single_family_house',
            name                                =   obj_id,
            year_of_construction                =   obj["year"],
            number_of_floors                    =   obj["stories"],
            height_of_floors                    =   obj[Objects.HEIGHT],
            net_leased_area                     =   obj[Objects.AREA],
            inner_wall_approximation_approach   =   "teaser_default",
            )
    prj.calc_all_buildings()
    
    bldg    =   prj.buildings[0]

    bldg.calc_building_parameter(
        number_of_elements  =   1, # aggregates to 1R, 1C
        merge_windows       =   True,
        used_library        =   "IBPSA"
        )
    
    zone    =   bldg.thermal_zones[0]
    m       =   zone.model_attr

    resistance_total    =   float(m.r_total_ow)
    capacitance_air     =   zone.volume * zone.density_air * zone.heat_capac_air
    capacitance_total   =   float(m.c1_ow) + capacitance_air
    
    obj[Objects.RESISTANCE]     =   resistance_total
    obj[Objects.CAPACITANCE]    =   capacitance_total 

    ts[Types.HVAC]=hvac_r1c1_gen.generate(obj,data)

print("Summary:")
print(summary)