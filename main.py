import logging
import os
import numpy as np
import pandas as pd

from entise.constants.columns                           import Columns
from entise.constants                                   import Types, SEP
from entise.constants.objects                           import Objects
from entise.constants.general                           import Keys
from entise.core.generator                              import Generator
from entise.methods.auxiliary.internal.strategies       import InternalOccupancy
from entise.methods.hvac                                import R1C1
from entise.methods.auxiliary.ventilation.strategies    import VentilationTimeSeries
from teaser.project                                     import Project
from helpers import (generate_ach_series,
                     get_selection_score,
                     plot_normalized_boxplot,
                     relative_error,
                     rmse)


SELECTION_METRIC    = "relative_error"          # "rmse" or "relative_error"
SELECTION_COMPONENT = "heating"      # "heating", "cooling", or "total"

# Load data
cwd = ""
objects = pd.read_csv(os.path.join(cwd, "objects.csv"))
data = {}

data_folder = "data"
for file in os.listdir(os.path.join(cwd, data_folder)):
    if file.endswith(".csv"):
        name = file.split(".")[0]
        data[name] = pd.read_csv(os.path.join(cwd, data_folder, file), parse_dates=True)

print("Loaded data keys:", list(data.keys()))
print(objects)

gen = Generator(logging_level=logging.WARNING)
gen.add_objects(objects)

internal_occupancy_gen = InternalOccupancy()
ach_series = generate_ach_series()
ventilation_gen = VentilationTimeSeries()
hvac_r1c1_gen = R1C1()

summary, dfs = gen.generate(data, workers=1)

# Best candidate per dwelling (base_id)
best_by_dwelling = {}

for obj_id, ts in dfs.items():

    obj = objects[objects[Objects.ID] == obj_id].iloc[0].copy()

    full_id = str(obj[Objects.ID])
    base_id, construction_year = full_id.rsplit("_", 1)
    construction_year = int(construction_year)

    ts[Types.OCCUPANCY] = ts[Types.OCCUPANCY].squeeze("columns")
    data["internal_gains"] = internal_occupancy_gen.generate(obj, ts)
    data["ach_series"] = ach_series

    prj = Project()
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

    bldg = prj.buildings[0]

    bldg.calc_building_parameter(
        number_of_elements=1,
        merge_windows=True,
        used_library="IBPSA"
    )

    zone = bldg.thermal_zones[0]
    m = zone.model_attr

    resistance_total = float(m.r_total_ow)
    capacitance_air = zone.volume * zone.density_air * zone.heat_capac_air
    capacitance_total = float(m.c1_ow) + capacitance_air

    obj[Objects.RESISTANCE] = resistance_total
    obj[Objects.CAPACITANCE] = capacitance_total

    ts[Types.HVAC] = hvac_r1c1_gen.generate(obj, data)

    sim_heating_load_kwh = (
        ts[Types.HVAC][Keys.TIMESERIES][f"{Types.HEATING}{SEP}{Columns.LOAD}[W]"] / 4 / 1000
    ).values
    sim_cooling_load_kwh = (
        ts[Types.HVAC][Keys.TIMESERIES][f"{Types.COOLING}{SEP}{Columns.LOAD}[W]"] / 4 / 1000
    ).values

    real_heating_load_kwh = (
        ts[Types.ELECTRICITY][f"{Types.HEATING}{SEP}{Columns.LOAD}[W]"] / 4 / 1000
    ).values
    real_cooling_load_kwh = (
        ts[Types.ELECTRICITY][f"{Types.COOLING}{SEP}{Columns.LOAD}[W]"] / 4 / 1000
    ).values

    sim_total_kwh       =   sim_heating_load_kwh + sim_cooling_load_kwh
    real_total_kwh      =   real_heating_load_kwh + real_cooling_load_kwh

    sim_h_sum_kwh       =   sim_heating_load_kwh.sum()
    sim_c_sum_kwh       =   sim_cooling_load_kwh.sum()
    sim_total_sum_kwh   =   sim_total_kwh.sum()

    real_h_sum_kwh      =   real_heating_load_kwh.sum()
    real_c_sum_kwh      =   real_cooling_load_kwh.sum()
    real_total_sum_kwh  =   real_total_kwh.sum()

    metrics = {
        "rmse": {
            "heating"   :   rmse(real_heating_load_kwh, sim_heating_load_kwh),
            "cooling"   :   rmse(real_cooling_load_kwh, sim_cooling_load_kwh),
            "total"     :   rmse(real_total_kwh, sim_total_kwh),
        },
        "relative_error": {
            "heating"   :   relative_error(real_h_sum_kwh, sim_h_sum_kwh),
            "cooling"   :   relative_error(real_c_sum_kwh, sim_c_sum_kwh),
            "total"     :   relative_error(real_total_sum_kwh, sim_total_sum_kwh),
        },
    }

    selection_score = get_selection_score(
        metrics,
        selection_metric    =   SELECTION_METRIC,
        selection_component =   SELECTION_COMPONENT,
    )

    candidate = {
        "base_id"               : base_id,
        "year"                  : construction_year,
        "obj"                   : obj,
        "metrics"               : metrics,
        "selection_score"       : selection_score,
        "sim_heating_load_kwh"  : sim_heating_load_kwh,
        "sim_cooling_load_kwh"  : sim_cooling_load_kwh,
        "real_heating_load_kwh" : real_heating_load_kwh,
        "real_cooling_load_kwh" : real_cooling_load_kwh,
        "sim_total_kwh"         : sim_total_kwh,
        "real_total_kwh"        : real_total_kwh,
        "sim_h_sum_kwh"         : sim_h_sum_kwh,
        "sim_c_sum_kwh"         : sim_c_sum_kwh,
        "sim_total_sum_kwh"     : sim_total_sum_kwh,
        "real_h_sum_kwh"        : real_h_sum_kwh,
        "real_c_sum_kwh"        : real_c_sum_kwh,
        "real_total_sum_kwh"    : real_total_sum_kwh,
    }

    current_best = best_by_dwelling.get(base_id)
    if current_best is None or candidate["selection_score"] < current_best["selection_score"]:
        best_by_dwelling[base_id] = candidate

# Build normalized arrays ONLY from selected best years
comb_sim_list, comb_real_list = [], []
heat_sim_list, heat_real_list = [], []
cool_sim_list, cool_real_list = [], []

selection_rows = []

for base_id, rec in best_by_dwelling.items():
    selection_rows.append({
        "base_id"               : base_id,
        "selected_year"         : rec["year"],
        "selection_metric"      : SELECTION_METRIC,
        "selection_component"   : SELECTION_COMPONENT,
        "selection_score"       : rec["selection_score"],
        "rmse_heating"          : rec["metrics"]["rmse"]["heating"],
        "rmse_cooling"          : rec["metrics"]["rmse"]["cooling"],
        "rmse_total"            : rec["metrics"]["rmse"]["total"],
        "re_heating"            : rec["metrics"]["relative_error"]["heating"],
        "re_cooling"            : rec["metrics"]["relative_error"]["cooling"],
        "re_total"              : rec["metrics"]["relative_error"]["total"],
    })

    if rec["real_total_sum_kwh"] > 0:
        comb_sim_list.append(np.cumsum(rec["sim_total_kwh"]) / rec["real_total_sum_kwh"])
        comb_real_list.append(np.cumsum(rec["real_total_kwh"]) / rec["real_total_sum_kwh"])

    if rec["real_h_sum_kwh"] > 0:
        heat_sim_list.append(np.cumsum(rec["sim_heating_load_kwh"]) / rec["real_h_sum_kwh"])
        heat_real_list.append(np.cumsum(rec["real_heating_load_kwh"]) / rec["real_h_sum_kwh"])

    if rec["real_c_sum_kwh"] > 0:
        cool_sim_list.append(np.cumsum(rec["sim_cooling_load_kwh"]) / rec["real_c_sum_kwh"])
        cool_real_list.append(np.cumsum(rec["real_cooling_load_kwh"]) / rec["real_c_sum_kwh"])

selection_df = pd.DataFrame(selection_rows).sort_values(["base_id"])

print("\nSelected best year per dwelling:")
print(selection_df)

plot_normalized_boxplot(comb_sim_list, comb_real_list, "marine", "Combined HVAC", "combined", "combined")
plot_normalized_boxplot(heat_sim_list, heat_real_list, "marine", "Heating Only", "HEATING", "heating")
plot_normalized_boxplot(cool_sim_list, cool_real_list, "marine", "Cooling Only", "COOLING", "cooling")