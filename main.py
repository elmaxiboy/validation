import logging
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from entise.constants.columns                           import Columns
from entise.constants                                   import Types,SEP
from entise.constants.objects                           import Objects
from entise.constants.general                           import Keys
from entise.core.generator                              import Generator
from entise.methods.auxiliary.internal.strategies       import InternalOccupancy
from entise.methods.hvac                                import R1C1
from entise.methods.auxiliary.ventilation.strategies    import VentilationTimeSeries
from teaser.project                                     import Project
from helpers import generate_ach_series

def build_plot(hvac_sim_arr, hvac_real_arr, cz, title_suffix, filename_suffix, mode):
            if len(hvac_sim_arr) == 0:
                return

            hvac_sim_arr = np.array(hvac_sim_arr)
            hvac_real_arr = np.array(hvac_real_arr)

            # Colors by mode
            if mode == "combined":
                c_5_95 = "orange"
                c_25_75 = "orange"
                c_median = "darkred"
            elif mode == "heating":
                c_5_95 = "#ffcccc"
                c_25_75 = "#ff6666"
                c_median = "#990000"
            elif mode == "cooling":
                c_5_95 = "#cce6ff"
                c_25_75 = "#66b3ff"
                c_median = "#004c99"

            # -----------------------------------------
            # Correct quantile computation:
            # REAL = X-axis, SIM = Y-axis
            # -----------------------------------------
            q_levels = [0.05, 0.25, 0.5, 0.75, 0.95]

            qs_real = np.quantile(hvac_real_arr, q_levels, axis=0)
            qs_sim  = np.quantile(hvac_sim_arr,  q_levels, axis=0)

            # Sort by REAL median to ensure monotonic X
            x_med = qs_real[2]
            sort_idx = np.argsort(x_med)

            x_sorted = x_med[sort_idx]
            y_med = qs_sim[2][sort_idx]
            y_25  = qs_sim[1][sort_idx]
            y_75  = qs_sim[3][sort_idx]
            y_5   = qs_sim[0][sort_idx]
            y_95  = qs_sim[4][sort_idx]

            # -----------------------------------------
            # Plotting
            # -----------------------------------------
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.fill_between(x_sorted, y_5, y_95, color=c_5_95, alpha=0.35, label="5–95% range")
            ax.fill_between(x_sorted, y_25, y_75, color=c_25_75, alpha=0.60, label="25–75% quartile")
            ax.plot(x_sorted, y_med, color=c_median, lw=2.5, label="Median")

            # Perfect match line
            ax.plot([0, 1], [0, 1], "k--", label="Perfect Match")

            # Labels
            ax.set_xlabel("Normalized Real",fontsize=16)
            ax.set_ylabel("Normalized Simulated",fontsize=16)
            ax.set_title(f"HVAC Normalized Quartile Comparison ({title_suffix}) — {cz.capitalize()}",fontsize=16)
            
            ax.tick_params(axis='both', which='major', labelsize=12)
            ax.tick_params(axis='both', which='minor', labelsize=10)
            
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1.1)
            ax.grid(True)
            ax.legend(loc="lower right",fontsize=16)

            plt.tight_layout()
            plt.savefig(f"normalized_boxplot_quartiles_{filename_suffix}_{cz}.png", dpi=300)
            plt.close()

def hvac_loads_comparison(
    obj,
    sim_heat_kwh,
    sim_cool_kwh,
    real_heat_kwh,
    real_cool_kwh,
    weather,
    output_dir="report_plots",
):
    obj_id = obj[Objects.ID]
    os.makedirs(output_dir, exist_ok=True)

    # --- Build a clean plotting dataframe and align lengths ---
    n = min(
        len(sim_heat_kwh),
        len(sim_cool_kwh),
        len(real_heat_kwh),
        len(real_cool_kwh),
        len(weather),
    )

    plot_df = pd.DataFrame({
        "datetime": pd.to_datetime(weather[Columns.DATETIME]).iloc[:n].to_numpy(),
        "real_heat_kwh": np.asarray(real_heat_kwh)[:n],
        "real_cool_kwh": np.asarray(real_cool_kwh)[:n],
        "sim_heat_kwh": np.asarray(sim_heat_kwh)[:n],
        "sim_cool_kwh": np.asarray(sim_cool_kwh)[:n],
    }).dropna(subset=["datetime"])

    fig, ax1 = plt.subplots(figsize=(14, 6))

    # HVAC loads on primary axis
    l1 = ax1.plot(
        plot_df["datetime"], plot_df["real_heat_kwh"],
        color="darkred", linewidth=1.2, label="Real Heating Load", zorder=3
    )
    l2 = ax1.plot(
        plot_df["datetime"], plot_df["real_cool_kwh"],
        color="royalblue", linewidth=1.2, label="Real Cooling Load", zorder=3
    )
    l3 = ax1.plot(
        plot_df["datetime"], plot_df["sim_heat_kwh"],
        color="red", alpha=0.7, linewidth=1.0, label="Simulated Heating Load", zorder=2
    )
    l4 = ax1.plot(
        plot_df["datetime"], plot_df["sim_cool_kwh"],
        color="deepskyblue", alpha=0.7, linewidth=1.0, label="Simulated Cooling Load", zorder=2
    )

    # Keep load axis visually on top
    ax1.patch.set_visible(False)

    ax1.set_xlabel("Time")
    ax1.set_ylabel("HVAC load [kWh]")


    ax1.set_title(f"HVAC load comparison over time — {obj_id}")
    ax1.grid(True, alpha=0.3)

    # Combined legend
    handles = l1 + l2 + l3 + l4
    labels = [h.get_label() for h in handles]
    ax1.legend(handles, labels, loc="upper left", frameon=True)

    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{obj_id}_hvac_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

# Load data
cwd = f""
objects = pd.read_csv(os.path.join(cwd, "objects.csv"))
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
ach_series              =   generate_ach_series()
ventilation_gen         =   VentilationTimeSeries()
hvac_r1c1_gen           =   R1C1()

summary, dfs = gen.generate(data, workers=1)

comb_sim_list, comb_real_list = [], []
heat_sim_list, heat_real_list = [], []
cool_sim_list, cool_real_list = [], []

for obj_id,ts in dfs.items():
    
    obj                     =   objects[objects[Objects.ID]==obj_id].iloc[0]
    
    ts[Types.OCCUPANCY]         =   ts[Types.OCCUPANCY].squeeze("columns")
    data["internal_gains"]      =   internal_occupancy_gen.generate(obj,ts)
    data["ach_series"]          =   ach_series
    
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

    ts[Types.HVAC]              =   hvac_r1c1_gen.generate(obj,data)

    sim_heating_load_kwh      =   (ts[Types.HVAC][Keys.TIMESERIES][f"{Types.HEATING}{SEP}{Columns.LOAD}[W]"]/4/1000).values
    sim_cooling_load_kwh      =   (ts[Types.HVAC][Keys.TIMESERIES][f"{Types.COOLING}{SEP}{Columns.LOAD}[W]"]/4/1000).values

    real_heating_load_kwh     =   ((ts[Types.ELECTRICITY][f"{Types.HEATING}{SEP}{Columns.LOAD}[W]"]/4)/1000).values 
    real_cooling_load_kwh     =   ((ts[Types.ELECTRICITY][f"{Types.COOLING}{SEP}{Columns.LOAD}[W]"]/4)/1000).values

    sim_total_kwh   = sim_heating_load_kwh + sim_cooling_load_kwh
    real_total_kwh  = real_heating_load_kwh + real_cooling_load_kwh
    
    real_sum_kwh    = real_total_kwh.sum()

    real_h_sum_kwh = real_heating_load_kwh.sum()
    real_c_sum_kwh = real_cooling_load_kwh.sum()
    
    # Combined
    if real_sum_kwh > 0:
        comb_sim_list.append(np.cumsum(sim_total_kwh) / real_sum_kwh)
        comb_real_list.append(np.cumsum(real_total_kwh) / real_sum_kwh)
    # Heating
    if real_h_sum_kwh > 0:
        heat_sim_list.append(np.cumsum(sim_heating_load_kwh) / real_h_sum_kwh)
        heat_real_list.append(np.cumsum(real_heating_load_kwh) / real_h_sum_kwh)
    # Cooling
    if real_c_sum_kwh > 0:
        cool_sim_list.append(np.cumsum(sim_cooling_load_kwh) / real_c_sum_kwh)
        cool_real_list.append(np.cumsum(real_cooling_load_kwh) / real_c_sum_kwh)
    
    hvac_loads_comparison(obj,sim_heating_load_kwh,sim_cooling_load_kwh,real_heating_load_kwh,real_cooling_load_kwh,data["marine"])

build_plot(comb_sim_list, comb_real_list, "marine", "Combined HVAC", "combined", "combined")
build_plot(heat_sim_list, heat_real_list, "marine", "Heating Only", "HEATING", "heating")
build_plot(cool_sim_list, cool_real_list, "marine", "Cooling Only", "COOLING", "cooling")


print("Summary:")
print(summary)

