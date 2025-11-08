import pandas as pd
from filter import (filter_single_family_detached,
                    get_simulated_years,
                    to_object_file)
from generate_typology import (calculate_rc,
                               generate_buildings,
                               generate_windows_tipology)
from plot_results import barplot_ranking_fit_score, hvac_loads_comparison, plot_hvac_loads
from validate import (derive_hvac,
                      derive_internal_gains,
                      derive_occupancy_schedule, derive_solar_gains, summarize_hvac, calculate_fit_score)
from entise.constants.columns import Columns

climate_zone="marine"
gains_per_person=65
reduce_shading=0.2
reduce_window_area=1
ventilation_mode="typical"
capacitance_factor=1
resistance_factor=1


objects = pd.read_csv("data/validation/objects_entise.csv")


#objects=objects.loc[objects["year"]==1995]
#filter_single_family_detached()
##
#get_simulated_years()
#
#generate_buildings()
###
#calculate_rc()
###
#generate_windows_tipology()
#to_object_file()
#
#derive_occupancy_schedule(objects=objects)
#
derive_internal_gains(objects=objects,gains_per_person=gains_per_person)

derive_solar_gains(objects,reduce_window_area,reduce_shading)

derive_hvac(objects,capacitance_factor=capacitance_factor,resistance_factor=resistance_factor,ventilation_mode=ventilation_mode,method=Columns.OCCUPANCY_GEOMA)

df=summarize_hvac(objects,Columns.OCCUPANCY_GEOMA)

df_best,df_best_heating,df_best_cooling,fit_score,heating_error,cooling_error=calculate_fit_score(df,Columns.OCCUPANCY_GEOMA,name=f"vent_{ventilation_mode}_wshade_{reduce_shading}_warea_{reduce_window_area}_gpp_{gains_per_person}_capac_{capacitance_factor}_resis_{resistance_factor}")
barplot_ranking_fit_score(Columns.OCCUPANCY_GEOMA,name=f"vent_{ventilation_mode}_wshade_{reduce_shading}_warea_{reduce_window_area}_gpp_{gains_per_person}_capac_{capacitance_factor}_resis_{resistance_factor}")
hvac_loads_comparison(objects=df_best,res_factor=resistance_factor,cap_factor=capacitance_factor,solar_gains_factor=reduce_shading,method=Columns.OCCUPANCY_GEOMA)


