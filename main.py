import pandas as pd
from entise.constants.columns import Columns
from entise.constants import Types
from entise.constants.objects import Objects

from filter import (filter_single_family_detached,
                    get_simulated_years,
                    to_object_file)
from generate_typology import (calculate_rc,
                               generate_buildings,
                               generate_windows_tipology)
from plot_results import barplot_ranking_fit_score, hvac_loads_comparison, normalized_boxplot_2, normalized_boxplot_3, plot_hvac_loads, normalized_boxplot
from validate import (derive_hvac,
                      derive_internal_gains,
                      derive_occupancy_schedule, derive_solar_gains, summarize_hvac, calculate_fit_score)


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
#derive_internal_gains(objects=objects,gains_per_person=gains_per_person)
#
#derive_solar_gains(objects,reduce_window_area,reduce_shading)
#
#derive_hvac(objects,capacitance_factor=capacitance_factor,resistance_factor=resistance_factor,ventilation_mode=ventilation_mode,method=Columns.OCCUPANCY_GEOMA)

#df=summarize_hvac(objects,Columns.OCCUPANCY_GEOMA)
#
#df_best,df_best_heating,df_best_cooling,fit_score,heating_error,cooling_error=calculate_fit_score(df,Columns.OCCUPANCY_GEOMA,name=f"vent_{ventilation_mode}_wshade_{reduce_shading}_warea_{reduce_window_area}_gpp_{gains_per_person}_capac_{capacitance_factor}_resis_{resistance_factor}")
#barplot_ranking_fit_score(Columns.OCCUPANCY_GEOMA,name=f"vent_{ventilation_mode}_wshade_{reduce_shading}_warea_{reduce_window_area}_gpp_{gains_per_person}_capac_{capacitance_factor}_resis_{resistance_factor}")
#hvac_loads_comparison(objects=df_best,res_factor=resistance_factor,cap_factor=capacitance_factor,solar_gains_factor=reduce_shading,method=Columns.OCCUPANCY_GEOMA)

demand_folder="data/validation/demand"
hvac_folder=f"data/validation/hvac"
df_best=pd.read_csv(f"results/best_fit_score_geoma_vent_{ventilation_mode}_wshade_{reduce_shading}_warea_{reduce_window_area}_gpp_{gains_per_person}_capac_{capacitance_factor}_resis_{resistance_factor}.csv")
dfs=[]
for idx,obj in df_best.iterrows():

    obj_id =str(obj[Objects.ID])
    obj_year=obj["year"]
    obj_climate_zone=obj["climate_zone"]

    df_hvac=pd.read_csv(f"{hvac_folder}/{Columns.OCCUPANCY_GEOMA}/{obj_id}_{obj_year}.csv")
    df_hvac_real=pd.read_csv(f"{demand_folder}/{obj_id}.csv")

    df_hvac[Columns.DATETIME]=pd.to_datetime(df_hvac[Columns.DATETIME])
    df_hvac_real[Columns.DATETIME]=pd.to_datetime(df_hvac_real[Columns.DATETIME])

    
    df_hvac_real=df_hvac_real.rename(columns={f"{Types.HEATING}_{Columns.DEMAND}[W]":f"real_{Types.HEATING}_{Columns.DEMAND}[W]"})
    df_hvac_real=df_hvac_real.rename(columns={f"{Types.COOLING}_{Columns.DEMAND}[W]":f"real_{Types.COOLING}_{Columns.DEMAND}[W]"})
    df_hvac_real=df_hvac_real[[Columns.DATETIME,f"real_{Types.HEATING}_{Columns.DEMAND}[W]",f"real_{Types.COOLING}_{Columns.DEMAND}[W]"]]


    
    df_hvac=df_hvac.rename(columns={f"{Types.HEATING}_{Columns.DEMAND}[W]":f"simulated_{Types.HEATING}_{Columns.DEMAND}[W]"})
    df_hvac=df_hvac.rename(columns={f"{Types.COOLING}_{Columns.DEMAND}[W]":f"simulated_{Types.COOLING}_{Columns.DEMAND}[W]"})
    df_hvac=df_hvac[[Columns.DATETIME,f"simulated_{Types.HEATING}_{Columns.DEMAND}[W]",f"simulated_{Types.COOLING}_{Columns.DEMAND}[W]"]]


    df=pd.merge(df_hvac,df_hvac_real,on=Columns.DATETIME, how="outer")

    df[Objects.ID]=obj_id
    
    df["year"]=obj_year
    
    df["climate_zone"]=obj_climate_zone

    df=df.set_index(Columns.DATETIME)

    dfs.append(df)


df_all_hvac=pd.concat(dfs, axis=0)

df_all_hvac=df_all_hvac.reset_index()

normalized_boxplot_2(df_all_hvac)
normalized_boxplot_3(df_all_hvac)







#The Intuition Behind the Quartile Analysis
#
#In your context, you have many buildings within each climate zone, and for each one you can compute a relationship between simulated and real HVAC demand (normalized over time).
#
#Now, if you plot every building individually, you get dozens of overlapping lines — messy and hard to interpret.
#The quartile analysis provides a statistical summary of all those lines to see how consistent the simulation model is across buildings.
#
#⚙️ How It Works
#
#For each time step (or normalized progress point), say 0% → 100% of the total demand period,
#you look at the cumulative normalized HVAC demand for all buildings.
#
#Example: at the halfway point (x = 0.5 simulated demand), some buildings have reached 0.4 real demand, others 0.6, etc.
#
#Across those values, you compute percentiles (quantiles):
#
#5% (lower extreme)
#
#25% (lower quartile)
#
#50% (median)
#
#75% (upper quartile)
#
#95% (upper extreme)
#
#Then, instead of plotting every building, you plot:
#
#A shaded region between 25–75% → shows where half of buildings lie (interquartile range, IQR).
#
#A lighter region between 5–95% → shows where most buildings lie (main variability range).
#
#The median line → shows the “typical” or central trend.
#
#📈 What You Can Conclude from It
#1. Model Accuracy (Bias)
#
#If the median line lies close to the perfect-match line (the diagonal),
#it means that on average, the simulation matches reality quite well.
#
#If the median is below the diagonal → simulated HVAC demand is lower than real.
#If it’s above → simulation overestimates demand.
#
#2. Model Consistency (Spread)
#
#The width of the quartile bands shows how consistent the model is across buildings:
#
#Narrow bands → most buildings behave similarly → the model generalizes well across the zone.
#
#Wide bands → large differences between buildings → the model might fit some building types better than others.
#
#3. Extreme Outliers
#
#The 5–95% shaded region indicates the total variability — if it’s very wide,
#there are outliers or special cases (perhaps unusual building envelopes or occupancy patterns).
#
#4. Systematic Temporal Differences
#
#If the deviation changes along the x-axis (e.g., early/late in the demand curve),
#it means the simulation performs differently during low-load vs. peak-load periods.
#
#Example:
#
#If it’s close to perfect at the beginning but diverges at high loads →
#your model may underestimate peak HVAC demands.
#
#💡 Summary Table
#Feature	Interpretation
#Median near diagonal	Simulation accurate on average
#Median below diagonal	Simulation underestimates HVAC demand
#Median above diagonal	Simulation overestimates HVAC demand
#Narrow quartile bands	Model consistent across buildings
#Wide quartile bands	High variability — model less general
#Asymmetric bands	Systematic bias (e.g., worse for high demand periods)
#🧩 Why It’s So Useful
#
#This method gives a clear, aggregated diagnostic across a population of buildings:
#
#It combines bias (systematic error) and variance (consistency) into one figure.
#
#It’s a visual statistical validation of your model — much more informative than showing individual lines or a single RMSE value.