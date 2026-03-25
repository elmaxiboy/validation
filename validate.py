
import logging
import os
import time
import numpy
import pandas as pd

# Import the new TimeSeriesGenerator
from entise.constants import Types
from entise.constants.columns import Columns
from entise.constants.objects import Objects
from entise.constants.constants import Constants
from entise.core.generator import TimeSeriesGenerator
from entise.methods.auxiliary.internal.strategies import InternalOccupancy
from entise.methods.auxiliary.ventilation.strategies import VentilationTimeSeries,VentilationConstant
from entise.methods.hvac.R1C1 import calculate_timeseries 
from entise.methods.auxiliary.solar.strategies import SolarGainsPVLib
import pvlib
import tqdm

cwd="."
solar_gains_folder="data/validation/solar_gains"
demand_folder="data/validation/demand"
occupancy_folder=f"data/validation/{Types.OCCUPANCY}"
hvac_folder=f"data/validation/hvac"
weather_folder="data/validation/weather"
real_demand_folder="C:/Users/escobarm/Desktop/thesis/validation_data/counties"
temperature_setpoints_folder="data/validation/temperature_setpoints"


def summarize_hvac(objects,method:str=Columns.OCCUPANCY_GEOMA):
    objects=objects.copy()
    df_summary=pd.DataFrame(columns=[
    Objects.ID,
    "year",
    "climate_zone",
    "state",
    Objects.INHABITANTS,
    Objects.RESISTANCE,
    Objects.CAPACITANCE,
    Objects.AREA,
    "stories",
    "method",
    Objects.LAT,
    Objects.LON,
    Objects.TEMP_MAX,
    Objects.TEMP_MIN
    ])

    for idx,obj in tqdm.tqdm(objects.iterrows(), total=len(objects), desc="Summaryzing HVAC"):

        obj_id =str(obj[Objects.ID])
        obj_year=obj["year"]
        
        obj_stories=obj["stories"]
        obj_state=obj["state"]
        obj_climate_zone=obj["climate_zone"]
        obj_area=obj[Objects.AREA]
        obj_filename=obj[Objects.FILE]
        obj_inhabitants=obj[Objects.INHABITANTS]
        obj_resistance=obj[Objects.RESISTANCE]
        obj_capacitance=obj[Objects.CAPACITANCE]
        obj_latitude=   obj[Objects.LAT]
        obj_longitude=obj[Objects.LON]
        obj_temp_max=obj[Objects.TEMP_MAX]
        obj_temp_min=obj[Objects.TEMP_MIN]

        df_hvac=pd.read_csv(f"{hvac_folder}/{method}/{obj_id}_{obj_year}.csv")
        df_hvac_real=pd.read_csv(f"{demand_folder}/{obj_id}.csv")

        # RMSE calculation

        sim_h = df_hvac[f"{Types.HEATING}_{Columns.DEMAND}[W]"].fillna(0).values
        real_h = df_hvac_real[f"{Types.HEATING}_{Columns.DEMAND}[W]"].fillna(0).values
        
        sim_c = df_hvac[f"{Types.COOLING}_{Columns.DEMAND}[W]"].fillna(0).values
        real_c = df_hvac_real[f"{Types.COOLING}_{Columns.DEMAND}[W]"].fillna(0).values
        rmse_h = float(numpy.sqrt(numpy.mean((sim_h - real_h)**2)))
        rmse_c = float(numpy.sqrt(numpy.mean((sim_c - real_c)**2)))

        # Peak real loads (to normalize RMSE)
        peak_real_h = real_h.max() if real_h.max() > 0 else numpy.nan
        peak_real_c = real_c.max() if real_c.max() > 0 else numpy.nan

        # RMSE per m²
        rmse_h_area = rmse_h / obj_area
        rmse_c_area = rmse_c / obj_area

        # RMSE as % of peak real load
        rmse_h_pct = rmse_h / peak_real_h if peak_real_h not in [0, numpy.nan] else numpy.nan
        rmse_c_pct = rmse_c / peak_real_c if peak_real_c not in [0, numpy.nan] else numpy.nan

        # Save in summary dataframe
        df_summary.at[idx, "RMSE_heating[W]"] = round(rmse_h, 2)
        df_summary.at[idx, "RMSE_cooling[W]"] = round(rmse_c, 2)

        df_summary.at[idx, "RMSE_heating[W/m²]"] = round(rmse_h_area, 4)
        df_summary.at[idx, "RMSE_cooling[W/m²]"] = round(rmse_c_area, 4)

        df_summary.at[idx, "RMSE_heating[%peak]"] = round(rmse_h_pct, 4)
        df_summary.at[idx, "RMSE_cooling[%peak]"] = round(rmse_c_pct, 4)

        df_summary.at[idx,Objects.ID] = obj_id
        df_summary.at[idx, "method"] = method 
        df_summary.at[idx,"year"] = obj_year
        df_summary.at[idx,"stories"]=obj_stories
        df_summary.at[idx,"state"]=obj_state
        df_summary.at[idx,"climate_zone"]=obj_climate_zone
        df_summary.at[idx,Objects.AREA]=obj_area
        df_summary.at[idx,Objects.FILE]=obj_filename
        df_summary.at[idx,Objects.RESISTANCE]=obj_resistance
        df_summary.at[idx,Objects.CAPACITANCE]=obj_capacitance
        df_summary.at[idx,Objects.INHABITANTS]=obj_inhabitants
        df_summary.at[idx,Objects.LAT]=obj_latitude
        df_summary.at[idx,Objects.LON]=obj_longitude
        df_summary.at[idx,Objects.TEMP_MAX]=obj_temp_max
        df_summary.at[idx,Objects.TEMP_MIN]=obj_temp_min

        df_summary.at[idx,f"{Types.HEATING}_{Columns.DEMAND}_simulated[kWh]"]   = round((df_hvac[f"{Types.HEATING}_{Columns.DEMAND}[W]"].sum()/4)/1000,1)
        df_summary.at[idx,f"{Types.HEATING}_{Columns.DEMAND}_real[kWh]"]        = round((df_hvac_real[f"{Types.HEATING}_{Columns.DEMAND}[W]"].sum()/4)/1000,1)

        df_summary.at[idx,f"{Types.COOLING}_{Columns.DEMAND}_simulated[kWh]"]   = round((df_hvac[f"{Types.COOLING}_{Columns.DEMAND}[W]"].sum()/4)/1000,1)
        df_summary.at[idx,f"{Types.COOLING}_{Columns.DEMAND}_real[kWh]"]        = round((df_hvac_real[f"{Types.COOLING}_{Columns.DEMAND}[W]"].sum()/4)/1000,1)
        
        df_summary.at[idx,f"{Types.HEATING}_{Columns.DEMAND}_simulated[kWh]/{Objects.AREA}"]=   round(((df_hvac[f"{Types.HEATING}_{Columns.DEMAND}[W]"].sum()/4)/1000)/obj_area,1)
        df_summary.at[idx,f"{Types.HEATING}_{Columns.DEMAND}_real[kWh]/{Objects.AREA}"]     =   round((( df_hvac_real[f"{Types.HEATING}_{Columns.DEMAND}[W]"].sum()/4)/1000)/obj_area,1)
        
        df_summary.at[idx,f"{Types.COOLING}_{Columns.DEMAND}_simulated[kWh]/{Objects.AREA}"]=   round(((df_hvac[f"{Types.COOLING}_{Columns.DEMAND}[W]"].sum()/4)/1000)/obj_area,1)
        df_summary.at[idx,f"{Types.COOLING}_{Columns.DEMAND}_real[kWh]/{Objects.AREA}"]     =   round(((df_hvac_real[f"{Types.COOLING}_{Columns.DEMAND}[W]"].sum()/4)/1000)/obj_area,1)

        df_summary.at[idx, "RMSE_heating[W]"] = round(rmse_h, 2)
        df_summary.at[idx, "RMSE_cooling[W]"] = round(rmse_c, 2)
        
    df_summary.to_csv(f"results/hvac_summary_{method}.csv",index=False)
    return df_summary


def get_windows(obj):
    df_windows=pd.read_csv(f"data/validation/tipology/windows_nrel.csv")
    mask = (
        (df_windows["year"].astype(int) == int(obj["year"])) &
        (df_windows[Objects.ID].astype(str) == str(obj[Objects.ID]))
    )
    df_windows = df_windows.loc[mask]   
    return df_windows


def calculate_fit_score(df,method=Columns.OCCUPANCY_GEOMA,name=""):

    df["heating_demand_error"] = round((df["heating_demand_simulated[kWh]/area[m2]"] - df["heating_demand_real[kWh]/area[m2]"]))
    df["cooling_demand_error"] = round((df["cooling_demand_simulated[kWh]/area[m2]"] - df["cooling_demand_real[kWh]/area[m2]"]))

    df["heating_demand_rel_error"] = round(df["heating_demand_error"] / df["heating_demand_real[kWh]"].replace(0, numpy.nan))
    df["cooling_demand_rel_error"] = round(df["cooling_demand_error"] / df["cooling_demand_real[kWh]"].replace(0, numpy.nan))


    df["fit_score"] = df[[
    "heating_demand_rel_error",
    "cooling_demand_rel_error",
    ]].mean(axis=1)

    df = df.sort_values("fit_score", ascending=True)
    
    df_best_fit_score= (
    df.groupby("id", as_index=False)
      .apply(lambda g: g.loc[g["fit_score"].idxmin()])
      .reset_index(drop=True)
    )

    df_best_heating_score= (
    df.groupby("id", as_index=False)
      .apply(lambda g: g.loc[g["heating_demand_rel_error"].idxmin()])
      .reset_index(drop=True)
    )

    df_best_cooling_score=(
    df.groupby("id", as_index=False)
      .apply(lambda g: g.loc[g["cooling_demand_rel_error"].idxmin()])
      .reset_index(drop=True)
    )
    
    df_best_fit_score=df_best_fit_score.sort_values(by="fit_score", ascending=True)
    df_best_fit_score.to_csv(f"results/best_fit_score_{method}_{name}.csv",index=False)
    df_best_heating_score.to_csv(f"results/best_heating_score_{method}_{name}.csv",index=False)
    df_best_cooling_score.to_csv(f"results/best_cooling_score_{method}_{name}.csv",index=False)

    fit_score=df["fit_score"].iloc[0]
    heating_demand_error=df["heating_demand_rel_error"].iloc[0]
    cooling_demand_error=df["cooling_demand_rel_error"].iloc[0]

    print(f"Best fit for this iteration: {fit_score}")
    return df_best_fit_score

def detect_upper_limit(s: pd.Series) -> float:
    # Ignore empty or all-zero series
    if s.empty or s.max() == 0:
        return numpy.inf

    max_val = s.max()
    mask = (s == max_val).astype(int)

    # Detect if any two consecutive readings are at the max value
    if ((mask.shift(1) == 1) & (mask == 1)).any():
        return max_val
    else:
        return numpy.inf
    

def melt_best_results(df_best:pd.DataFrame):

    demand_folder="data/validation/demand"
    hvac_folder=f"data/validation/hvac"
    dfs=[]
    for idx,obj in df_best.iterrows():

        obj_id =str(obj[Objects.ID])
        obj_year=obj["year"]
        obj_climate_zone=obj["climate_zone"]
        obj_lat=obj[Objects.LAT]
        obj_lon=obj[Objects.LON]
        obj_area=obj[Objects.AREA]
        obj_inhabitants=obj[Objects.INHABITANTS]

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
        df[Objects.LAT]=obj_lat
        df[Objects.LON]=obj_lon
        df[Objects.AREA]=obj_area
        df[Objects.INHABITANTS]=obj_inhabitants

        df=df.set_index(Columns.DATETIME)

        dfs.append(df)

    output_df=pd.concat(dfs, axis=0)
    output_df=output_df.reset_index()
    output_df.to_csv("all_hvac_objects.csv",index=False)
    return output_df
