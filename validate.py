
import logging
import os
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

cwd="."
solar_gains_folder="data/validation/solar_gains"
demand_folder="data/validation/demand"
occupancy_folder=f"data/validation/{Types.OCCUPANCY}"
hvac_folder=f"data/validation/hvac"
weather_folder="data/validation/weather"
real_demand_folder="C:/Users/escobarm/Desktop/thesis/validation_data/counties"



#DERIVE OCCUPANCY SCHEDULE
def derive_occupancy_schedule():

    objects = pd.read_csv(os.path.join(cwd, "data/validation/objects_entise.csv"))

    df_summary=pd.DataFrame(columns=[Objects.ID,"method",f"average_{Types.OCCUPANCY}"])
    
    #Discard year,resistance and capacitance
    objects=objects.drop(columns=["year",Objects.RESISTANCE,Objects.CAPACITANCE])
    objects=objects.drop_duplicates()

    processed_ids = set()

    for idx,obj in objects.iterrows():
        data = {}
        obj_id = str(obj[Objects.ID])

        if obj_id in processed_ids:
            continue

        matching_files = [f for f in os.listdir(os.path.join(cwd, demand_folder)) if f.startswith(obj_id)]

        if not matching_files:
            continue
        
        file = matching_files[0]  # only one match
        
        df_demand = pd.read_csv(os.path.join(cwd, demand_folder, file), parse_dates=True)

        data[Columns.DEMAND]=df_demand

        print("Loaded data keys:", list(data.keys()))

        # Instantiate and configure the generator
        gen = TimeSeriesGenerator(logging_level=logging.WARNING)

        #GEOMA
        obj[Types.OCCUPANCY]=Columns.OCCUPANCY_GEOMA
        obj[Objects.APPLY_NIGHT_SCHEDULE]=True
        obj[Objects.ID]=Columns.OCCUPANCY_GEOMA
        gen.add_objects(obj.to_dict())

        #PHT
        obj[Types.OCCUPANCY]=Columns.OCCUPANCY_PHT
        obj[Objects.APPLY_NIGHT_SCHEDULE]=True
        obj[Objects.ID]=Columns.OCCUPANCY_PHT
        gen.add_objects(obj.to_dict())

        # Generate Occupancy time series
        summary, df = gen.generate(data, workers=1)

        # Print summary
        print(f"Summary occupancy: {obj_id}")
        print(summary)

        for idx,row in summary.iterrows():

            df_summary.loc[len(df_summary)] = {
                Objects.ID:obj_id,
                "method":idx,
                f"average_{Types.OCCUPANCY}":row["average_occupation"]
            }

        df[Columns.OCCUPANCY_GEOMA][Types.OCCUPANCY].to_csv(f"data/validation/{Types.OCCUPANCY}/{Columns.OCCUPANCY_GEOMA}/{obj_id}.csv")
        df[Columns.OCCUPANCY_PHT][Types.OCCUPANCY].to_csv(f"data/validation/{Types.OCCUPANCY}/{Columns.OCCUPANCY_PHT}/{obj_id}.csv")
    
        processed_ids.add(obj_id)
    
    df_summary.to_csv(f"data/validation/{Types.OCCUPANCY}/summary_{Types.OCCUPANCY}.csv",index=False)


def derive_ventilation(obj,data):

    full_index = pd.date_range(start="2018-01-01 00:00:00", end="2018-12-31 23:45:00", freq="15min",name=Columns.DATETIME)

    if os.path.exists(f"data/validation/ventilation/{obj[Columns.ID]}.csv"):
        df_ventilation=pd.read_csv(f"data/validation/ventilation/{obj[Columns.ID]}.csv")
        df_ventilation[Columns.DATETIME]=pd.to_datetime(df_ventilation[Columns.DATETIME],
                                                          utc=True,
                                                          errors="coerce").dt.tz_localize(None)
        df_ventilation=df_ventilation.set_index(Columns.DATETIME)


    else:

        ventilation_generator = VentilationTimeSeries()

        df_air_exchange=pd.read_csv(f"data/validation/tipology/ventilation.csv")

        df_air_exchange[Columns.DATETIME] = pd.to_datetime(df_air_exchange[Columns.DATETIME],
                                                          utc=True,
                                                          errors="coerce").dt.tz_localize(None)

        df_air_exchange = df_air_exchange.set_index(Columns.DATETIME).reindex(full_index)
        df_air_exchange = df_air_exchange.bfill()
        df_air_exchange[obj[Objects.VENTILATION_COL]]=df_air_exchange[obj[Objects.VENTILATION_COL]]/4
        data[Objects.VENTILATION]=df_air_exchange
        df_ventilation=ventilation_generator.generate(obj,data)
        df_ventilation=df_ventilation.ffill()
        df_ventilation.to_csv(f"data/validation/ventilation/{obj[Columns.ID]}.csv")

    data[Objects.VENTILATION]=df_ventilation
    
    
    return obj,data

def derive_internal_gains():

    objects = pd.read_csv(os.path.join(cwd, "data/validation/objects_entise.csv"))

    objects=objects.drop(columns=["year",Objects.RESISTANCE,Objects.CAPACITANCE])
    objects=objects.drop_duplicates()

    objects[Objects.GAINS_INTERNAL_PER_PERSON]=65 #FISCHER NUMBER

    for idx,obj in objects.iterrows():
        data = {}
        obj_id = str(obj["id"])
        inhabitants=obj[Objects.INHABITANTS]

        print(f"Calculating internal gains for building:{obj_id}. {inhabitants} inhabitants.")

        #GEOMA-DERIVED OCCUPANCY
        data[Types.OCCUPANCY]=pd.read_csv(f"{occupancy_folder}/{Columns.OCCUPANCY_GEOMA}/{obj_id}.csv")
        data[Types.OCCUPANCY].set_index(Columns.DATETIME,inplace=True)

        df=InternalOccupancy().generate(obj=obj.to_dict(),data=data)

        df.to_csv(f"data/validation/internal_gains/{Columns.OCCUPANCY_GEOMA}/{obj_id}.csv")

        #PHT-DERIVED OCCUPANCY
        data[Types.OCCUPANCY]=pd.read_csv(f"{occupancy_folder}/{Columns.OCCUPANCY_PHT}/{obj_id}.csv")
        data[Types.OCCUPANCY].set_index(Columns.DATETIME,inplace=True)

        df=InternalOccupancy().generate(obj=obj.to_dict(),data=data)

        df.to_csv(f"data/validation/internal_gains/{Columns.OCCUPANCY_PHT}/{obj_id}.csv")

def derive_solar_gains():
    print(f"Solar Gains")
    objects = pd.read_csv(os.path.join(cwd, "data/validation/objects_entise.csv"))
    solar_gains_generator = SolarGainsPVLib()
    
    for idx,obj in objects.iterrows():
        data={}
        
        print(f"Processing ID:{obj[Columns.ID]}, year:{obj["year"]}")
        df_weather=get_weather_timeseries(obj)
        df_weather.set_index(Columns.DATETIME,inplace=True)
        
        data[Objects.WEATHER]=df_weather
        
        df_windows=get_windows(obj)

        #REDUCED AREA AND SHADING BY 50%
        df_windows[Objects.AREA]=df_windows[Objects.AREA]*0.5
        df_windows[Columns.SHADING]=df_windows[Columns.SHADING]
        data[Objects.WINDOWS]=df_windows
        
        df_solar_gains=solar_gains_generator.generate(obj=obj,data=data)
        df_solar_gains=df_solar_gains.reset_index()
        df_solar_gains.to_csv(f"{solar_gains_folder}/{obj[Objects.ID]}_{obj["year"]}.csv",index=False)
        

def derive_hvac(method:str=Columns.OCCUPANCY_GEOMA):
    print(f"HVAC calculation: {method}")
    
    internal_gains_folder="data/validation/internal_gains"
    weather_folder="data/validation/weather"
    hvac_folder="data/validation/hvac"
    real_demand_folder="data/validation/demand"

    objects = pd.read_csv(os.path.join(cwd, "data/validation/objects_entise.csv"))
    objects=objects[[Objects.ID,"year",
                     Objects.INHABITANTS,
                     Objects.RESISTANCE,
                     Objects.CAPACITANCE,
                     Objects.TEMP_MIN,
                     Objects.TEMP_MAX,
                     Objects.FILE,
                     Objects.AREA,
                     "stories",
                     "climate_zone"]]
    
    objects[Types.HVAC]             =   "1r1c"
    objects[Objects.GAINS_INTERNAL] =   Objects.GAINS_INTERNAL
    objects[Objects.GAINS_INTERNAL_COL] = Objects.GAINS_INTERNAL
    objects[Objects.VENTILATION]    =   Objects.VENTILATION
    objects[Objects.VENTILATION_COL] = "optimal"
    objects[Objects.WINDOWS]    =   Objects.WINDOWS

    #Initialize missing parameters
    objects[Objects.ACTIVE_COOLING] = True
    objects[Objects.ACTIVE_HEATING] = True
    objects[Objects.POWER_HEATING]  = numpy.inf

    full_index = pd.date_range(start="2018-01-01 00:00:00", end="2018-12-31 23:45:00", freq="15min",name=Columns.DATETIME)

    for idx,obj in objects.iterrows():

            data = {}
            obj_id =str(obj[Objects.ID])
            obj_year=obj["year"]
            obj[Objects.TEMP_MAX]=obj[Objects.TEMP_MAX]+273.15
            obj[Objects.TEMP_MIN]=obj[Objects.TEMP_MIN]+273.15
            obj[Objects.TEMP_INIT] = (obj[Objects.TEMP_MAX]+obj[Objects.TEMP_MIN])/2

            #Cooling load cannot be higher than the total electricity load of the building?
            try:
                obj[Objects.POWER_COOLING]  = pd.read_csv(f"{real_demand_folder}/{obj_id}.csv")[f"total_{Columns.POWER}"].max()
            
            except:
                obj[Objects.POWER_COOLING]=numpy.inf
                
            print(f"Processing ID:{obj_id}, year:{obj_year}")

            #Add Weather Data
            df_weather=get_weather_timeseries(obj)
            df_weather[Columns.TEMP_AIR]=df_weather[Columns.TEMP_AIR]+273.15
            data[Objects.WEATHER]=df_weather

            #Add Solar Gains Data
            df_solar_gains=pd.read_csv(f"{solar_gains_folder}/{obj_id}_{obj_year}.csv")
            df_solar_gains=set_datetime_index(df_solar_gains)
            data[Objects.GAINS_SOLAR]=df_solar_gains

            #Add Internal Gains Data
            df_internal_gains=pd.read_csv(f"{internal_gains_folder}/{method}/{obj_id}.csv")
            df_internal_gains=set_datetime_index(df_internal_gains)
            data[Objects.GAINS_INTERNAL]=df_internal_gains
            

            #Add Ventilation
            obj,data=derive_ventilation(obj,data)

            # Generate HVAC time series
            temp_in,p_heat,p_cool=calculate_timeseries(obj=obj,data=data)
            df_hvac = pd.DataFrame(
                {
                    Columns.TEMP_IN: temp_in,
                    f"{Types.HEATING}_{Columns.DEMAND}[W]": p_heat,
                    f"{Types.COOLING}_{Columns.DEMAND}[W]": p_cool,
                },
                    index=full_index
                )

            df_hvac.reset_index(inplace=True)
            df_hvac[Columns.TEMP_IN]=df_hvac[Columns.TEMP_IN]-273.15
            df_hvac.to_csv(f"{hvac_folder}/{method}/{obj_id}_{obj_year}.csv",index=False)

def summarize_hvac(method:str=Columns.OCCUPANCY_GEOMA):
    print(f"Summarize HVAC")
    objects = pd.read_csv(os.path.join(cwd, "data/validation/objects_entise.csv"))

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
    "method"
    ])

    for idx,obj in objects.iterrows():
        
        print(f"Processing ID: {obj[Columns.ID]}, year:{obj["year"]}")

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

        df_hvac=pd.read_csv(f"{hvac_folder}/{method}/{obj_id}_{obj_year}.csv")
        df_hvac_real=pd.read_csv(f"{demand_folder}/{obj_id}.csv")

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

        df_summary.at[idx,f"{Types.HEATING}_{Columns.DEMAND}_simulated[kWh]"]   = round((df_hvac[f"{Types.HEATING}_{Columns.DEMAND}[W]"].sum()/4)/1000,1)
        df_summary.at[idx,f"{Types.HEATING}_{Columns.DEMAND}_real[kWh]"]        = round((df_hvac_real[f"{Types.HEATING}_{Columns.DEMAND}[W]"].sum()/4)/1000,1)

        df_summary.at[idx,f"{Types.HEATING}_{Columns.LOAD}_simulated_max[kW]"]  = round(df_hvac[f"{Types.HEATING}_{Columns.DEMAND}[W]"].max()/1000,1)
        df_summary.at[idx,f"{Types.HEATING}_{Columns.LOAD}_real_max[kW]"]       = round(df_hvac_real[f"{Types.HEATING}_{Columns.DEMAND}[W]"].max()/1000,1)

        df_summary.at[idx,f"{Types.COOLING}_{Columns.DEMAND}_simulated[kWh]"]   = round((df_hvac[f"{Types.COOLING}_{Columns.DEMAND}[W]"].sum()/4)/1000,1)
        df_summary.at[idx,f"{Types.COOLING}_{Columns.DEMAND}_real[kWh]"]        = round((df_hvac_real[f"{Types.COOLING}_{Columns.DEMAND}[W]"].sum()/4)/1000,1)
        
        df_summary.at[idx,f"{Types.COOLING}_{Columns.LOAD}_simulated_max[kW]"]  = round(df_hvac[f"{Types.COOLING}_{Columns.DEMAND}[W]"].max()/1000,1)
        df_summary.at[idx,f"{Types.COOLING}_{Columns.LOAD}_real_max[kW]"]       = round(df_hvac_real[f"{Types.COOLING}_{Columns.DEMAND}[W]"].max()/1000,1)
        
        df_summary.at[idx,f"{Types.HEATING}_{Columns.DEMAND}_simulated[kWh]/{Objects.AREA}"]=   round(((df_hvac[f"{Types.HEATING}_{Columns.DEMAND}[W]"].sum()/4)/1000)/obj_area,1)
        df_summary.at[idx,f"{Types.HEATING}_{Columns.DEMAND}_real[kWh]/{Objects.AREA}"]     =   round((( df_hvac_real[f"{Types.HEATING}_{Columns.DEMAND}[W]"].sum()/4)/1000)/obj_area,1)
        
        df_summary.at[idx,f"{Types.COOLING}_{Columns.DEMAND}_simulated[kWh]/{Objects.AREA}"]=   round(((df_hvac[f"{Types.COOLING}_{Columns.DEMAND}[W]"].sum()/4)/1000)/obj_area,1)
        df_summary.at[idx,f"{Types.COOLING}_{Columns.DEMAND}_real[kWh]/{Objects.AREA}"]     =   round(((df_hvac_real[f"{Types.COOLING}_{Columns.DEMAND}[W]"].sum()/4)/1000)/obj_area,1)

        
    df_summary.to_csv(f"results/hvac_summary_{method}.csv",index=False)


def get_weather_timeseries(obj):
    weather_folder="data/validation/weather"
    df_weather=pd.read_csv(f"{weather_folder}/cleaned/{obj["climate_zone"]}.csv")
    df_weather[Columns.DATETIME]=pd.to_datetime(df_weather[Columns.DATETIME])
    return df_weather

def format_weather_timeseries():
    raw_weather_folder=os.path.join(weather_folder,"raw")
    files=os.listdir(raw_weather_folder)

    for file in files:
        df=df=pd.read_csv(f"{raw_weather_folder}/{file}")
        df.columns = df.columns.str.strip()
        df=df.rename(columns={"date_time":                          Columns.DATETIME,
                           "Dry Bulb Temperature [°C]":             Columns.TEMP_AIR,
                           "Global Horizontal Radiation [W/m2]":    Columns.SOLAR_GHI,
                           "Direct Normal Radiation [W/m2]":        Columns.SOLAR_DNI,
                           "Diffuse Horizontal Radiation [W/m2]":   Columns.SOLAR_DHI})
        df=df[[Columns.DATETIME,
                Columns.TEMP_AIR,
                Columns.SOLAR_GHI,
                Columns.SOLAR_DNI,
                Columns.SOLAR_DHI]]
        
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

        df=df.set_index(Columns.DATETIME)
        df=df.resample("15min").interpolate('linear')
        df=df.reset_index()
        df.to_csv(f"{weather_folder}/cleaned/{file}",index=False)

def get_windows(obj):
    df_windows=pd.read_csv(f"data/validation/tipology/windows.csv")
    mask = (
        (df_windows["year"].astype(int) == int(obj["year"])) &
        (df_windows[Objects.ID].astype(str) == str(obj[Objects.ID]))
    )
    df_windows = df_windows.loc[mask]   
    return df_windows


def calculate_dni(df_weather: pd.DataFrame, object,method):

    solpos = pvlib.solarposition.get_solarposition(
        time=df_weather[Columns.DATETIME],
        latitude=object[Objects.LAT],
        longitude=object[Objects.LON]
    )
    #solpos = solpos.reindex(df_weather.index)
    zenith=solpos["zenith"]

    location=pvlib.location.Location(latitude=object[Objects.LAT],longitude=object[Objects.LON])

    df_weather[Columns.DATETIME] = pd.to_datetime(df_weather[Columns.DATETIME])
    df_weather.set_index(Columns.DATETIME, inplace=True)

    df_clearsky=location.get_clearsky(times=df_weather.index)

    df_clearsky = df_clearsky.reindex(df_weather.index)
    dni_clear = df_clearsky["dni"]

    dhi=df_weather[Columns.SOLAR_DHI]
    ghi=df_weather[Columns.SOLAR_GHI]
    
    match method:
        case "zeros":        
            df_weather[Columns.SOLAR_DNI]=0.0
        
        case "pvlib":
            ghi=ghi.align(dhi, join="inner")[0]
            dhi=dhi.align(ghi, join="inner")[0]
            zenith=zenith.align(ghi, join="inner")[0]
            dni_clear=dni_clear.reindex_like(ghi)
            dni = pvlib.irradiance.dni(
                ghi=ghi,
                dhi=dhi,
                zenith=zenith,
                dni_clear=dni_clear,
                clearsky_tolerance=1.1
            )
            dni = dni.clip(lower=0).fillna(0.0)
            df_weather[Columns.SOLAR_DNI]=dni

        case "erbs":
            df=pvlib.irradiance.erbs(ghi,zenith,df_weather.index)
            df_weather[Columns.SOLAR_DNI]=df["dni"]

        case "clear_sky":
            df_weather[Columns.SOLAR_DNI]=dni_clear

    df_weather=df_weather.reset_index()

    return df_weather

def calculate_dhi(df_weather: pd.DataFrame, object):

    solpos = pvlib.solarposition.get_solarposition(
        time=df_weather[Columns.DATETIME],
        latitude=object[Objects.LAT],
        longitude=object[Objects.LON]
    )

    zenith=solpos["zenith"]
    zenith=zenith.clip(upper=90)
    
    location=pvlib.location.Location(latitude=object[Objects.LAT],longitude=object[Objects.LON])

    df_weather[Columns.DATETIME] = pd.to_datetime(df_weather[Columns.DATETIME])
    df_weather.set_index(Columns.DATETIME, inplace=True)

    df_clearsky=location.get_clearsky(times=df_weather.index)

    df_clearsky = df_clearsky.reindex(df_weather.index)
    dni=df_weather[Columns.SOLAR_DNI]
    ghi=df_weather[Columns.SOLAR_GHI]

    ghi_fraction=numpy.cos(numpy.radians(zenith))

    dhi = ghi - dni*ghi_fraction

    dhi = dhi.clip(lower=0).fillna(0.0)
    
    df_weather[Columns.SOLAR_DHI]=dhi
    df_weather["zenith[°]"]=zenith
    
    df_weather=df_weather.reset_index()
    
    df_weather.to_csv(f"{weather_folder}/cleaned/{object["climate_zone"]}.csv",index=False)
    

    return df_weather



def set_datetime_index(df):
    df[Columns.DATETIME]=pd.to_datetime(df[Columns.DATETIME])
    df.set_index(Columns.DATETIME,inplace=True)
    return df

def get_real_demand_files():

    objects = pd.read_csv(os.path.join(cwd, "data/validation/objects_entise.csv")).drop_duplicates(subset="id", keep="first")
    cz_folders=os.listdir(real_demand_folder)

    for idx,obj in objects.iterrows():
        cz_code=str(obj["filename"]).split("_")[0]
        cz_folder=next((f for f in cz_folders if f.split("_")[0] == cz_code), None)
        real_demand=pd.read_csv(f"{real_demand_folder}/{cz_folder}/households/{obj[Columns.ID]}/{obj[Columns.ID]}_timeseries_adjusted.csv")
        print(f"Processing ID: {obj[Columns.ID]}, year:{obj["year"]}")
        try:
            raw_demand_parquet=pd.read_parquet(f"{real_demand_folder}/{cz_folder}/households/{obj[Columns.ID]}/{obj[Columns.ID]}_timeseries.parquet")
        except Exception as e:
                raw_demand_parquet=pd.read_parquet(f"{real_demand_folder}/{cz_folder}/households/{obj[Columns.ID]}/{obj[Columns.ID]}-10.parquet")
        raw_demand_parquet=raw_demand_parquet[["timestamp",
                                               "out.electricity.total.energy_consumption",
                                               "out.electricity.cooling_fans_pumps.energy_consumption",
                                               "out.electricity.cooling.energy_consumption"]]
        raw_demand_parquet[f"{Types.COOLING}_{Columns.DEMAND}[W]"] = (raw_demand_parquet["out.electricity.cooling_fans_pumps.energy_consumption"]+raw_demand_parquet["out.electricity.cooling.energy_consumption"])*4000
        raw_demand_parquet=raw_demand_parquet.rename(columns={"timestamp":Columns.DATETIME,
                                                              "out.electricity.total.energy_consumption":f"total_{Columns.POWER}"})
        
        #Turn into W from kWh for 15' resolution readings
        raw_demand_parquet[f"total_{Columns.POWER}"]=raw_demand_parquet[f"total_{Columns.POWER}"]*4000
        
        raw_demand_parquet[Columns.DATETIME]=pd.to_datetime(raw_demand_parquet[Columns.DATETIME], errors="coerce")

        raw_demand_parquet=raw_demand_parquet[[Columns.DATETIME,
                                                f"total_{Columns.POWER}",
                                                f"{Types.COOLING}_{Columns.DEMAND}[W]"]]
        
        real_demand=real_demand.rename(columns={"heating":f"{Types.HEATING}_{Columns.DEMAND}[W]",
                               "cooling":f"{Types.COOLING}_{Columns.DEMAND}[W]",
                               "timestamp":Columns.DATETIME,
                               "power":f"{Columns.POWER}"})
        
        real_demand=real_demand[[Columns.DATETIME,
                                 f"{Columns.POWER}",
                                 f"{Types.HEATING}_{Columns.DEMAND}[W]"]]
        
        real_demand[Columns.DATETIME]=pd.to_datetime(real_demand[Columns.DATETIME], errors="coerce")

        real_demand=real_demand.merge(
            raw_demand_parquet,
            how="outer",
            on=Columns.DATETIME
        )

        real_demand[f"total_{Columns.POWER}"]=real_demand[f"total_{Columns.POWER}"].bfill()
        real_demand[f"{Types.COOLING}_{Columns.DEMAND}[W]"]=real_demand[f"{Types.COOLING}_{Columns.DEMAND}[W]"].bfill()

        lowerr_heating,upper_heating = real_demand[f"{Types.HEATING}_{Columns.DEMAND}[W]"].quantile([0.05, 0.999])
        real_demand[f"{Types.HEATING}_{Columns.DEMAND}[W]"] =real_demand[f"{Types.HEATING}_{Columns.DEMAND}[W]"].clip(upper=upper_heating) 
        real_demand.drop(real_demand.tail(1).index,inplace=True)
        real_demand.to_csv(f"{demand_folder}/{obj[Columns.ID]}.csv",index=False)

#get_real_demand_files()

def calculate_fit_score(method=Columns.OCCUPANCY_GEOMA):
    df=pd.read_csv(f"results/hvac_summary_{method}.csv")

    df["heating_demand_error"] = (df["heating_demand_simulated[kWh]"] - df["heating_demand_real[kWh]"]).abs()
    df["heating_load_error"]   = (df["heating_load_simulated_max[kW]"] - df["heating_load_real_max[kW]"]).abs()
    df["cooling_demand_error"] = (df["cooling_demand_simulated[kWh]"] - df["cooling_demand_real[kWh]"]).abs()
    df["cooling_load_error"]   = (df["cooling_load_simulated_max[kW]"] - df["cooling_load_real_max[kW]"]).abs()

    df["heating_demand_rel_error"] = df["heating_demand_error"] / df["heating_demand_real[kWh]"].replace(0, numpy.nan)
    df["heating_load_rel_error"]   = df["heating_load_error"] / df["heating_load_real_max[kW]"].replace(0, numpy.nan)
    df["cooling_demand_rel_error"] = df["cooling_demand_error"] / df["cooling_demand_real[kWh]"].replace(0, numpy.nan)
    df["cooling_load_rel_error"]   = df["cooling_load_error"] / df["cooling_load_real_max[kW]"].replace(0, numpy.nan)

    df["fit_score"] = df[[
    "heating_demand_rel_error",
    "heating_load_rel_error",
    "cooling_demand_rel_error",
    "cooling_load_rel_error"
    ]].mean(axis=1)

    df = df.sort_values("fit_score", ascending=True)

    df.to_csv(f"results/fit_score_{method}.csv",index=False)
