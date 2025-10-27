
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
from entise.methods.hvac.R1C1 import calculate_timeseries 
from entise.methods.auxiliary.solar.strategies import SolarGainsPVLib
import pvlib

cwd="."
solar_gains_folder="data/validation/solar_gains"
demand_folder="data/validation/demand"
occupancy_folder=f"data/validation/{Types.OCCUPANCY}"
hvac_folder=f"data/validation/hvac"



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

    objects = pd.read_csv(os.path.join(cwd, "data/validation/objects_entise.csv"))
    solar_gains_generator = SolarGainsPVLib()
    
    
    for idx,obj in objects.iterrows():
        data={}
        
        df_weather=get_weather_timeseries(obj)
        df_weather=calculate_dni(df_weather,obj)
        df_weather.set_index(Columns.DATETIME,inplace=True)
        data[Objects.WEATHER]=df_weather
        
        df_windows=get_windows(obj)
        data[Objects.WINDOWS]=df_windows
        
        df_solar_gains=solar_gains_generator.generate(obj=obj,data=data)
        df_solar_gains=df_solar_gains.reset_index()
        df_solar_gains.to_csv(f"{solar_gains_folder}/{obj[Objects.ID]}.csv",index=False)

        

def derive_hvac(method:str=Columns.OCCUPANCY_GEOMA):
    print(f"HVAC calculation: {method}")
    
    internal_gains_folder="data/validation/internal_gains"
    weather_folder="data/validation/weather"
    hvac_folder="data/validation/hvac"
    objects = pd.read_csv(os.path.join(cwd, "data/validation/objects_entise.csv"))
    objects=objects[[Objects.ID,"year",
                     Objects.INHABITANTS,
                     Objects.RESISTANCE,
                     Objects.CAPACITANCE,
                     Objects.TEMP_MIN,
                     Objects.TEMP_MAX,
                     Objects.FILE,
                     Objects.AREA,
                     "stories"]]
    
    objects[Types.HVAC]             =   "1r1c"
    objects[Objects.GAINS_INTERNAL] =   Objects.GAINS_INTERNAL
    objects[Objects.GAINS_INTERNAL_COL] = Objects.GAINS_INTERNAL
    objects[Objects.VENTILATION]    =   Objects.VENTILATION
    objects[Objects.VENTILATION_COL] = "typical"
    objects[Objects.WINDOWS]    =   Objects.WINDOWS

    #Initialize missing parameters
    objects[Objects.TEMP_INIT] = objects[[Objects.TEMP_MAX, Objects.TEMP_MIN]].mean(axis=1)
    objects[Objects.ACTIVE_COOLING] = True
    objects[Objects.ACTIVE_HEATING] = True
    objects[Objects.POWER_COOLING]  = 50000
    objects[Objects.POWER_HEATING]  = 50000  
    
    df_summary=pd.DataFrame(columns=[
    Objects.ID,
    "year",
    Objects.FILE,
    Objects.INHABITANTS,
    Objects.RESISTANCE,
    Objects.CAPACITANCE,
    Objects.AREA,
    "stories",
    "method",
    f"{Types.HEATING}_{Columns.DEMAND}[Wh]",
    f"{Types.HEATING}_{Columns.LOAD}_max[W]",
    f"{Types.COOLING}_{Columns.DEMAND}[Wh]",
    f"{Types.COOLING}_{Columns.LOAD}_max[W]",
    ])

    full_index = pd.date_range(start="2018-01-01 00:00:00", end="2018-12-31 23:45:00", freq="15min",name=Columns.DATETIME)

    for idx,obj in objects.iterrows():

            data = {}
            obj_id =str(obj[Objects.ID])
            obj_year=obj["year"]
            obj_stories=obj["stories"]
            obj_area=obj[Objects.AREA]
            obj_filename=obj[Objects.FILE]
            obj_inhabitants=obj[Objects.INHABITANTS]
            obj_resistance=obj[Objects.RESISTANCE]
            obj_capacitance=obj[Objects.CAPACITANCE]

            print(f"Processing ID:{obj_id}, year:{obj_year}")

            #Add Weather Data
            df_weather=get_weather_timeseries(obj)
            data[Objects.WEATHER]=df_weather

            #Add Solar Gains Data
            df_solar_gains=pd.read_csv(f"{solar_gains_folder}/{obj_id}.csv")
            df_solar_gains=set_datetime_index(df_solar_gains)
            data[Objects.GAINS_SOLAR]=df_solar_gains

            #Add Internal Gains Data
            df_internal_gains=pd.read_csv(f"{internal_gains_folder}/{method}/{obj_id}.csv")
            df_internal_gains=set_datetime_index(df_internal_gains)
            data[Objects.GAINS_INTERNAL]=df_internal_gains
            
            #Add Windows Data
            df_windows=get_windows(obj)
            data[Objects.WINDOWS]=df_windows

            #Add Ventilation
            df_ventilation=pd.read_csv(f"data/validation/tipology/ventilation.csv")
            df_ventilation[Columns.DATETIME] = pd.to_datetime(df_ventilation[Columns.DATETIME],
                                                              utc=True,
                                                              errors="coerce").dt.tz_localize(None)
            df_ventilation = df_ventilation.set_index(Columns.DATETIME).reindex(full_index)
            df_ventilation = df_ventilation.interpolate(method="time")
            df_ventilation = df_ventilation[["typical"]]
            data[Objects.VENTILATION]=df_ventilation

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
            df_hvac.to_csv(f"{hvac_folder}/{method}/{obj_id}_{obj_year}.csv",index=False)

def summarize_hvac(method:str=Columns.OCCUPANCY_GEOMA):

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
    "method",
    f"{Types.HEATING}_{Columns.DEMAND}[kWh]",
    f"{Types.HEATING}_{Columns.LOAD}_max[kW]",
    f"{Types.COOLING}_{Columns.DEMAND}[kWh]",
    f"{Types.COOLING}_{Columns.LOAD}_max[kW]",
    f"KWh/{Objects.AREA}"
    ])

    for idx,obj in objects.iterrows():
        

        obj_id =str(obj[Objects.ID])
        obj_year=obj["year"]

        print(f"Processing ID:{obj_id}, year:{obj_year}")
        
        obj_stories=obj["stories"]
        obj_state=obj["state"]
        obj_climate_zone=obj["climate_zone"]
        obj_area=obj[Objects.AREA]
        obj_filename=obj[Objects.FILE]
        obj_inhabitants=obj[Objects.INHABITANTS]
        obj_resistance=obj[Objects.RESISTANCE]
        obj_capacitance=obj[Objects.CAPACITANCE]

        df_hvac=pd.read_csv(f"{hvac_folder}/{method}/{obj_id}_{obj_year}.csv")

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
        df_summary.at[idx,f"{Types.HEATING}_{Columns.DEMAND}[kWh]"]  = (df_hvac[f"{Types.HEATING}_{Columns.DEMAND}[W]"].sum()/4)/1000
        df_summary.at[idx,f"{Types.HEATING}_{Columns.LOAD}_max[kW]"] = df_hvac[f"{Types.HEATING}_{Columns.DEMAND}[W]"].max()/1000
        df_summary.at[idx,f"{Types.COOLING}_{Columns.DEMAND}[kWh]"]  = (df_hvac[f"{Types.COOLING}_{Columns.DEMAND}[W]"].sum()/4)/1000
        df_summary.at[idx,f"{Types.COOLING}_{Columns.LOAD}_max[kW]"] = df_hvac[f"{Types.COOLING}_{Columns.DEMAND}[W]"].max()/1000
        df_summary.at[idx,f"{Types.HEATING}_{Columns.DEMAND}[kWh]/{Objects.AREA}"]=((df_hvac[f"{Types.HEATING}_{Columns.DEMAND}[W]"].sum()/4)/1000)/obj_area
        df_summary.at[idx,f"{Types.COOLING}_{Columns.DEMAND}[kWh]/{Objects.AREA}"]=((df_hvac[f"{Types.COOLING}_{Columns.DEMAND}[W]"].sum()/4)/1000)/obj_area

    df_summary.to_csv(f"results/hvac_summary_{method}.csv",index=False)


def get_weather_timeseries(obj):
    weather_folder="data/validation/weather"
    df_weather=pd.read_csv(f"{weather_folder}/{obj["filename"]}")
    df_weather["datetime"]=pd.to_datetime(df_weather["timestamp"], unit="s")
    df_weather[Columns.SOLAR_GHI]=df_weather["ghi"]
    df_weather[Columns.SOLAR_DHI]=df_weather["dhi"]
    df_weather=df_weather[[Columns.DATETIME,Columns.TEMP_AIR,Columns.SOLAR_GHI,Columns.SOLAR_DHI]]
    df_weather[Columns.TEMP_AIR]=pd.to_numeric(df_weather[Columns.TEMP_AIR])-273.15
    return df_weather

def get_windows(obj):
    df_windows=pd.read_csv(f"data/validation/tipology/windows.csv")
    mask = (
        (df_windows["year"].astype(int) == int(obj["year"])) &
        (df_windows[Objects.ID].astype(str) == str(obj[Objects.ID]))
    )
    df_windows = df_windows.loc[mask]   
    return df_windows

def calculate_dni(df_weather,object):
    solpos = pvlib.solarposition.get_solarposition(
    time=df_weather[Columns.DATETIME],
    latitude=object[Objects.LAT],
    longitude=object[Objects.LON]
    )

    # Calculate DNI safely
    cos_zenith = numpy.cos(numpy.radians(solpos['zenith'].to_numpy()))
    dni = (df_weather[Columns.SOLAR_GHI].to_numpy() - df_weather[Columns.SOLAR_DHI].to_numpy()) / cos_zenith

    # Replace impossible values (e.g., sun below horizon)
    dni[cos_zenith <= 0] = 0
    dni[dni < 0] = 0

    df_weather[Columns.SOLAR_DNI] = dni

    return df_weather

def set_datetime_index(df):
    df[Columns.DATETIME]=pd.to_datetime(df[Columns.DATETIME])
    df.set_index(Columns.DATETIME,inplace=True)
    return df