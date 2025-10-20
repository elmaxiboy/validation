
import logging
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Import the new TimeSeriesGenerator
from entise.constants import Types
from entise.constants.columns import Columns
from entise.constants.objects import Objects
from entise.core.generator import TimeSeriesGenerator
from entise.methods.auxiliary.internal.strategies import InternalOccupancy


cwd="."


#DERIVE OCCUPANCY SCHEDULE
def derive_occupancy_schedule():

    objects = pd.read_csv(os.path.join(cwd, "objects_validation.csv"))
    demand_folder="data/validation/demand"
    

    #Discard year,resistance and capacitance
    objects=objects.drop(columns=["year",Objects.RESISTANCE,Objects.CAPACITANCE])
    objects=objects.drop_duplicates()

    processed_ids = set()

    for idx,obj in objects.iterrows():
        data = {}
        obj_id = str(obj["id"])

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
        obj[Types.OCCUPANCY]="GeoMA"
        obj[Objects.APPLY_NIGHT_SCHEDULE]=True
        obj[Objects.ID]=Columns.OCCUPANCY_GEOMA
        gen.add_objects(obj.to_dict())
        #PHT
        obj[Types.OCCUPANCY]="PHT"
        obj[Objects.APPLY_NIGHT_SCHEDULE]=True
        obj[Objects.ID]=Columns.OCCUPANCY_PHT
        gen.add_objects(obj.to_dict())

        # Generate Occupancy time series
        summary, df = gen.generate(data, workers=1)

        # Print summary
        print("Summary occupancy:")
        print(summary)

        df[Columns.OCCUPANCY_GEOMA][Types.OCCUPANCY].to_csv(f"data/validation/occupancy/{Columns.OCCUPANCY_GEOMA}/{obj_id}.csv")
        df[Columns.OCCUPANCY_PHT][Types.OCCUPANCY].to_csv(f"data/validation/occupancy/{Columns.OCCUPANCY_PHT}/{obj_id}.csv")

        processed_ids.add(obj_id)

def derive_internal_gains():

    objects = pd.read_csv(os.path.join(cwd, "objects_validation.csv"))
    occupancy_folder="data/validation/occupancy"
    
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

def derive_hvac():
    objects = pd.read_csv(os.path.join(cwd, "objects_validation.csv"))
    internal_gains_folder="data/validation/internal_gains"
    weather_folder="data/validation/weather"
    hvac_folder="data/validation/hvac"
    objects=objects[[Objects.ID,"year",Objects.INHABITANTS,Objects.RESISTANCE,Objects.CAPACITANCE,Objects.TEMP_MIN,Objects.TEMP_MAX,Objects.FILE]]
    objects[Types.HVAC]="1r1c"
    df_summary=pd.DataFrame(columns=[
    "id",
    "year",
    "method",    
    "heating:demand[Wh]",
    "heating:load_max[W]",
    "cooling:demand[Wh]",
    "cooling:load_max[W]"
    ])

    for idx,obj in objects.iterrows():
        try:
            gen = TimeSeriesGenerator(logging_level=logging.WARNING)
            data = {}
            obj_id = str(obj["id"])
            obj_year= obj["year"]

            df_weather=pd.read_csv(f"{weather_folder}/{obj["filename"]}")
            df_weather["datetime"]=pd.to_datetime(df_weather["timestamp"], unit="s")
            df_weather=df_weather[[Columns.DATETIME,Columns.TEMP_AIR]]
            df_weather[Columns.TEMP_AIR]=pd.to_numeric(df_weather[Columns.TEMP_AIR])-273.15
            data[Objects.WEATHER]=df_weather

            #GEOMA
            df_internal_gains=pd.read_csv(f"{internal_gains_folder}/{Columns.OCCUPANCY_GEOMA}/{obj_id}.csv")
            data[Objects.GAINS_INTERNAL]=df_internal_gains
            gen.add_objects(obj.to_dict())

            # Generate HVAC time series
            summary, df = gen.generate(data, workers=1)

            # Print summary
            print(f"Summary occupancy GEOMA for bldg: {obj_id}, year: {obj_year}:")
            print(summary)

            df_summary.loc[len(df_summary)] = [
            obj_id,
            obj_year,
            Columns.OCCUPANCY_GEOMA,
            summary["heating:demand[Wh]"],
            summary["heating:load_max[W]"],
            summary["cooling:demand[Wh]"],
            summary["cooling:load_max[W]"]
            ]

            df[int(obj_id)][Types.HVAC].to_csv(f"{hvac_folder}/{Columns.OCCUPANCY_GEOMA}/{obj_id}_{obj_year}.csv")

            
            #PHT
            gen = TimeSeriesGenerator(logging_level=logging.WARNING)
            data = {}
            data[Objects.WEATHER]=df_weather

            df_internal_gains=pd.read_csv(f"{internal_gains_folder}/{Columns.OCCUPANCY_PHT}/{obj_id}.csv")
            data[Objects.GAINS_INTERNAL]=df_internal_gains
            gen.add_objects(obj.to_dict())

            # Generate HVAC time series
            summary, df = gen.generate(data, workers=1)

            # Print summary
            print(f"Summary occupancy PHT for bldg: {obj_id}, year: {obj_year}:")
            print(summary)

            df[int(obj_id)][Types.HVAC].to_csv(f"{hvac_folder}/{Columns.OCCUPANCY_PHT}/{obj_id}_{obj_year}.csv")

            df_summary.loc[len(df_summary)] = [
            obj_id,
            obj_year,
            Columns.OCCUPANCY_PHT,
            summary["heating:demand[Wh]"],
            summary["heating:load_max[W]"],
            summary["cooling:demand[Wh]"],
            summary["cooling:load_max[W]"]
            ]
        except Exception as e:
            logging.warning(f"Building: {obj_id}, year: {obj_year}: {e}")
    
    df_summary.to_csv("hvac_summary.csv",index=False)

derive_hvac()