
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
    occupancy_folder="data/validation/occupancy"
    objects=objects[[Objects.ID,"year",Objects.INHABITANTS,Objects.RESISTANCE,Objects.CAPACITANCE,Objects.TEMP_MIN,Objects.TEMP_MAX]]

    for idx,obj in objects.iterrows():
        data = {}
        obj_id = str(obj["id"])
        weather