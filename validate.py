
import logging
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Import the new TimeSeriesGenerator
from entise.constants import Types
from entise.constants.columns import Columns
from entise.core.generator import TimeSeriesGenerator
from entise.methods.auxiliary.internal.strategies import InternalOccupancy


cwd="."
objects = pd.read_csv(os.path.join(cwd, "objects_validation.csv"))
demand_folder="data/validation/demand"
data = {}

#DERIVE OCCUPANCY SCHEDULE

processed_ids = set()
for idx,obj in objects.iterrows():
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

    gen.add_objects(objects)

    # Generate Occupancy time series
    summary, df = gen.generate(data, workers=1)

    # Print summary
    print("Summary occupancy:")
    print(summary)

    df[int(obj_id)][Types.OCCUPANCY].to_csv(f"data/validation/occupancy/{Columns.OCCUPANCY_GEOMA}/{obj_id}.csv")

    processed_ids.add(obj_id)

#df_internal_gains=InternalOccupancy().generate(obj=objects_filtered,data=df[building_id])