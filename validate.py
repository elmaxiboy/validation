
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
data = {}
#common_data_folder = "common_data"
#for file in os.listdir(os.path.join(cwd, common_data_folder)):
#    if file.endswith(".csv"):
#        name = file.split(".")[0]
#        data[name] = pd.read_csv(os.path.join(os.path.join(cwd, common_data_folder, file)), parse_dates=True)

data_folder = "data"
for file in os.listdir(os.path.join(cwd, data_folder)):
    if file.endswith(".csv"):
        name = file.split(".")[0]
        data[name] = pd.read_csv(os.path.join(os.path.join(cwd, data_folder, file)), parse_dates=True)
print("Loaded data keys:", list(data.keys()))

# Instantiate and configure the generator
gen = TimeSeriesGenerator(logging_level=logging.WARNING)

# Define the building ID to process and visualize

# Filter objects to only include one object (for debugging)
#print(f"Processing only object with ID: {building_id}")

gen.add_objects(objects)

# Generate time series
summary, df = gen.generate(data, workers=1)



#df_internal_gains=InternalOccupancy().generate(obj=objects_filtered,data=df[building_id])


# Print summary
print("Summary occupancy:")
print(summary)