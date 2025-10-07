
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
objects = pd.read_csv(os.path.join(cwd, "objects.csv"))
data = {}
common_data_folder = "common_data"
for file in os.listdir(os.path.join(cwd, common_data_folder)):
    if file.endswith(".csv"):
        name = file.split(".")[0]
        data[name] = pd.read_csv(os.path.join(os.path.join(cwd, common_data_folder, file)), parse_dates=True)

data_folder = "data"
for file in os.listdir(os.path.join(cwd, data_folder)):
    if file.endswith(".csv"):
        name = file.split(".")[0]
        data[name] = pd.read_csv(os.path.join(os.path.join(cwd, data_folder, file)), parse_dates=True)
print("Loaded data keys:", list(data.keys()))

# Instantiate and configure the generator
gen = TimeSeriesGenerator(logging_level=logging.WARNING)

# Define the building ID to process and visualize
building_id = 1  # Select the object you want to simulate

# Filter objects to only include one object (for debugging)
objects_filtered = objects[objects["id"] == building_id]
print(f"Processing only object with ID: {building_id}")
gen.add_objects(objects_filtered)

# Generate time series
summary, df = gen.generate(data, workers=1)

df_internal_gains=InternalOccupancy().generate(obj=objects_filtered,data=df[building_id])

df[building_id]["occupancy"].to_csv(os.path.join(cwd,"results/occupancy_geoma.csv"))
df[building_id]["electricity"].to_csv(os.path.join(cwd,"results/electricity.csv"))

# Print summary
print("Summary occupancy:")
print(summary)


# Create figure and axes
fig, axs = plt.subplots(figsize=(24, 12))

axs.plot(df[building_id]["occupancy"].index, np.log10(df[building_id]["electricity"]["power[W]"]), color='tab:cyan', alpha=0.8)
axs.plot(df[building_id]["occupancy"].index, np.log10(df_internal_gains["gains_internal[W]"]), color='red', alpha=0.8)
axs.set_title(f"GeoMA Occupancy for object {building_id}")
axs.set_xlabel("time ")
axs.set_ylabel("power (W)")
axs.minorticks_on()
axs.grid(which='major', linestyle='-', linewidth='0.5', color='black')
axs.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
# Highlight regions where y is greater than 5
axs.fill_between(df[building_id]["occupancy"].index,np.log10(df[building_id]["electricity"]["power[W]"]), where=df[building_id]["occupancy"]["occupation"] ==1, color='green', alpha=0.3)
axs.legend(["Electricity demand","internal gains", "Occupied"], loc='upper left',fontsize='small')

plt.tight_layout()
plt.savefig(os.path.join(cwd,"results/occupancy.png"))

building_data = df[building_id][Types.HVAC]

# Figure 1: Indoor & Outdoor Temperature and Solar Radiation (GHI)
fig, ax1 = plt.subplots(figsize=(15, 6))

# Solar radiation plot (GHI) with separate y-axis
ax2 = ax1.twinx()
ax2.plot(
    building_data.index, data["weather"][Columns.SOLAR_GHI],
    label="Solar Radiation (GHI)", color="tab:orange", alpha=0.3
)
ax2.set_ylabel("Solar Radiation (W/m²)")
ax2.legend(loc="upper right")
ax2.set_ylim(-250, 1000)

# Temperature plot
ax1.plot(building_data.index, data["weather"][f"{Columns.TEMP_AIR}@2m"], label="Outdoor Temp", color="tab:cyan", alpha=0.7)
ax1.plot(building_data.index, building_data[Columns.TEMP_IN], label="Indoor Temp", color="tab:blue")
ax1.set_ylabel("Temperature (°C)")
ax1.set_title(f"Building ID: {building_id} - Temperatures and Solar Radiation")
ax1.legend(loc="upper left")
ax1.grid(True)
ax1.set_ylim(-10, 40)

ax1.set_zorder(ax2.get_zorder() + 1)
ax1.patch.set_visible(False)  # required to see through ax1 to ax2
plt.tight_layout()
plt.savefig(os.path.join(cwd,"results/temperature_and_radiation.png"))

# Figure 2: Heating and Cooling Loads
fig, ax = plt.subplots(figsize=(14, 5))
heating_MWh = summary.loc[building_id, "heating:demand[Wh]"] / 1e6
cooling_MWh = summary.loc[building_id, "cooling:demand[Wh]"] / 1e6
(line1,) = ax.plot(
    building_data.index,
    building_data["heating:load[W]"],
    label=f"Heating: {heating_MWh:.1f} MWh",
    color="tab:red",
    alpha=0.8,
)
(line2,) = ax.plot(
    building_data.index,
    building_data["cooling:load[W]"],
    label=f"Cooling: {cooling_MWh:.1f} MWh",
    color="tab:cyan",
    alpha=0.8,
)
# Create the combined legend in the upper left corner
ax.set_ylabel("Load (W)")
ax.set_title(f"Building ID: {building_id} - HVAC Loads")
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.show()

# Figure 3: Outdoor Temperature with Heating & Cooling Loads
fig, ax1 = plt.subplots(figsize=(15, 6))

# Plot outdoor temperature on left y-axis
air_temp = data["weather"][f"{Columns.TEMP_AIR}@2m"]
ax1.plot(building_data.index, air_temp
         , label="Outdoor Temp", color="tab:cyan", alpha=0.7)

ax1.set_ylabel("Outdoor Temp (°C)")
ax1.set_ylim(air_temp.min().round() - 2, air_temp.max().round() + 2)

# Create second y-axis for loads
ax2 = ax1.twinx()
ax2.plot(building_data.index, building_data["heating:load[W]"], label="Heating Load", color="tab:red", alpha=0.8)
ax2.plot(building_data.index, building_data["cooling:load[W]"], label="Cooling Load", color="tab:blue", alpha=0.8)
ax2.set_ylabel("HVAC Load (W)")
ax2.set_ylim(
    min(building_data["heating:load[W]"].min(), building_data["cooling:load[W]"].min()) * 1.1,
    max(building_data["heating:load[W]"].max(), building_data["cooling:load[W]"].max()) * 1.1,
)

# Combine legends from both axes
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

ax1.set_title(f"Building ID: {building_id} - Outdoor Temp & HVAC Loads")
ax1.grid(True)
fig.tight_layout()
plt.tight_layout()
plt.savefig(os.path.join(cwd,"results/hvac_loads.png"))