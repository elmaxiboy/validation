import pandas as pd
import os


folder_path = "data/validation/zones"

column_to_filter = "in.geometry_building_type_recs"
values_to_keep = ["Single-Family Detached"]

filtered_dfs=[]

# Loop through all CSV files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path)
        
        filtered_df = df[df[column_to_filter].isin(values_to_keep)]

        # Add a column with the file name
        filtered_df["filename"] = filename
        
        # Store in the list
        filtered_dfs.append(filtered_df)

# Concatenate all filtered DataFrames into one
if filtered_dfs:  # check if list is not empty
    df_combined = pd.concat(filtered_dfs, ignore_index=True)
else:
    df_combined = pd.DataFrame()  # empty DataFrame if no files matched


# Relevant columns for 1R1C HVAC validation
relevant_columns = [

    # ID
    "bldg_id",
    "filename",

    # Building envelope / geometry
    "in.geometry_floor_area",
    "in.sqft",
    "in.geometry_stories",
    "in.geometry_wall_type",
    "in.geometry_wall_exterior_finish",
    "in.geometry_foundation_type",
    "in.geometry_attic_type",
    "in.door_area",
    "in.doors",
    "in.window_areas",
    "in.windows",
    "in.eaves",
    "in.orientation",
    "in.insulation_ceiling",
    "in.insulation_wall",
    "in.insulation_floor",
    "in.insulation_foundation_wall",
    "in.insulation_rim_joist",
    "in.insulation_roof",
    "in.insulation_slab",
    
    # HVAC / thermal systems
    "in.hvac_heating_type_and_fuel",
    "in.hvac_heating_efficiency",
    "in.hvac_cooling_type",
    "in.hvac_cooling_efficiency",
    "in.hvac_has_ducts",
    "in.hvac_secondary_heating_type_and_fuel",
    "in.heating_setpoint",
    "in.cooling_setpoint",
    "in.heating_setpoint_offset_magnitude",
    "in.cooling_setpoint_offset_magnitude",
    
    # Ventilation / infiltration
    "in.infiltration",
    "in.mechanical_ventilation",
    "in.natural_ventilation",
    "in.bathroom_spot_vent_hour",
    "in.range_spot_vent_hour",
    
    # Occupancy / internal gains
    "in.occupants",
    "in.ceiling_fan",
    "in.lighting_interior_use",
    "in.cooking_range",
    "in.clothes_dryer",
    "in.clothes_washer",
    
    # Climate / location
    "in.ashrae_iecc_climate_zone_2004",
    "in.city",
    "in.state",
    "in.weather_file_city",
    "in.weather_file_latitude",
    "in.weather_file_longitude"
]

# Filter the dataset
df_combined = df_combined[relevant_columns]

# Optional: inspect
# Show result
df_combined.to_csv("data/validation/single_family_detached.csv")