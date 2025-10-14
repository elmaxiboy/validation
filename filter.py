import pandas as pd
import os


def filter_by_geometry():

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

def get_simulated_years():
    simulated_years_path = "C:/Users/escobarm/Desktop/thesis/validation_data/counties"
    climate_zone_folders = os.listdir(simulated_years_path)
    print(climate_zone_folders)

    df = pd.read_csv("data/validation/single_family_detached.csv",index_col=0)
    df_with_years = pd.DataFrame()

    for idx, row in df.iterrows():
        climate_zone = str(row["filename"]).split("_")[0]
        climate_zone_folder = [folder for folder in climate_zone_folders if climate_zone in folder][0]
        print(f"{climate_zone_folder} contains building {row['bldg_id']}")

        available_files = os.listdir(f"{simulated_years_path}/{climate_zone_folder}/households/{row['bldg_id']}")
        sim_files = [file for file in available_files if "sim" in file]
        years = [file.split("_")[2].split(".")[0] for file in sim_files]

        # Convert years into a DataFrame
        years_df = pd.DataFrame({"year": years})

        # Convert row to a one-row DataFrame
        row_df = pd.DataFrame([row])

        # Cross join
        df_ = row_df.merge(years_df, how="cross")

        # Append results
        df_with_years = pd.concat([df_with_years, df_], ignore_index=True)

        print(f"Years found for bldg_id:{row['bldg_id']} = {years}")


    df.reset_index(drop=True,inplace=True)
    df_with_years.to_csv("data/validation/single_family_detached_per_year.csv",index=False)    

    return df_with_years
    


def copy_demand_files():
    path = "C:/Users/escobarm/Desktop/thesis/validation_data/counties"
    climate_zone_folders = os.listdir(path)
    buildings=pd.read_csv("data/validation/objects_rc.csv")
    save_path="data/validation/demand"

    for _, bldg in buildings.iterrows():
        foldername_elements=str(bldg["filename"]).split(".")[0].split("_")
        try:
            demand_path=f"{foldername_elements[0]}_{foldername_elements[1]}_{foldername_elements[2]}/households/{bldg["id"]}/{bldg["id"]}_timeseries_adjusted.csv"
        except:
            demand_path=f"{foldername_elements[0]}_{foldername_elements[1]}/households/{bldg["id"]}/{bldg["id"]}_timeseries_adjusted.csv"

        demand_file=pd.read_csv(path+"/"+demand_path)
        demand_file.to_csv(save_path+"/"+str(bldg["id"])+".csv",index=False)

copy_demand_files()