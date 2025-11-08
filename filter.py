import numpy as np
import pandas as pd
import os
from entise.constants.objects import Objects
from entise.constants.columns import Columns
from entise.constants.ts_types import Types
from entise.constants.constants import Constants

def filter_single_family_detached():

    folder_path = "data/validation/zones"
    # Define all filtering rules here:
    filter_rules = {
        "in.geometry_building_type_recs": ["Single-Family Detached"],
        "in.heating_fuel": ["Electricity"],
        "in.misc_pool": [np.nan],
        "in.vacancy_status":["Occupied"]
    }
    filtered_dfs = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)
            # Apply all filters in sequence
            for column, values in filter_rules.items():
                if column in df.columns:
                    df = df[df[column].isin(values)]
            # Add filename column (for traceability)
            df["filename"] = filename
            # Store only non-empty filtered data
            if not df.empty:
                filtered_dfs.append(df)

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
        "in.heating_setpoint_offset_period",
        "in.cooling_setpoint_offset_period",
        "in.heating_setpoint_has_offset",
        "in.cooling_setpoint_has_offset",
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
        "in.weather_file_longitude",
    ]

    # Filter the dataset
    df_combined = df_combined[relevant_columns]

    df_combined.to_csv("data/validation/single_family_detached.csv",index=False)

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
    df_with_years.to_csv("data/validation/single_family_detached.csv",index=False)    

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


def to_object_file():
    df=pd.read_csv("data/validation/single_family_detached.csv")
    df=df[["bldg_id",
           "year",
           "in.occupants",
           Objects.RESISTANCE,
           Objects.CAPACITANCE,
           "in.heating_setpoint",
           "in.heating_setpoint_offset_magnitude",
           "in.heating_setpoint_offset_period",
           "in.cooling_setpoint",
           "in.cooling_setpoint_offset_magnitude",
           "in.cooling_setpoint_offset_period",
           "in.window_areas",
           "filename",
           "in.weather_file_latitude",
           "in.weather_file_longitude",
           "in.sqft",
           "in.orientation",
           "in.geometry_stories",
           "in.state"]]
    
    df.rename(
    columns={
        "in.occupants":              Objects.INHABITANTS,
        "bldg_id":                   Objects.ID,
        "in.heating_setpoint":       Objects.TEMP_MIN,
        "in.cooling_setpoint":       Objects.TEMP_MAX,
        "in.weather_file_latitude":  Objects.LAT,
        "in.weather_file_longitude": Objects.LON,
        "in.geometry_stories":       "stories",
        "in.orientation":            Objects.ORIENTATION,
        "in.sqft":                   Objects.AREA,
        "in.state":                  "state"

    },
    inplace=True
    )

    #Convert Units

    df[Objects.TEMP_MIN] = pd.to_numeric(df[Objects.TEMP_MIN].astype(str).str.replace("F", "", regex=False), errors="coerce")
    df["in.heating_setpoint_offset_magnitude"] = pd.to_numeric(df["in.heating_setpoint_offset_magnitude"].astype(str).str.replace("F", "", regex=False), errors="coerce")
    df[Objects.TEMP_MAX] = pd.to_numeric(df[Objects.TEMP_MAX].astype(str).str.replace("F", "", regex=False), errors="coerce")
    df["in.cooling_setpoint_offset_magnitude"] = pd.to_numeric(df["in.cooling_setpoint_offset_magnitude"].astype(str).str.replace("F", "", regex=False), errors="coerce")
    
    df["in.heating_setpoint_offset_magnitude"] = ((df["in.heating_setpoint_offset_magnitude"]+ df[Objects.TEMP_MIN]- 32) * 5 / 9).round(2)
    df["in.cooling_setpoint_offset_magnitude"] = ((df["in.cooling_setpoint_offset_magnitude"]+ df[Objects.TEMP_MAX] - 32) * 5 / 9).round(2)

    df[Objects.TEMP_MIN] = ((df[Objects.TEMP_MIN]- 32) * 5 / 9).round(2)
    df[Objects.TEMP_MAX] = ((df[Objects.TEMP_MAX]- 32) * 5 / 9).round(2)
    
    df["in.heating_setpoint_offset_magnitude"] = (df["in.heating_setpoint_offset_magnitude"]- df[Objects.TEMP_MIN]).round(2)
    df["in.cooling_setpoint_offset_magnitude"] = (df["in.cooling_setpoint_offset_magnitude"]- df[Objects.TEMP_MAX]).round(2)

    df=df.rename(columns={  "in.heating_setpoint_offset_magnitude":f"offset_{Objects.TEMP_MIN}",
                            "in.cooling_setpoint_offset_magnitude":f"offset_{Objects.TEMP_MAX}",
                            "in.heating_setpoint_offset_period":f"{Types.HEATING}_offset_period",
                            "in.cooling_setpoint_offset_period":f"{Types.COOLING}_offset_period",
                          })

    df=df.loc[df[Objects.TEMP_MIN]<df[Objects.TEMP_MAX]]

    df[Objects.AREA] = pd.to_numeric(df[Objects.AREA])*0.09290304

    df[Objects.HEIGHT] = df["stories"]*Constants.DEFAULT_HEIGHT.value

    #Extract climate zone

    df["climate_zone"]=df["filename"].str.replace(".csv", "", regex=False).str.split("_").str[1:].str.join(" ")

    #Filter out year 2016, does not exist in teaser

    df=df.loc[pd.to_numeric(df["year"])!=2016]
    
    df.to_csv("data/validation/objects_entise.csv",index=False)


def get_unique_offset_periods():
    df= pd.read_csv("data/validation/objects_entise.csv")
    heating_offset_periods=df[f"{Types.HEATING}_offset_period"].unique()
    cooling_offset_periods=df[f"{Types.COOLING}_offset_period"].unique()

    print(f"Heating offset periods:{heating_offset_periods}")
    print()
    print(f"Cooling offset periods:{cooling_offset_periods}")


