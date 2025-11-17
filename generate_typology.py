import os
import numpy
import pandas as pd
from teaser.project import Project
import json
from entise.constants.objects import Objects
from entise.constants.constants import Constants 



def generate_buildings():

    prj = Project()
    prj.name = "ThesisValidation"

    df=pd.read_csv("data/validation/single_family_detached.csv")

    """ construction_data: Any,
    geometry_data: Any,
    name: Any,
    year_of_construction: Any,
    number_of_floors: Any,
    height_of_floors: Any,
    net_leased_area: Any,
    with_ahu: bool = False,
    internal_gains_mode: int = 1,
    inner_wall_approximation_approach: str = 'teaser_default',
    residential_layout: Any | None = None,
    neighbour_buildings: Any | None = None,
    attic: Any | None = None,
    cellar: Any | None = None,
    dormer: Any | None = None,
    number_of_apartments: Any | None = None """

    for idx,row in df.iterrows():
        #No database entry found for construction=tabula_de_standard, year2016
        if int(row["year"]==2016): 
            continue
        prj.add_residential(
            construction_data='tabula_de_standard',
            geometry_data='tabula_de_single_family_house',
            name=f"{row["bldg_id"]}_{row["year"]}",
            year_of_construction=row["year"],
            number_of_floors=row["in.geometry_stories"],
            height_of_floors=Constants.DEFAULT_HEIGHT.value,
            net_leased_area=row["in.sqft"]*0.09290304,
            inner_wall_approximation_approach="teaser_default",#typical length * height of floors + 2 * typical width * height of floors
            )


    prj.calc_all_buildings()

    prj.save_project("tipology","data/validation/tipology")


def calculate_rc():

    df= pd.read_csv("data/validation/single_family_detached.csv")
    df[Objects.RESISTANCE]=0
    df[Objects.CAPACITANCE]=0

    # Load the TEASER JSON
    with open("data/validation/tipology/tipology.json") as f:
        data = json.load(f)

    # Helper function: compute R and C for a single layer
    def layer_R_C(layer, area):
        thickness = layer["thickness"]            # m
        mat = layer["material"]
        k = mat["thermal_conduc"]                 # W/m·K
        rho = mat["density"]                      # kg/m³
        c = mat["heat_capac"]*1000           # kJ/kg·K → J/kg·K
        R = thickness / (k * area)                # K/W
        C = rho * c * area * thickness            # J/K
        return R, C

    # Compute R and C for a component (sum layers in series)
    def component_R_C(component):
        area = component["area"]
        layers = component["layer"].values()
        R_total = sum(layer_R_C(l, area)[0] for l in layers)
        C_total = sum(layer_R_C(l, area)[1] for l in layers)
        return R_total, C_total

    # Compute R and C for all elements in a dictionary
    def elements_R_C(elements):
        R_list = []
        C_list = []
        for elem in elements.values():
            R, C = component_R_C(elem)
            R_list.append(R)
            C_list.append(C)
        # Parallel combination of resistances
        if len(R_list)==0:
            R_eff=0
        else:
            R_eff = 1 / sum(1/r for r in R_list)
        if len (C_list)==0:
            C_eff=0
        else:
            C_eff = sum(C_list)

        return R_eff, C_eff

    for k,building in data["project"]["buildings"].items():
        
        try:
            # Extract the thermal zone
            if int(building["year_of_construction"])==2016:
                continue
            print(f"building : {k}")
            tz = building["thermal_zones"]["SingleDwelling"]
            # Compute R and C for all building components
            R_outer, C_outer = elements_R_C(tz["outer_walls"])
            R_windows, C_windows = elements_R_C(tz["windows"])
            R_roof, C_roof = elements_R_C(tz["rooftops"])
            R_floor, C_floor = elements_R_C(tz.get("floors",{}))
            R_ground, C_ground = elements_R_C(tz.get("ground_floors", {}))
            R_doors, C_doors = elements_R_C(tz.get("doors", {}))
            R_ceiling, C_ceiling = elements_R_C(tz.get("ceilings", {}))
            # Combine everything (walls, windows, roofs, floors, doors, ceilings)
            # For R, we consider the main envelope in parallel: walls, windows, roof, floor, doors, ceiling
            R_total = 1 / sum(1/r for r in [R_outer, R_windows, R_roof, R_floor, R_ground, R_doors, R_ceiling] if r > 0)
            C_total = sum([C_outer, C_windows, C_roof, C_floor, C_ground, C_doors, C_ceiling])
            print(f"Overall building R: {R_total:.5f} K/W")
            print(f"Overall building C: {C_total:.0f} J/K")
            bldg_id_str = str(k).split("B")[1].split("_")[0]
            year_value = int(building["year_of_construction"])  # ensure both are same type (int)

            mask = (
                (df["bldg_id"].astype(str) == bldg_id_str) &
                (df["year"].astype(int) == year_value)
            )

            df.loc[mask, Objects.RESISTANCE] = R_total
            df.loc[mask, Objects.CAPACITANCE] = C_total
            print("")

        except Exception as e:
            print("Error: "+str(e))
    df.to_csv("data/validation/single_family_detached.csv")    

def generate_windows_tipology():
    
    def extract_values(windows):
        windows_data = []
        for win_name, win_info in windows.items():
            layer = win_info.get("layer", {}).get("0", {})
            material = layer.get("material", {})

            # Compute transmittance (U-value) if thermal_conduc & thickness available
            # U = k / thickness, in W/m²K
            #k = material.get("thermal_conduc", None)
            #thickness = layer.get("thickness", None)
            transmittance=win_info.get("g_value")
            area= win_info.get("area")
            tilt=win_info.get("tilt")
            orientation=win_info.get("orientation")
            shading=win_info.get("shading_g_total")*win_info.get("g_value")
            values={
                Objects.AREA:area,
                Objects.TRANSMITTANCE:transmittance,
                Objects.ORIENTATION:orientation,
                Objects.TILT:tilt,
                "shading[1]":shading
            }
            windows_data.append(values)
        return windows_data

    df_windows=pd.DataFrame(columns=[
    Objects.ID,
    "year",
    Objects.AREA,
    Objects.TRANSMITTANCE,
    Objects.ORIENTATION,
    Objects.TILT,
    "shading[1]"
    ])
    
    with open("data/validation/tipology/tipology.json") as f:
        data = json.load(f)
    for k,building in data["project"]["buildings"].items():

        try:
            
            bldg_id_str = str(k).split("B")[1].split("_")[0]
            year_value = int(building["year_of_construction"]) 

            if int(building["year_of_construction"])==2016:
                continue
            print(f"building : {k}")
            tz = building["thermal_zones"]["SingleDwelling"]
            # Compute R and C for all building components
            windows_data = extract_values(tz["windows"])
            
            
            for window in windows_data:
                df_windows.loc[len(df_windows)] = {
                    Objects.ID: bldg_id_str,
                    "year": year_value,
                    Objects.AREA: window[Objects.AREA],
                    Objects.TRANSMITTANCE: window[Objects.TRANSMITTANCE],
                    Objects.ORIENTATION: window[Objects.ORIENTATION],
                    Objects.TILT: window[Objects.TILT],
                    "shading[1]": window["shading[1]"]
                }

        except Exception as e:
            print("Error: "+str(e))
    
    df_windows.to_csv("data/validation/tipology/windows.csv",index=False)


def estimate_window_areas_nrel():
    df=pd.read_csv("data/validation/objects_entise.csv")
    df_windows=pd.read_csv("data/validation/tipology/windows.csv")

    results = []

    # Orientation mapping for façades (assuming 'orientation' is main façade)
    orientation_map = {
    "North": 0,
    "Northeast": 45,
    "East": 90,
    "Southeast": 135,
    "South": 180,
    "Southwest": 225,
    "West": 270,
    "Northwest": 315
}

    # Rotation offset by main façade orientation
    rotation_offsets = {
        "Front": 0, "Right": 90, "Back": 180, "Left": 270
    }

    for _, row in df.iterrows():
        floor_area = row["area[m2]"]
        stories = row["stories"]
        main_orientation = row["orientation[degree]"]
        window_codes = row["in.window_areas"].split()

        # Parse codes like "F9" -> {"F": 0.09}
        ratios = {code[0]: int(code[1:]) / 100 for code in window_codes}

        # Approximate building geometry
        L = W = numpy.sqrt(floor_area / stories)
        wall_area = L * (stories * Constants.DEFAULT_HEIGHT.value)

        # Main orientation angle
        main_angle = orientation_map.get(main_orientation[:2], 0)

        # Map façades to absolute orientations
        facade_orientations = {
            "F": float(main_angle),
            "R": float((main_angle + 90) % 360),
            "B": float((main_angle + 180) % 360),
            "L": float((main_angle + 270) % 360)
        }

        # Compute per-facade window areas
        for key, ratio in ratios.items():
            win_area = wall_area * ratio
            results.append({
                "id": row["id"],
                "year": row["year"],
                "nrel_area[m2]": win_area,
                "orientation[degree]": facade_orientations.get(key, numpy.nan)
            })

    df_nrel_windows=pd.DataFrame(results)
    df_windows=df_windows.merge(df_nrel_windows,on=["id","year","orientation[degree]"],how="outer")
    df_windows.to_csv("data/validation/tipology/windows_nrel.csv",index=False)


# ---------------------------------------------------------
# 1. PARAMETERS
# ---------------------------------------------------------

CLIMATE_ZONES = {
    "hot dry": {"base": 0.5, "season_amp": 0.3},
    "hot humid": {"base": 0.7, "season_amp": 0.25},
    "mixed dry": {"base": 0.6, "season_amp": 0.20},
    "cold": {"base": 0.4, "season_amp": 0.35},
    "very cold": {"base": 0.3, "season_amp": 0.40},
    "marine": {"base": 0.5, "season_amp": 0.20},
}

MODES = {
    "typical": 1.0,
    "efficient": 0.7,
    "optimal": 0.4
}

OUTPUT_FOLDER = "data/validation/tipology/ventilation"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ---------------------------------------------------------
# 2. CREATE HOURLY TIMESTAMPS (2018, 15-min resolution)
# ---------------------------------------------------------

dt_index = pd.date_range(
    start="2018-01-01 00:00",
    end="2018-12-31 23:45",
    freq="15min"
)

N = len(dt_index)
day_of_year = dt_index.dayofyear.values

# ---------------------------------------------------------
# 3. GENERATION FUNCTION
# ---------------------------------------------------------

def generator_ach_series(base, season_amp, mode_factor):
    """
    Produce synthetic ACH:
    - base ACH for the climate
    - seasonal sinusoidal variation (peaks in June–July–August)
    - diurnal ventilation behavior
    - random perturbation
    """

    # Extract day of year and hour
    day_of_year = dt_index.dayofyear.values
    hour = dt_index.hour.values
    N = len(dt_index)

    # ---------------------------
    # Seasonal profile: peak ~ day 200 (mid-July)
    # ---------------------------
    summer_peak_day = 200
    seasonal = season_amp * numpy.cos(
    2 * numpy.pi * (day_of_year - summer_peak_day) / 365
)
    # Diurnal pattern (higher daytime ventilation)
    diurnal = 0.3 * numpy.sin(2 * numpy.pi * (hour / 24))

    # Random noise
    noise = numpy.random.normal(0, 0.05, N)

    # Combine all effects
    ach = base + seasonal + diurnal + noise

    # Apply ventilation mode factor
    ach = ach * mode_factor

    # ACH cannot be negative
    ach[ach < 0] = 0

    return ach

# ---------------------------------------------------------
# 4. GENERATE AND SAVE CSV FILES
# ---------------------------------------------------------

def generate_ach_series():
    for cz_name, params in CLIMATE_ZONES.items():
        for mode_name, mode_factor in MODES.items():

            base = params["base"]
            amp = params["season_amp"]

            ach = generator_ach_series(base, amp, mode_factor)

            df = pd.DataFrame({
                "datetime": dt_index,
                f"{mode_name}": ach
            })

            filename = f"{cz_name.replace(' ', '_')}_{mode_name}.csv"
            df.to_csv(os.path.join(OUTPUT_FOLDER, filename), index=False)

    print("ACH time series generated in:", OUTPUT_FOLDER)


generate_ach_series()

def total_window_area():
    df=pd.read_csv("data/validation/objects_entise.csv")
    df_windows=pd.read_csv("data/validation/tipology/windows.csv")
    df["window_area[m2]"]=0
    for idx,row in df.iterrows():
        window_area=df_windows.loc[(df_windows["id"]==row["id"]) & (df_windows["year"]==row["year"]),"area[m2]"].sum()
        df.at[idx,"window_area[m2]"]=window_area
    df["window_area[m2]"]=df["window_area[m2]"].round(2)
    df.to_csv("data/validation/objects_entise.csv",index=False)    

