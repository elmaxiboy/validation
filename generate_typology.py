import pandas as pd
from teaser.project import Project
import json

def generate_buildings():

    prj = Project()
    prj.name = "ThesisValidation"

    df=pd.read_csv("data/validation/single_family_detached_per_year.csv")

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
        prj.add_residential(
            construction_data='tabula_de_standard',
            geometry_data='tabula_de_single_family_house',
            name=f"{row["bldg_id"]}_{row["year"]}",
            year_of_construction=row["year"],
            number_of_floors=row["in.geometry_stories"],
            height_of_floors=2.7,
            net_leased_area=row["in.sqft"]*0.09290304,
            inner_wall_approximation_approach="teaser_default",#typical length * height of floors + 2 * typical width * height of floors
            height_of_floors=2.7,
            net_leased_area=row["in.sqft"]*0.09290304,
            inner_wall_approximation_approach="teaser_default",#typical length * height of floors + 2 * typical width * height of floors
            )


    prj.calc_all_buildings()
    prj.calc_all_buildings()

    prj.save_project("results","results/")
    prj.save_project("results","results/")


def calculate_rc():

    df= pd.read_csv("data/validation/single_family_detached_per_year.csv")
    df["resistance[K W-1]"]=0
    df["capacitance[J K-1]"]=0

    df= pd.read_csv("data/validation/single_family_detached_per_year.csv")
    df["resistance[K W-1]"]=0
    df["capacitance[J K-1]"]=0

    # Load the TEASER JSON
    with open("results/results.json") as f:
        data = json.load(f)

    # Helper function: compute R and C for a single layer
    def layer_R_C(layer, area):
        thickness = layer["thickness"]            # m
        mat = layer["material"]
        k = mat["thermal_conduc"]                 # W/m·K
        rho = mat["density"]                      # kg/m³
        c = mat["heat_capac"] * 1000              # kJ/kg·K → J/kg·K
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
        print(f"building : {k}")

        #if k== "B498771_1918":
        try:
            # Extract the thermal zone
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

            df.loc[mask, "resistance[K W-1]"] = R_total
            df.loc[mask, "capacitance[J K-1]"] = C_total
            print("")

        except Exception as e:
            print("Error: "+str(e))
    df.to_csv("data/validation/single_family_detached_per_year_rc.csv")    


def to_object_file():
    df=pd.read_csv("data/validation/single_family_detached_per_year_rc.csv")
    df=df[["bldg_id","year","in.occupants","resistance[K W-1]","capacitance[J K-1]","in.heating_setpoint","in.cooling_setpoint","in.window_areas","filename","in.geometry_stories","in.weather_file_latitude","in.weather_file_longitude"]]
    df.rename(
    columns={
        "in.occupants": "inhabitants",
        "bldg_id": "id",
        "in.heating_setpoint": "min_temperature[C]",
        "in.cooling_setpoint": "max_temperature[C]",
        "in.weather_file_latitude": "latitude[degree]",
        "in.weather_file_longitude": "longitude[degree]",
        "in.window_areas": "windows_area",
    },
    inplace=True
    )
    df["min_temperature[C]"] = pd.to_numeric(df["min_temperature[C]"].astype(str).str.replace("F", "", regex=False), errors="coerce")
    df["max_temperature[C]"] = pd.to_numeric(df["max_temperature[C]"].astype(str).str.replace("F", "", regex=False), errors="coerce")

    df["min_temperature[C]"] = ((df["min_temperature[C]"] - 32) * 5 / 9).round(2)
    df["max_temperature[C]"] = ((df["max_temperature[C]"] - 32) * 5 / 9).round(2)



    df.to_csv("data/validation/objects_rc.csv",index=False)

#calculate_rc()
to_object_file()



#generate_buildings()