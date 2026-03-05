# if needed later, can split this into three files

import json
import pandas as pd
from typing import Any, Dict, List
from teaser.project import Project
from entise.constants.objects import Objects
from entise.constants import Columns as C


def create_building_data():
"""
creates dataframe buildings.csv with basic building data
"""
building_data = [
{
'name': "mfh_06_standard",
'construction_data': 'tabula_de_standard',
'geometry_data': 'tabula_de_multi_family_house',
'year_of_construction': YEAR_OF_CONSTRUCTION,
'number_of_floors': NUMBER_OF_FLOORS,
'height_of_floors': HEIGHT_OF_FLOORS,
Objects.AREA: NET_LEASED_AREA,
},
{
'name': "mfh_06_retrofit",
'construction_data': 'tabula_de_retrofit',
'geometry_data': 'tabula_de_multi_family_house',
'year_of_construction': YEAR_OF_CONSTRUCTION,
'number_of_floors': NUMBER_OF_FLOORS,
'height_of_floors': HEIGHT_OF_FLOORS,
Objects.AREA: NET_LEASED_AREA,
},
{
'name': "mfh_06_adv_retrofit",
'construction_data': 'tabula_de_adv_retrofit',
'geometry_data': 'tabula_de_multi_family_house',
'year_of_construction': YEAR_OF_CONSTRUCTION,
'number_of_floors': NUMBER_OF_FLOORS,
'height_of_floors': HEIGHT_OF_FLOORS,
Objects.AREA: NET_LEASED_AREA,
}
]

buildings = pd.DataFrame(building_data)
buildings = buildings.reset_index().rename(columns={'index': 'id'}) # todo what is needed for EnTiSe ID?

buildings.to_csv(TYPOLOGY_DIR / "buildings.csv", index=False)


def generate_teaser_project():
prj = Project()
prj.name = "Paper_Doepfert2026"

buildings = pd.read_csv(TYPOLOGY_DIR / "buildings.csv")

for idx, row in buildings.iterrows():
# No database entry found for construction=tabula_de_standard, year2016
if int(row["year_of_construction"] == 2016):
continue
prj.add_residential(
construction_data=row["construction_data"],
geometry_data=row["geometry_data"],
# name=f"{row["bldg_id"]}_{row["year"]}",
name=row["name"], # geht das so, oder muss spezifisch sein?
year_of_construction=row["year_of_construction"],
number_of_floors=row["number_of_floors"],
height_of_floors=row["height_of_floors"],
net_leased_area=row[Objects.AREA], # todo correct area?
inner_wall_approximation_approach="teaser_default",
)

prj.calc_all_buildings() # todo better understand this
prj.save_project("typology", TYPOLOGY_DIR)
return prj


def teaser_calc_rc(prj):
"""
Adds R and C to buildings.csv.
"""
df = pd.read_csv(TYPOLOGY_DIR / "buildings.csv")
df[Objects.RESISTANCE] = 0.0
df[Objects.CAPACITANCE] = 0.0

for bldg in prj.buildings:
try:
year_value = int(bldg.year_of_construction)
if year_value == 2016:
continue

bldg.calc_building_parameter(
number_of_elements=1, # aggregates to 1R, 1C
merge_windows=True,
used_library="IBPSA"
)

# only 1 thermal zone (single dwelling)
zone = bldg.thermal_zones[0]
m = zone.model_attr

R_total = float(m.r_total_ow)
C_air = zone.volume * zone.density_air * zone.heat_capac_air
C_total = float(m.c1_ow) + C_air

bldg_id_str = str(bldg.name)

mask = (
(df["name"].astype(str) == bldg_id_str) &
(df["year_of_construction"].astype(int) == year_value)
)

df.loc[mask, Objects.RESISTANCE] = R_total
df.loc[mask, Objects.CAPACITANCE] = C_total

print(
f" Zone {zone.name}: "
f"R={R_total:.4f} K/W, "
f"C={C_total:.2e} J/K"
)

except Exception as e:
print(f"Error in building {getattr(bldg, 'name', '?')}: {e}")

df.to_csv(TYPOLOGY_DIR/ "buildings.csv", index=False)


def create_windows_csv():

json_path = TYPOLOGY_DIR / "typology.json"
with open(json_path, "r", encoding="utf-8") as f:
json_obj = json.load(f)

rows: List[Dict[str, Any]] = []
buildings = json_obj.get("project", {}).get("buildings", {})
id_numeric = 0
for bldg_id, bldg in buildings.items():
thermal_zones = bldg.get("thermal_zones", {})
for zone_id, zone in thermal_zones.items():
windows = zone.get("windows", {})
for win_id, win in windows.items():
rows.append({
C.ID: f"{id_numeric}",
C.ORIENTATION: win.get("orientation"),
C.TILT: win.get("tilt"),
C.AREA: win.get("area"),
C.TRANSMITTANCE: win.get("g_value"),
# C.SHADING: win.get("shading_g_total"), # TODO shading anpassen?
C.SHADING: 0.6, # so ist es in TABULA, in TEASER aber shading = 1
})
id_numeric += 1

df = pd.DataFrame(
rows,
columns=[Objects.ID, Objects.ORIENTATION, Objects.TILT, Objects.AREA, Objects.TRANSMITTANCE, "shading[1]"]
)

df.to_csv(TYPOLOGY_DIR / "windows.csv", index=False)


def prepare_buildings_for_entise():
# ADD ENTISE INPUT
buildings = pd.read_csv(TYPOLOGY_DIR / "buildings.csv")
buildings["hvac"] = "1R1C"
buildings[Objects.WEATHER] = "weather"

# VENTILATION TIMESERIES NEED TO BE ACH
buildings[Objects.VENTILATION] = "ventilation"
buildings[Objects.VENTILATION_COL] = "ACH[1/h]"

# SOLAR GAINS
buildings[Objects.WINDOWS] = "windows"
buildings[Objects.LAT] = LAT
buildings[Objects.LON] = LON

# INTERNAL GAINS
# konstante berechnung analog zu TABULA
internal_gains = 3 # W/m2
buildings[Objects.GAINS_INTERNAL] = internal_gains * NET_LEASED_AREA

buildings.to_csv(TYPOLOGY_DIR / "buildings.csv", index=False)


def typology():
"""
Creates building.csv, creates typology.json with TEASER, adds R+C to buildings.csv via TEASER.
"""
create_building_data()
teaser_project = generate_teaser_project()
teaser_calc_rc(teaser_project)
create_windows_csv()
prepare_buildings_for_entise()


if __name__ == "__main__":
typology()
