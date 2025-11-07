import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm

from validate import calculate_fit_score, derive_hvac, derive_solar_gains, summarize_hvac
from entise.constants.columns import Columns

# Parameter grids
reduce_shading_values = np.arange(0, 1.01, 0.1)     
reduce_window_area_values = np.arange(0, 1.01, 0.1)
ventilation_modes = ["optimal"]
capacitance_factors = np.arange(0, 0.05, 0.1)
resistance_factors = np.arange(1, 1.51, 0.1)

# Load and filter objects
objects = pd.read_csv("data/validation/objects_entise.csv")
objects = objects.loc[objects["climate_zone"] == "marine"]
objects = objects.loc[objects["year"].isin([2009])]
objects = objects.loc[objects["id"]==288215]

# Prepare results
results = []

# Create all combinations
param_combos = list(itertools.product(
    reduce_shading_values,
    reduce_window_area_values,
    ventilation_modes,
    capacitance_factors,
    resistance_factors
))

# Main loop with progress bar
for reduce_shading, reduce_window_area, ventilation_mode, capacitance_factor, resistance_factor in tqdm(
    param_combos, desc="Optimizing parameters", unit="combo"
):
    derive_solar_gains(objects, reduce_window_area, reduce_shading)
    derive_hvac(
        objects,
        capacitance_factor=capacitance_factor,
        resistance_factor=resistance_factor,
        ventilation_mode=ventilation_mode,
        method=Columns.OCCUPANCY_GEOMA
    )

    df = summarize_hvac(objects, Columns.OCCUPANCY_GEOMA)

    fit_score, heating_error, cooling_error = calculate_fit_score(
        df,
        Columns.OCCUPANCY_GEOMA,
        name=f"vent_{ventilation_mode}_wshade_{reduce_shading:.2f}_warea_{reduce_window_area:.2f}_capac_{capacitance_factor:.2f}_resis_{resistance_factor:.2f}"
    )

    results.append({
        "reduce_shading": reduce_shading,
        "reduce_window_area": reduce_window_area,
        "ventilation_mode": ventilation_mode,
        "capacitance_factor": capacitance_factor,
        "resistance_factor": resistance_factor,
        "fit_score": fit_score,
        "heating_error": heating_error,
        "cooling_error": cooling_error
    })

results_df = pd.DataFrame(results)

best_row = results_df.loc[results_df["fit_score"].idxmin()]
print("\n✅ Best parameters:")
print(best_row)

results_df.to_csv("fit_score_results.csv", index=False)
