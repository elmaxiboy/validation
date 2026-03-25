import numpy
import pandas as pd

from constants import CLIMATE_ZONES, DT_INDEX, VENTILATION_MODES


def generator_ach_series(base, season_amp, mode_factor):
    """
    Produce synthetic ACH:
    - base ACH for the climate
    - seasonal sinusoidal variation (peaks in June–July–August)
    - diurnal ventilation behavior
    - random perturbation
    """

    # Extract day of year and hour
    day_of_year = DT_INDEX.dayofyear.values
    hour = DT_INDEX.hour.values
    N = len(DT_INDEX)

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

def generate_ach_series(cz_name: str = "marine"):
    params = CLIMATE_ZONES.get(cz_name)

    base = params["base"]
    amp = params["season_amp"]

    data = {}

    for mode_name, mode_factor in VENTILATION_MODES.items():
        ach = generator_ach_series(base, amp, mode_factor)
        data[mode_name] = ach

    df = pd.DataFrame(data, index=DT_INDEX)
    df.index.name = "datetime"

    return df
