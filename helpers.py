import os
from matplotlib import pyplot as plt
import numpy
import pandas as pd
from entise.constants.objects                           import Objects
from entise.constants.columns                           import Columns


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

def rmse(y_true, y_pred):
    y_true = numpy.asarray(y_true, dtype=float)
    y_pred = numpy.asarray(y_pred, dtype=float)
    return numpy.sqrt(numpy.mean((y_pred - y_true) ** 2))

def relative_error(sum_real, sum_sim, eps=1e-12):
    """
    Relative error on totals.
    Returns inf when real total is effectively zero and simulated is nonzero.
    Returns 0 when both are effectively zero.
    """
    if abs(sum_real) < eps:
        return 0.0 if abs(sum_sim) < eps else numpy.inf
    return abs(sum_sim - sum_real) / abs(sum_real)


def get_selection_score(metrics_dict, selection_metric="rmse", selection_component="total"):
    """
    Returns the scalar score used to decide the best year for a dwelling.
    The lower the better.
    """
    if selection_metric not in ("rmse", "relative_error"):
        raise ValueError(f"Unsupported selection_metric: {selection_metric}")

    if selection_component not in ("heating", "cooling", "total"):
        raise ValueError(f"Unsupported selection_component: {selection_component}")

    return metrics_dict[selection_metric][selection_component]

def plot_normalized_boxplot(hvac_sim_arr, hvac_real_arr, cz, title_suffix, filename_suffix, mode):
    if len(hvac_sim_arr) == 0:
        return

    hvac_sim_arr = numpy.array(hvac_sim_arr)
    hvac_real_arr = numpy.array(hvac_real_arr)

    if mode == "combined":
        c_5_95 = "orange"
        c_25_75 = "orange"
        c_median = "darkred"
    elif mode == "heating":
        c_5_95 = "#ffcccc"
        c_25_75 = "#ff6666"
        c_median = "#990000"
    elif mode == "cooling":
        c_5_95 = "#cce6ff"
        c_25_75 = "#66b3ff"
        c_median = "#004c99"
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    q_levels = [0.05, 0.25, 0.5, 0.75, 0.95]

    qs_real = numpy.quantile(hvac_real_arr, q_levels, axis=0)
    qs_sim  = numpy.quantile(hvac_sim_arr,  q_levels, axis=0)

    x_med = qs_real[2]
    sort_idx = numpy.argsort(x_med)

    x_sorted = x_med[sort_idx]
    y_med = qs_sim[2][sort_idx]
    y_25  = qs_sim[1][sort_idx]
    y_75  = qs_sim[3][sort_idx]
    y_5   = qs_sim[0][sort_idx]
    y_95  = qs_sim[4][sort_idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.fill_between(x_sorted, y_5, y_95, color=c_5_95, alpha=0.35, label="5–95% range")
    ax.fill_between(x_sorted, y_25, y_75, color=c_25_75, alpha=0.60, label="25–75% quartile")
    ax.plot(x_sorted, y_med, color=c_median, lw=2.5, label="Median")
    ax.plot([0, 1], [0, 1], "k--", label="Perfect Match")

    ax.set_xlabel("Normalized Real", fontsize=16)
    ax.set_ylabel("Normalized Simulated", fontsize=16)
    ax.set_title(f"HVAC Normalized Quartile Comparison ({title_suffix}) — {cz.capitalize()}", fontsize=16)

    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.tick_params(axis="both", which="minor", labelsize=10)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.1)
    ax.grid(True)
    ax.legend(loc="lower right", fontsize=16)

    plt.tight_layout()
    plt.savefig(f"normalized_boxplot_quartiles_{filename_suffix}_{cz}.png", dpi=300)
    plt.close(fig)


def plot_hvac_loads_comparison(
    obj,
    sim_heat_kwh,
    sim_cool_kwh,
    real_heat_kwh,
    real_cool_kwh,
    weather,
    output_dir="report_plots",
):
    obj_id = obj[Objects.ID]

    os.makedirs(output_dir, exist_ok=True)

    n = min(
        len(sim_heat_kwh),
        len(sim_cool_kwh),
        len(real_heat_kwh),
        len(real_cool_kwh),
        len(weather),
    )

    plot_df = pd.DataFrame({
        "datetime"      : pd.to_datetime(weather[Columns.DATETIME]).iloc[:n].to_numpy(),
        "real_heat_kwh" : numpy.asarray(real_heat_kwh)[:n],
        "real_cool_kwh" : numpy.asarray(real_cool_kwh)[:n],
        "sim_heat_kwh"  : numpy.asarray(sim_heat_kwh)[:n],
        "sim_cool_kwh"  : numpy.asarray(sim_cool_kwh)[:n],
    }).dropna(subset=["datetime"])

    fig, ax1 = plt.subplots(figsize=(14, 6))

    l1 = ax1.plot(
        plot_df["datetime"], plot_df["real_heat_kwh"],
        color="darkred", linewidth=1.2, label="Real Heating Load", zorder=3
    )
    l2 = ax1.plot(
        plot_df["datetime"], plot_df["real_cool_kwh"],
        color="royalblue", linewidth=1.2, label="Real Cooling Load", zorder=3
    )
    l3 = ax1.plot(
        plot_df["datetime"], plot_df["sim_heat_kwh"],
        color="red", alpha=0.7, linewidth=1.0, label="Simulated Heating Load", zorder=2
    )
    l4 = ax1.plot(
        plot_df["datetime"], plot_df["sim_cool_kwh"],
        color="deepskyblue", alpha=0.7, linewidth=1.0, label="Simulated Cooling Load", zorder=2
    )

    ax1.patch.set_visible(False)
    ax1.set_xlabel("Time")
    ax1.set_ylabel("HVAC load [kWh]")
    ax1.set_title(f"HVAC load comparison over time — {obj_id}")
    ax1.grid(True, alpha=0.3)

    handles = l1 + l2 + l3 + l4
    labels = [h.get_label() for h in handles]
    ax1.legend(handles, labels, loc="upper left", frameon=True)

    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{obj_id}_hvac_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

