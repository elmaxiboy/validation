import os
from matplotlib import pyplot as plt
import matplotlib
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import geopandas as gpd
import seaborn as sns
from entise.constants import Types
from entise.constants.columns import Columns
from entise.constants.objects import Objects
from matplotlib.patches import Patch
from mpl_toolkits.basemap import Basemap
import tqdm
from matplotlib.colors import ListedColormap
matplotlib.use("Agg")


def digest_summary():
    df = pd.read_csv("results/hvac_summary_geoma.csv")
    
    df =df.loc[df[Objects.AREA]<500] #remove mansions
    
    df["climate_zone"] = df["filename"].str.replace(".csv", "", regex=False).str.split("_").str[1:].str.join(" ")

    # Define desired order
    climate_order = ["very cold", "cold", "marine", "hot humid", "mixed dry","hot dry"]

    # Convert to categorical with order
    df["climate_zone"] = pd.Categorical(df["climate_zone"], categories=climate_order, ordered=True)

    # Melt the dataframe to long format
    df_melted = df.melt(
        id_vars=["year", "inhabitants", "climate_zone"],
        value_vars=["heating:demand[Wh]", "cooling:demand[Wh]"],
        var_name="type",
        value_name="demand_Wh"
    )

    # Optional: rename for nicer legend labels
    df_melted["type"] = df_melted["type"].replace({
        "heating:demand[Wh]": "Heating Demand",
        "cooling:demand[Wh]": "Cooling Demand"
    })

    # Convert Wh → kWh for readability
    df_melted["demand_kWh"] = df_melted["demand_Wh"] / 1000

    palette = {
    "Heating Demand": "tab:red",
    "Cooling Demand": "tab:blue"
    }

    # Create the FacetGrid (rows = climate zone, columns = inhabitants group)
    g = sns.FacetGrid(df_melted,
                      col="inhabitants",
                      row="climate_zone",
                      height=4,
                      aspect=0.8,
                      sharey=False)

    # Plot side-by-side bars (hue = type of demand)
    g.map_dataframe(sns.barplot,
                    x="year",
                    y="demand_kWh",
                    hue="type",
                    palette=palette)

    # Adjust legend, labels, and rotation
    g.add_legend(title="Type of Demand")
    for ax in g.axes.flat:
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_horizontalalignment('right')

    plt.tight_layout()
    g.savefig("barplot_heating_vs_cooling_demand_without_big_houses.png", dpi=300)
    plt.close()

def plot_bar_plot_resistance_capacitance(df):

    climate_zones = df["climate_zone"].unique()

    # Loop over each climate zone
    for zone in climate_zones:
        df_zone = df[df["climate_zone"] == zone]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)

        # Plot Resistance
        sns.barplot(
            data=df_zone,
            x="year",
            y=Objects.RESISTANCE,
            ax=axes[0],
            color="tab:green"
        )
        axes[0].set_title(f"Resistance per Year ({zone})")
        axes[0].set_ylabel(Objects.RESISTANCE)
        axes[0].tick_params(axis='x', rotation=45)

        # Plot Capacitance
        sns.barplot(
            data=df_zone,
            x="year",
            y=Objects.CAPACITANCE,
            ax=axes[1],
            color="tab:purple"
        )
        axes[1].set_title(f"Capacitance per Year ({zone})")
        axes[1].set_ylabel(Objects.CAPACITANCE)
        axes[1].tick_params(axis='x', rotation=45)

        plt.suptitle(f"Thermal Properties for Climate Zone: {zone}", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(f"thermal_properties_{zone}.png", dpi=300)
        plt.close()

def plot_histogram_area_of_houses():
     df = pd.read_csv("results/hvac_summary_geoma.csv")
     df =df.loc[df[Objects.AREA]<500]
     df["climate_zone"] = df["filename"].str.replace(".csv", "", regex=False).str.split("_").str[1:].str.join(" ")
     g = sns.FacetGrid(df, col="climate_zone", height=3.5, aspect=.65)
     g.map(sns.histplot, Objects.AREA)
     g.savefig(f"histogram_areas_without_big_houses.png")

def box_plot():

    df = pd.read_csv("results/hvac_summary_geoma.csv")
    
    df =df.loc[df[Objects.AREA]<100] #remove mansions
    
    df["climate_zone"] = df["filename"].str.replace(".csv", "", regex=False).str.split("_").str[1:].str.join(" ")

    # Define desired order
    climate_order = ["very cold", "cold", "marine", "hot humid", "mixed dry","hot dry"]

    # Convert to categorical with order
    df["climate_zone"] = pd.Categorical(df["climate_zone"], categories=climate_order, ordered=True)

    # Melt the dataframe to long format
    df_melted = df.melt(
    id_vars=[
        "id", "year", "climate_zone", "inhabitants",
        "method", "stories", "area[m2]"
    ],
    value_vars=["resistance[K W-1]", "capacitance[J K-1]"],
    var_name="thermal_property",
    value_name="value"
)
    
    # Create faceted boxplots by year
    g = sns.catplot(
        data=df_melted,
        x="climate_zone",
        y="value",
        hue="thermal_property",
        col="year",
        kind="box",
        col_wrap=3,  # wrap columns if many years
        height=4,
        aspect=1.2,
        palette={"resistance[K W-1]": "tab:green", "capacitance[J K-1]": "tab:purple"}
    )
    
    # Rotate x labels and set titles
    g.set_xticklabels(rotation=45, ha="right")
    g.set_axis_labels("Climate Zone", "Thermal property")
    g.figure.subplots_adjust(top=0.9)
    g.figure.suptitle("Thermal properties by Climate Zone and Year")
    
    # Save the figure
    g.savefig("boxplot_thermal_properties.png", dpi=300)
    plt.close()

def distribution_thermal_props(df, thermal_prop=Objects.RESISTANCE, mode: str = "box"):

    match thermal_prop:
        case Objects.RESISTANCE:
            color = "tab:green"
            title = "Resistance"
        case Objects.CAPACITANCE:
            color = "tab:purple"
            title = "Capacitance"

    # Remove mansions
    df = df.loc[df[Objects.AREA] < 500]

    # ---- NEW YEAR CATEGORIES ----
    df["year group"] = pd.cut(
        df["year"],
        bins=[1900, 1950, 2000, 9999],
        labels=["1900–1950", "1950–2000", "2000+"],
        include_lowest=True
    )

    # Climate order
    climate_order = ["very cold", "cold", "marine", "hot humid", "mixed dry", "hot dry"]
    df["climate_zone"] = pd.Categorical(df["climate_zone"], categories=climate_order, ordered=True)

    # --------------------------------------------------
    # INLINE OUTLIER FILTER (IQR per climate zone & year group)
    # --------------------------------------------------
    cleaned = []
    for (cz, yr), group in df.groupby(["climate_zone", "year group"], dropna=False):
        if len(group) < 3:
            cleaned.append(group)
            continue
        
        q1 = group[thermal_prop].quantile(0.25)
        q3 = group[thermal_prop].quantile(0.75)
        iqr = q3 - q1
        
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        
        filtered = group[(group[thermal_prop] >= lower) & (group[thermal_prop] <= upper)]
        cleaned.append(filtered)

    df = pd.concat(cleaned, ignore_index=True)

    # --------------------------------------------------
    # Plot using YEAR GROUP instead of original year
    # --------------------------------------------------
    g = sns.catplot(
        data=df,
        x="climate_zone",
        y=thermal_prop,
        col="year group",
        col_wrap=3,
        kind=mode,
        height=4,
        aspect=1.2,
        color=color,
    )

    for ax in g.axes.flatten():
        ax.set_xticklabels(climate_order, rotation=45, ha="right", fontsize=13)
        ax.set_xlabel("")
        ax.set_ylabel(thermal_prop, fontsize=15)

    g.figure.subplots_adjust(top=0.85)
    g.figure.suptitle(
        f"{title} by Climate Zone and Construction Period",
        fontsize=18
    )

    g.savefig(f"{mode}plot_{thermal_prop}.png", dpi=300)
    plt.close()

def distribution_area(mode:str="box"):

    df = pd.read_csv("results/hvac_summary_geoma.csv")
    
    
    df["climate_zone"] = df["filename"].str.replace(".csv", "", regex=False).str.split("_").str[1:].str.join(" ")

    # Define desired order
    climate_order = ["very cold", "cold", "marine", "hot humid", "mixed dry","hot dry"]

    # Convert to categorical with order
    df["climate_zone"] = pd.Categorical(df["climate_zone"], categories=climate_order, ordered=True)
    
    # Create faceted boxplots by year
    g = sns.catplot(
        data=df,
        x="climate_zone",
        y=Objects.AREA,
        kind=mode,
        height=4,
        aspect=1.2
    )
    
    for ax in g.axes.flatten():
      ax.set_xticklabels(climate_order, rotation=45, ha="right")
      ax.set_xlabel("Climate Zone")
      ax.set_ylabel(Objects.AREA)

    g.figure.subplots_adjust(top=0.9)
    g.figure.suptitle(f"{Objects.AREA} by Climate Zone and Year")
    
    # Save the figure
    g.savefig(f"{mode}plot_{Objects.AREA}.png", dpi=300)
    plt.close()

def scatterplot_resistance_capacitance():

    df = pd.read_csv("results/hvac_summary_geoma.csv")    
    df["climate_zone"] = df["filename"].str.replace(".csv", "", regex=False).str.split("_").str[1:].str.join(" ")

    climate_order = ["very cold", "cold", "marine", "hot humid", "mixed dry","hot dry"]

    # Convert to categorical with order
    df["climate_zone"] = pd.Categorical(df["climate_zone"], categories=climate_order, ordered=True)

    for cz in df["climate_zone"].unique():
        df_cz=df.loc[df["climate_zone"]==cz]
        df_cz.to_csv(f"results/per_climate_zone/{cz}.csv")
        g = sns.FacetGrid(df_cz, col="year")
        g.map(sns.scatterplot,Objects.RESISTANCE, Objects.CAPACITANCE)
        g.savefig(f"scatter_plot_resistance_capacitance_{cz}.png")

def plot_hvac_loads(method):
    print(f"Plotting HVAC timeseries {method}")
    objects = pd.read_csv(os.path.join(".", f"results/fit_score_{method}.csv")).head(10)

    for idx,obj in objects.iterrows():
        
        id   =obj[Objects.ID]
        year =obj["year"]

        print(f"Processing ID:{id}, year:{year}")

        climate_zone=objects.loc[objects[Objects.ID]==id,"climate_zone"].iloc[0]
        df_hvac = pd.read_csv(f"data/validation/hvac/{method}/{id}_{year}.csv", parse_dates=[Columns.DATETIME])
        df_internal_gains=pd.read_csv(f"data/validation/internal_gains/{method}/{id}.csv")
        df_solar_gains=pd.read_csv(f"data/validation/solar_gains/{id}_{year}.csv")
        df_weather=pd.read_csv(f"data/validation/weather/cleaned/{climate_zone}.csv")
        df_weather[Columns.DATETIME] = pd.to_datetime(df_weather[Columns.DATETIME])
        df_solar_gains["datetime"]=pd.to_datetime(df_solar_gains["datetime"])

        fig, ax1 = plt.subplots(figsize=(12, 5))
        ax2=ax1.twinx()

        ax2.plot(df_weather[Columns.DATETIME], df_weather[f"{Columns.TEMP_AIR}"], color="tab:olive", label="Outter Temperature")
        ax2.plot(df_hvac[Columns.DATETIME], df_hvac[f"{Columns.TEMP_IN}"], color="tab:cyan", label="Inner Temperature")

        ax1.plot(df_solar_gains[Columns.DATETIME],df_solar_gains[Objects.GAINS_SOLAR]/1000, color="tab:orange", label="Solar Gains")
        ax1.plot(df_hvac[Columns.DATETIME], df_hvac[f"{Types.HEATING}_{Columns.DEMAND}[W]"]/1000, color="tab:red", label="Heating Load")
        ax1.plot(df_hvac[Columns.DATETIME], df_hvac[f"{Types.COOLING}_{Columns.DEMAND}[W]"]/1000, color="tab:blue", label="Cooling Load")

        ax2.set_ylabel("Temperature [°C]")
        ax1.set_ylabel("Load [kW]")


        # Combine legends
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        lines, labels = ax1.get_legend_handles_labels()
        

        ax1.legend(lines, labels , loc="upper left")
        ax2.legend(lines_2, labels_2 , loc="upper right")
        

        # Improve layout
        plt.title(f"Loads Over Time, climate zone:{climate_zone}")
        plt.xticks(rotation=45)

        # Save figure
        plt.savefig(f"results/images/timeseries/{method}/hvac_loads_solar_gains_{id}_{year}.png", dpi=300)

def plot_box_plot_demand_by_area(method):

    df = pd.read_csv("results/hvac_summary_geoma.csv")

    climate_order = ["very cold", "cold", "marine", "mixed dry","hot humid","hot dry",]
    df["climate_zone"] = pd.Categorical(df["climate_zone"], categories=climate_order, ordered=True)

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    year_order = sorted(df["year"].dropna().unique())

    df_melted = df.melt(
        id_vars=["climate_zone","year"],
        value_vars=[
            f"{Types.HEATING}_{Columns.DEMAND}_real[kWh]/{Objects.AREA}",
            f"{Types.COOLING}_{Columns.DEMAND}_real[kWh]/{Objects.AREA}"
        ],
        var_name="hvac_type",
        value_name="demand_per_m2"
    )

    palette = {
        "Heating": "#d73027",   
        "Cooling": "#4575b4"    
    }

    df_melted["hvac_type"] = df_melted["hvac_type"].str.replace("_demand_real\\[kWh\\]/area\\[m2\\]", "", regex=True)
    df_melted["hvac_type"] = df_melted["hvac_type"].str.capitalize()
    
    # HVAC together
    sns.set_theme(style="whitegrid", font_scale=1.2)
    g = sns.catplot(
        data=df_melted,
        x="hvac_type",
        y="demand_per_m2",
        col="climate_zone",
        row="year",
        row_order=year_order,
        kind="box",
        sharey=True,   # or True, depending on comparability
        height=4,
        aspect=0.8,
        palette=palette
    )
    
    g.set_titles(row_template="Year: {row_name}", col_template="Climate: {col_name}")
    g.set_axis_labels("", "Demand [kWh/m²]")
    g.figure.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.05)
    g.figure.text(0.02, 0.5, "Year", va="center", rotation="vertical", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"results/images/boxplots/{method}/boxplot_hvac_by_climate_zone", dpi=300)

    #Heating alone
    g = sns.catplot(
        data=df,
        x="climate_zone",
        y=f"{Types.HEATING}_{Columns.DEMAND}_real[kWh]/{Objects.AREA}",
        row="year",                     
        kind="box",
        row_order=year_order,           
        sharey=False,                  
        palette="Reds",
        height=4,
        aspect=1.2
    )

    g.set_titles(row_template="Year: {row_name}")
    g.set_axis_labels("Climate zone", "Heating demand [kWh/m²]")
    g.set_xticklabels(rotation=30)


    plt.tight_layout()    

    plt.savefig(f"results/images/boxplots/{method}/boxplot_heating_by_climate_zone", dpi=300)

    #Cooling alone

    g = sns.catplot(
        data=df,
        x="climate_zone",
        y=f"{Types.COOLING}_{Columns.DEMAND}_real[kWh]/{Objects.AREA}",
        row="year",                     
        kind="box",
        row_order=year_order,           
        sharey=False,                  
        palette="Blues",
        height=4,
        aspect=1.2
    )

    g.set_titles(row_template="Year: {row_name}")
    g.set_axis_labels("Climate zone", "Cooling demand [kWh/m²]")
    g.set_xticklabels(rotation=30)
    
    plt.tight_layout() 

    plt.savefig(f"results/images/boxplots/{method}/boxplot_cooling_by_climate_zone", dpi=300)

def scatter_plot_demand_real_vs_simulated(method):
    df = pd.read_csv(f"results/hvac_summary_{method}.csv")


    climate_order = ["very cold", "cold", "marine", "mixed dry","hot dry","hot humid"]
    years=[2002,2009]

    # Convert to categorical with order
    df["climate_zone"] = pd.Categorical(df["climate_zone"], categories=climate_order, ordered=True)
    
    for cz in climate_order:

        df_cz=df.loc[(df["climate_zone"]==cz)]
        
        df_cz=df_cz[["id",
                    "year",
                    "climate_zone",
                    "heating_demand_real[kWh]/area[m2]",
                    "heating_demand_simulated[kWh]/area[m2]",
                    "cooling_demand_real[kWh]/area[m2]",
                    "cooling_demand_simulated[kWh]/area[m2]",
                    ]]
        
        df_melted = df_cz.melt(
        id_vars=["id", "year", "climate_zone"],
        value_vars=[
            "heating_demand_real[kWh]/area[m2]",
            "heating_demand_simulated[kWh]/area[m2]",
            "cooling_demand_real[kWh]/area[m2]",
            "cooling_demand_simulated[kWh]/area[m2]",
        ],
        var_name="variable",
        value_name="demand[kWh]/area[m2]"
        )

        # Extract type (heating/cooling) and source (real/simulated)
        df_melted["type"] = df_melted["variable"].apply(lambda x: "heating" if "heating" in x else "cooling")
        df_melted["source"] = df_melted["variable"].apply(lambda x: "real" if "real" in x else "simulated")
        
        df_pivot = df_melted.pivot_table(
            index=["id", "year", "climate_zone", "type"],
            columns="source",
            values="demand[kWh]/area[m2]"
            ).reset_index()

        df_pivot.columns.name = None
        df_pivot = df_pivot.rename(columns={"real": "real_demand", "simulated": "simulated_demand"})
        
        palette = {
        "heating": "#d73027",   
        "cooling": "#4575b4"    
        }

        g = sns.FacetGrid(df_pivot,
                    col="year",
                    hue="type",
                    palette=palette)

        g.map(sns.scatterplot,"real_demand","simulated_demand")

        for ax in g.axes.flatten():
            lims = [
                min(ax.get_xlim()[0], ax.get_ylim()[0]),
                max(ax.get_xlim()[1], ax.get_ylim()[1])
            ]
            ax.plot(lims, lims, "--", color="gray", linewidth=1)
            ax.set_xlim(lims)
            ax.set_ylim(lims)

        g.add_legend()
        g.set_axis_labels("Real [kWh/m²]", "Simulated [kWh/m²]")
        
        g.savefig(f"results/images/scatterplots/{method}/demand_comparison_{cz}.png")

def scatter_plot_max_load_real_vs_simulated(method):
    df = pd.read_csv(f"results/hvac_summary_{method}.csv")


    climate_order = ["very cold", "cold", "marine", "mixed dry","hot dry","hot humid"]
    years=[2002,2009]

    # Convert to categorical with order
    df["climate_zone"] = pd.Categorical(df["climate_zone"], categories=climate_order, ordered=True)
    
    for cz in climate_order:

        df_cz=df.loc[(df["climate_zone"]==cz)]
        
        df_cz=df_cz[["id",
                    "year",
                    "climate_zone",
                    "heating_load_real_max[kW]",
                    "heating_load_simulated_max[kW]",
                    "cooling_load_real_max[kW]",
                    "cooling_load_simulated_max[kW]",
                    ]]
        
        df_melted = df_cz.melt(
        id_vars=["id", "year", "climate_zone"],
        value_vars=[
            "heating_load_real_max[kW]",
            "heating_load_simulated_max[kW]",
            "cooling_load_real_max[kW]",
            "cooling_load_simulated_max[kW]",
        ],
        var_name="variable",
        value_name="max_load[kW]"
        )

        # Extract type (heating/cooling) and source (real/simulated)
        df_melted["type"] = df_melted["variable"].apply(lambda x: "heating" if "heating" in x else "cooling")
        df_melted["source"] = df_melted["variable"].apply(lambda x: "real" if "real" in x else "simulated")
        
        df_pivot = df_melted.pivot_table(
            index=["id", "year", "climate_zone", "type"],
            columns="source",
            values="max_load[kW]"
            ).reset_index()

        df_pivot.columns.name = None
        df_pivot = df_pivot.rename(columns={"real": "real_max_load", "simulated": "simulated_max_load"})
        
        palette = {
        "heating": "#d73027",   
        "cooling": "#4575b4"    
        }

        g = sns.FacetGrid(df_pivot,
                    col="year",
                    hue="type",
                    palette=palette)

        g.map(sns.scatterplot,"real_max_load","simulated_max_load")

        for ax in g.axes.flatten():
            lims = [
                min(ax.get_xlim()[0], ax.get_ylim()[0]),
                max(ax.get_xlim()[1], ax.get_ylim()[1])
            ]
            ax.plot(lims, lims, "--", color="gray", linewidth=1)
            ax.set_xlim(lims)
            ax.set_ylim(lims)

        g.add_legend()
        g.set_axis_labels("Real Max. Load [kW]", "Simulated Max. Load [kW]")
        
        g.savefig(f"results/images/scatterplots/{method}/max_load_comparison_{cz}.png")

def barplot_ranking_fit_score(method,name):

    rel_error_cols = [
        "heating_demand_rel_error",
        "cooling_demand_rel_error",
    ]

    ################## OVERALL PERFOMANCE ###########################
    
    df = pd.read_csv(f"results/best_fit_score_{method}_{name}.csv")

    #Filter outliers

    
    climate_order = (
        df.groupby("climate_zone")["fit_score"]
          .mean()
          .sort_values(ascending=True)
          .index.to_list()
    )

    df["climate_zone"] = pd.Categorical(df["climate_zone"], categories=climate_order, ordered=True)


    plt.figure(figsize=(8, 5))
    g = sns.barplot(
        data=df,
        x="climate_zone",
        y="fit_score",
        hue="year",
        order=climate_order)
    
    g.set_xlabel("Climate Zone")
    g.set_ylabel("Relative Error %")
    g.set_title(f"Overall Fit Score by Climate Zone — {method}")
    plt.legend(title="Building year")
    plt.tight_layout()
    plt.savefig(f"fit_score_{method}.png", dpi=300)
    plt.close()

    ################## PER ERROR COMPONENT (WIDE LAYOUT) ###########################

    # Columns to melt
    rel_error_cols = ["heating_demand_rel_error", "cooling_demand_rel_error"]

    df_melted = df.melt(
        id_vars=["climate_zone"],
        value_vars=rel_error_cols,
        var_name="metric",
        value_name="relative_error"
    )

    # Convert to percentage
    df_melted["Relative Error (%)"] = df_melted["relative_error"] * 100

    # Heating / Cooling labels
    df_melted["Type"] = df_melted["metric"].apply(
        lambda x: "Heating" if "heating" in x.lower() else "Cooling"
    )

    # Colors
    palette = {"Heating": "#E74C3C", "Cooling": "#3498DB"}

    # Create a single-row subplot grid with one column per climate zone
    n_cols = len(climate_order)
    fig, axes = plt.subplots(
        nrows=1,
        ncols=n_cols,
        figsize=(3 * n_cols, 5),
        sharey=True
    )

    if n_cols == 1:
        axes = [axes]  # ensure iterability

    for ax, cz in zip(axes, climate_order):
        subset = df_melted[df_melted["climate_zone"] == cz]

        sns.boxplot(
            data=subset,
            x="Type",              # vertical boxplots
            y="Relative Error (%)",
            palette=palette,
            ax=ax,
        )

        ax.set_title(f"{cz}")
        ax.set_xlabel("")

        # *** REMOVE X–AXIS LABELS ***
        ax.set_xticklabels([])

    # Unified legend on the right side
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=palette[name], label=name) for name in palette]

    fig.legend(
        handles=legend_elements,
        title="HVAC Type",
        loc="center right",
        bbox_to_anchor=(0.99, 0.75),
        frameon=True
    )

    # Global title
    fig.suptitle("Heating and Cooling Error Distribution by Climate Zone", fontsize=16)

    plt.tight_layout(rect=[0, 0, 1, 1])  # make space for title & right legend

    plt.savefig(f"relative_error_{method}.png", dpi=300)
    plt.close()

def hvac_loads_comparison(objects,res_factor,cap_factor,solar_gains_factor,method= "occupancy_geoma"):

    objects=objects.copy()

    for idx,obj in tqdm.tqdm(objects.iterrows(), total=len(objects), desc="Plotting HVAC timeseries"):
        
        id   =obj[Objects.ID]
        year =obj["year"]
    

        climate_zone=objects.loc[objects[Objects.ID]==id,"climate_zone"].iloc[0]
        df_hvac_real= pd.read_csv(f"data/validation/demand/{id}.csv", parse_dates=[Columns.DATETIME])
        df_hvac_real[f"{Types.HEATING}_{Columns.DEMAND}[W]"]=df_hvac_real[f"{Types.HEATING}_{Columns.DEMAND}[W]"]

        df_hvac_sim = pd.read_csv(f"data/validation/hvac/{method}/{id}_{year}.csv", parse_dates=[Columns.DATETIME])
        df_solar_gains=pd.read_csv(f"data/validation/solar_gains/{id}_{year}.csv")
        df_weather=pd.read_csv(f"data/validation/weather/cleaned/{climate_zone}.csv")
        df_weather[Columns.DATETIME] = pd.to_datetime(df_weather[Columns.DATETIME])
        df_solar_gains["datetime"]=pd.to_datetime(df_solar_gains["datetime"])

        fig, ax1 = plt.subplots(figsize=(12, 5))

        ax2=ax1.twinx()
        ax1.plot(df_solar_gains[Columns.DATETIME],df_solar_gains[Objects.GAINS_SOLAR]/1000, color="gold", label="Solar Gains",alpha=0.5,zorder=1)
        ax2.plot(df_weather[Columns.DATETIME],df_weather[Columns.TEMP_AIR],color="tab:orange", label="Air Temperature",alpha=0.5,zorder=1)
        ax1.plot(df_hvac_real[Columns.DATETIME], df_hvac_real[f"{Types.HEATING}_{Columns.DEMAND}[W]"]/1000, color="darkred", label="Real Heating Load",zorder=2)
        ax1.plot(df_hvac_real[Columns.DATETIME], df_hvac_real[f"{Types.COOLING}_{Columns.DEMAND}[W]"]/1000, color="royalblue", label="Real Cooling Load",zorder=2)
        ax1.plot(df_hvac_sim[Columns.DATETIME], df_hvac_sim[f"{Types.HEATING}_{Columns.DEMAND}[W]"]/1000, color="red", label="Simulated Heating Load",alpha=0.7,zorder=2)
        ax1.plot(df_hvac_sim[Columns.DATETIME], df_hvac_sim[f"{Types.COOLING}_{Columns.DEMAND}[W]"]/1000, color="deepskyblue", label="Simulated Cooling Load",alpha=0.7,zorder=2)
        
        #ax1.plot(df_hvac_real[Columns.DATETIME], df_hvac_real[f"total_{Columns.POWER}"]/1000, color="black", label="Real total Load",alpha=0.8)

        ax1.set_zorder(ax2.get_zorder() + 1)
        ax1.patch.set_visible(False)

        ax1.set_ylabel("Load [kW]")
        ax2.set_ylabel("Temperature [C]")

        # collect handles *before* reordering zorder
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()

        # draw legends manually after layout
        leg1 = ax1.legend(lines, labels, loc="upper left", frameon=True)
        leg2 = ax2.legend(lines2, labels2, loc="upper right", frameon=True)

        # make sure both legends are visible above everything
        for leg in (leg1, leg2):
            leg.set_zorder(10)

        plt.figtext(0.015, 0.01, f"ID: {id}, Year: {year}, %Res. :{res_factor}, %Cap. :{cap_factor}, %S. Gains: {solar_gains_factor}", ha='left')

        # Improve layout
        plt.title(f"HVAC loads comparison Over Time, climate zone:{climate_zone}")
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Save figure
        plt.savefig(f"results/images/timeseries/{method}/{climate_zone}/{id}_{year}_hvac_comparison.png", dpi=100)
        plt.close

def hvac_load_real(method:str="occupancy_geoma",climate_zone:str="marine"):

    print(f"Plotting HVAC timeseries {method}")
    objects = pd.read_csv(os.path.join(".", f"results/fit_score_{method}.csv"))

    objects=objects.loc[(objects["climate_zone"]==climate_zone)&(objects["year"].isin([2009]))]

    for idx,obj in objects.iterrows():
        
        id   =obj[Objects.ID]
        year =obj["year"]

        print(f"Processing ID:{id}, year:{year}")

        climate_zone=objects.loc[objects[Objects.ID]==id,"climate_zone"].iloc[0]
        df_hvac_real= pd.read_csv(f"data/validation/demand/{id}.csv", parse_dates=[Columns.DATETIME])
        df_solar_gains=pd.read_csv(f"data/validation/solar_gains/{id}_{year}.csv")
        df_weather=pd.read_csv(f"data/validation/weather/cleaned/{climate_zone}.csv")
        df_weather[Columns.DATETIME] = pd.to_datetime(df_weather[Columns.DATETIME])
        df_solar_gains["datetime"]=pd.to_datetime(df_solar_gains["datetime"])

        fig, ax1 = plt.subplots(figsize=(12, 5))


        ax1.plot(df_solar_gains[Columns.DATETIME],df_solar_gains[Objects.GAINS_SOLAR]/1000, color="tab:orange", label="Solar Gains",alpha=0.5)
        ax1.plot(df_hvac_real[Columns.DATETIME], df_hvac_real[f"{Types.HEATING}_{Columns.DEMAND}[W]"]/1000, color="darkred", label="Real Heating Load")
        ax1.plot(df_hvac_real[Columns.DATETIME], df_hvac_real[f"{Types.COOLING}_{Columns.DEMAND}[W]"]/1000, color="royalblue", label="Real Cooling Load")


        ax1.set_ylabel("Load [kW]")

        # Combine legends
        lines, labels = ax1.get_legend_handles_labels()

        ax1.legend(lines, labels , loc="upper left")


        # Improve layout
        plt.title(f"Real HVAC loads Over Time, climate zone:{climate_zone}")
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Save figure
        plt.savefig(f"results/images/timeseries/{method}/{climate_zone}/{id}_{year}_real_hvac_.png", dpi=300)
        plt.close

def normalized_boxplot(df: pd.DataFrame):
    """
    Create a normalized boxplot comparing simulated vs real HVAC demand (heating, cooling, or combined),
    normalized by total demand (sum over time).
    """
    

    climate_zones=df["climate_zone"].unique()

    for cz in climate_zones:
        
        df_cz = df.copy()
        
        df_cz=df_cz.loc[df_cz["climate_zone"]==cz]

        ids=df_cz["id"].unique()

        for id in ids:

            df_cz_id=df_cz.copy()

            df_cz_id=df_cz_id.loc[df_cz_id["id"]==id]

            year=df_cz["year"].unique()[0]

            df_cz_id["hvac_simulated"]=df_cz_id[f"simulated_{Types.HEATING}_{Columns.DEMAND}[W]"] + df_cz_id[f"simulated_{Types.COOLING}_{Columns.DEMAND}[W]"]
            df_cz_id["hvac_real"]=df_cz_id[f"real_{Types.HEATING}_{Columns.DEMAND}[W]"] + df_cz_id[f"real_{Types.COOLING}_{Columns.DEMAND}[W]"]

            hvac_real_sum           =   df_cz_id["hvac_real"].sum()### <- USE THIS TO DIVIDE ALL CUMULATIVE SUMS
            hvac_simulated_sum      =   df_cz_id["hvac_simulated"].sum()
            heating_real_sum        =   df_cz_id[f"real_{Types.HEATING}_{Columns.DEMAND}[W]"].sum()
            heating_simulated_sum   =   df_cz_id[f"simulated_{Types.HEATING}_{Columns.DEMAND}[W]"].sum()
            cooling_real_sum        =   df_cz_id[f"real_{Types.COOLING}_{Columns.DEMAND}[W]"].sum()
            cooling_simulated_sum   =   df_cz_id[f"simulated_{Types.COOLING}_{Columns.DEMAND}[W]"].sum()

            df_cz_id["hvac_simulated_cum_sum"]     =   (df_cz_id["hvac_simulated"].cumsum())/hvac_simulated_sum
            df_cz_id["hvac_real_cum_sum"]          =   (df_cz_id["hvac_real"].cumsum())/hvac_real_sum
            df_cz_id["heating_simulated_cum_sum"]  =   (df_cz_id[f"simulated_{Types.HEATING}_{Columns.DEMAND}[W]"].cumsum())/heating_simulated_sum
            df_cz_id["heating_real_cum_sum"]       =   (df_cz_id[f"real_{Types.HEATING}_{Columns.DEMAND}[W]"].cumsum())/heating_real_sum
            df_cz_id["cooling_simulated_cum_sum"]  =   (df_cz_id[f"simulated_{Types.COOLING}_{Columns.DEMAND}[W]"].cumsum())/cooling_simulated_sum
            df_cz_id["cooling_real_cum_sum"]       =   (df_cz_id[f"real_{Types.COOLING}_{Columns.DEMAND}[W]"].cumsum())/cooling_real_sum

            fig, ax1 = plt.subplots(figsize=(12, 5))

            ax1.plot(df_cz_id["hvac_real_cum_sum"]      ,   df_cz_id["hvac_simulated_cum_sum"]    , label="Total HVAC")
            ax1.plot(df_cz_id["heating_real_cum_sum"]   ,   df_cz_id["heating_simulated_cum_sum"] , color="red",label="Heating")
            ax1.plot(df_cz_id["cooling_real_cum_sum"]   ,   df_cz_id["cooling_simulated_cum_sum"] , color="blue",label="Cooling")

            ax1.plot([0, 1], [0, 1], "k--", label="Perfect Match")

            plt.ylabel(f"Normalized Simulated Demand (by total)")
            plt.xlabel(f"Normalized Real Demand (by total)")
            plt.title(f"Normalized Comparison by Total Demand. ID: {id}, Climate Zone: {cz}, Year: {year}")
            plt.xlim(0, 1.1)
            plt.ylim(0, 1.1)
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"normalized_boxplots_{cz}_{id}.png")
            plt.close()

def build_large_distinct_palette():
    """Combine many categorical colormaps into a large distinct palette."""
    cmaps = ["tab20", "tab20b", "tab20c", "Set3", "Set2", "Set1", "Paired"]
    big = []

    for cmap in cmaps:
        cm = plt.get_cmap(cmap)
        big.extend([cm(i) for i in range(cm.N)])

    # Shuffle so similar colors are far apart
    np.random.seed(42)
    np.random.shuffle(big)

    return big

def normalized_individual_boxplot(df: pd.DataFrame):
    """
    Normalized HVAC comparison per climate zone.
    Uses a very large shuffled categorical palette (100+ colors)
    with no color repetition and a compact legend always inside the plot.
    """

    # Build big color palette once
    palette = build_large_distinct_palette()

    climate_zones = df["climate_zone"].unique()

    for cz in climate_zones:
        df_cz = df[df["climate_zone"] == cz].copy()
        fig, ax = plt.subplots(figsize=(10, 6))

        ids = df_cz["id"].unique()
        n_ids = len(ids)

        # Extend palette if needed (unlikely)
        colors = (palette * ((n_ids // len(palette)) + 1))[:n_ids]

        legend_handles = []
        legend_labels = []

        for idx, id in enumerate(ids):
            color = colors[idx]
            df_cz_id = df_cz[df_cz["id"] == id].copy()

            df_cz_id["hvac_simulated"] = (
                df_cz_id[f"simulated_{Types.HEATING}_{Columns.DEMAND}[W]"]
                + df_cz_id[f"simulated_{Types.COOLING}_{Columns.DEMAND}[W]"]
            )
            df_cz_id["hvac_real"] = (
                df_cz_id[f"real_{Types.HEATING}_{Columns.DEMAND}[W]"]
                + df_cz_id[f"real_{Types.COOLING}_{Columns.DEMAND}[W]"]
            )

            hvac_sim_sum = df_cz_id["hvac_simulated"].sum()
            hvac_real_sum = df_cz_id["hvac_real"].sum()
            if hvac_sim_sum == 0 or hvac_real_sum == 0:
                continue

            df_cz_id["hvac_simulated_cum_sum"] = (
                df_cz_id["hvac_simulated"].cumsum() / hvac_real_sum
            )
            df_cz_id["hvac_real_cum_sum"] = (
                df_cz_id["hvac_real"].cumsum() / hvac_real_sum
            )

            x = df_cz_id["hvac_real_cum_sum"].to_numpy()
            y = df_cz_id["hvac_simulated_cum_sum"].to_numpy()

            ax.plot(x, y, alpha=0.75, color=color)
            ax.scatter([x[-1]], [y[-1]], s=25, marker="o", color=color)

            handle = Line2D([], [], linestyle="none", marker="o", color=color, markersize=5)
            legend_handles.append(handle)
            legend_labels.append(str(id))

        # Diagonal
        ax.plot([0, 1], [0, 1], "k--")

        ax.set_xlabel("Normalized Real HVAC Demand")
        ax.set_ylabel("Normalized Simulated HVAC Demand")
        ax.set_title(f"Normalized HVAC Comparison — Climate Zone: {cz}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True)

        # Auto-size legend → always inside
        ncols = min(6, max(1, n_ids // 15))

        ax.legend(
            legend_handles,
            legend_labels,
            title="Building ID",
            fontsize="xx-small",
            loc="lower right",            
            bbox_to_anchor=(0.98, 0.02),
            ncol=ncols,
            frameon=True,
            borderpad=0.2,
            fancybox=True,
            framealpha=0.95
        )

        plt.tight_layout()
        plt.savefig(f"normalized_boxplot_{cz}.png", dpi=300)
        plt.close()

def smart_offset(x, y, scale=0.03):
    """
    Computes an inward orthogonal offset at the end of the curve.
    """
    if len(x) < 2:
        return -scale, scale

    dx = x[-1] - x[-2]
    dy = y[-1] - y[-2]

    # Normalize direction vector
    norm = np.hypot(dx, dy)
    if norm == 0:
        return -scale, scale
    dx, dy = dx / norm, dy / norm

    # Orthogonal vector
    ox, oy = -dy, dx

    # Always push it inward to the plot center at (0.5,0.5)
    sign = 1 if (ox*(0.5-x[-1]) + oy*(0.5-y[-1])) > 0 else -1

    return sign * ox * scale, sign * oy * scale

def normalized_joint_boxplot(df: pd.DataFrame):
    climate_zones = df["climate_zone"].unique()

    for cz in climate_zones:
        df_cz = df[df["climate_zone"] == cz].copy()
        ids = df_cz["id"].unique()

        # --------------------------
        # Helper function to build a plot
        # --------------------------
        def build_plot(hvac_sim_arr, hvac_real_arr, cz, title_suffix, filename_suffix, mode):
            if len(hvac_sim_arr) == 0:
                return

            hvac_sim_arr = np.array(hvac_sim_arr)
            hvac_real_arr = np.array(hvac_real_arr)

            # Colors by mode
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

            # -----------------------------------------
            # Correct quantile computation:
            # REAL = X-axis, SIM = Y-axis
            # -----------------------------------------
            q_levels = [0.05, 0.25, 0.5, 0.75, 0.95]

            qs_real = np.quantile(hvac_real_arr, q_levels, axis=0)
            qs_sim  = np.quantile(hvac_sim_arr,  q_levels, axis=0)

            # Sort by REAL median to ensure monotonic X
            x_med = qs_real[2]
            sort_idx = np.argsort(x_med)

            x_sorted = x_med[sort_idx]
            y_med = qs_sim[2][sort_idx]
            y_25  = qs_sim[1][sort_idx]
            y_75  = qs_sim[3][sort_idx]
            y_5   = qs_sim[0][sort_idx]
            y_95  = qs_sim[4][sort_idx]

            # -----------------------------------------
            # Plotting
            # -----------------------------------------
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.fill_between(x_sorted, y_5, y_95, color=c_5_95, alpha=0.35, label="5–95% range")
            ax.fill_between(x_sorted, y_25, y_75, color=c_25_75, alpha=0.60, label="25–75% quartile")
            ax.plot(x_sorted, y_med, color=c_median, lw=2.5, label="Median")

            # Perfect match line
            ax.plot([0, 1], [0, 1], "k--", label="Perfect Match")

            # Labels
            ax.set_xlabel("Normalized Real",fontsize=16)
            ax.set_ylabel("Normalized Simulated",fontsize=16)
            ax.set_title(f"HVAC Normalized Quartile Comparison ({title_suffix}) — {cz.capitalize()}",fontsize=16)
            
            ax.tick_params(axis='both', which='major', labelsize=12)
            ax.tick_params(axis='both', which='minor', labelsize=10)
            
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1.1)
            ax.grid(True)
            ax.legend(loc="lower right",fontsize=16)

            plt.tight_layout()
            plt.savefig(f"normalized_boxplot_quartiles_{filename_suffix}_{cz}.png", dpi=300)
            plt.close()

        # --------------------------
        # Accumulators per category
        # --------------------------
        comb_sim_list, comb_real_list = [], []
        heat_sim_list, heat_real_list = [], []
        cool_sim_list, cool_real_list = [], []

        # --------------------------
        # Process each building
        # --------------------------
        for bid in ids:
            df_b = df_cz[df_cz["id"] == bid].copy().sort_values("datetime")
            area=df_b.loc[df_b["id"]==bid,"area[m2]"].iloc[0]

            sim_h = (df_b[f"simulated_{Types.HEATING}_{Columns.DEMAND}[W]"].fillna(0)/area).values
            sim_c = (df_b[f"simulated_{Types.COOLING}_{Columns.DEMAND}[W]"].fillna(0)/area).values
            real_h = (df_b[f"real_{Types.HEATING}_{Columns.DEMAND}[W]"].fillna(0)/area).values
            real_c = (df_b[f"real_{Types.COOLING}_{Columns.DEMAND}[W]"].fillna(0)/area).values

            sim_total = sim_h + sim_c
            real_total = real_h + real_c

            real_sum = real_total.sum()
            real_h_sum = real_h.sum()
            real_c_sum = real_c.sum()

            # Combined
            if real_sum > 0:
                comb_sim_list.append(np.cumsum(sim_total) / real_sum)
                comb_real_list.append(np.cumsum(real_total) / real_sum)

            # Heating
            if real_h_sum > 0:
                heat_sim_list.append(np.cumsum(sim_h) / real_h_sum)
                heat_real_list.append(np.cumsum(real_h) / real_h_sum)

            # Cooling
            if real_c_sum > 0:
                cool_sim_list.append(np.cumsum(sim_c) / real_c_sum)
                cool_real_list.append(np.cumsum(real_c) / real_c_sum)

        # --------------------------
        # Build 3 plots with correct colors
        # --------------------------
        build_plot(comb_sim_list, comb_real_list, cz, "Combined HVAC", "combined", "combined")
        build_plot(heat_sim_list, heat_real_list, cz, "Heating Only", "HEATING", "heating")
        build_plot(cool_sim_list, cool_real_list, cz, "Cooling Only", "COOLING", "cooling")

def plot_us_buildings(df:pd.DataFrame):


    lat = df[Objects.LAT].values
    lon = df[Objects.LON].values
    area = df[Objects.AREA].values
    climate = df["climate_zone"].values

    # ----------------------------------------------------
    # FIXED CLIMATE ZONE ORDER (cold → hot)
    # ----------------------------------------------------
    climate_order = [
        "very cold",
        "cold",
        "marine",
        "mixed dry",
        "hot humid",
        "hot dry"
    ]

    # Filter only existing categories (avoid errors)
    climate_order = [cz for cz in climate_order if cz in df["climate_zone"].unique()]

    # ----------------------------------------------------
    # CUSTOM COLORMAP: blue → neutral → red
    # ----------------------------------------------------
    climate_colors = {
        "very cold": "#08306B",   # very deep blue
        "cold": "#2171B5",        # strong blue
        "marine": "#6A51A3",      # purple / midpoint
        "mixed dry": "#FD8D3C",   # orange
        "hot humid": "#FC4E2A",   # strong orange-red
        "hot dry": "#B10026",     # deep red
    }

    colors = [climate_colors[z] for z in climate_order]

    zone_to_idx = {z: i for i, z in enumerate(climate_order)}
    climate_color_idx = np.array([zone_to_idx[z] for z in climate])

    # ----------------------------------------------------
    # Jitter
    # ----------------------------------------------------
    jitter_scale = 1.8
    lat_jitter = lat + np.random.uniform(-jitter_scale, jitter_scale, size=len(lat))
    lon_jitter = lon + np.random.uniform(-jitter_scale, jitter_scale, size=len(lon))

    # ----------------------------------------------------
    # Basemap
    # ----------------------------------------------------
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)

    m = Basemap(
        projection='lcc',
        resolution='i',
        width=3.9e6, height=3e6,
        lat_0=39, lon_0=-102,
        ax=ax
    )

    m.shadedrelief()
    m.drawcoastlines(color='gray')
    m.drawcountries(color='gray')
    m.drawstates(color='gray')

    # ----------------------------------------------------
    # Scatter
    # ----------------------------------------------------
    sc = m.scatter(
        lon_jitter, lat_jitter,
        latlon=True,
        c=climate_color_idx,
        cmap=ListedColormap(colors),
        s=area,
        alpha=0.75,
        edgecolor='k',
        linewidth=0.25
    )

    # ----------------------------------------------------
    # CLIMATE ZONE LEGEND (ordered)
    # ----------------------------------------------------
    handles_climate = [
        plt.Line2D(
            [], [], marker="o", linestyle="",
            markersize=10,
            markerfacecolor=climate_colors[zone],
            markeredgecolor='k',
            label=zone
        )
        for zone in climate_order
    ]

    legend1 = fig.legend(
        handles=handles_climate,
        title="Climate Zone",
        loc="center left",
        bbox_to_anchor=(0.7, 0.78),
        frameon=True
    )

    fig.add_artist(legend1)

    # ----------------------------------------------------
    # AREA LEGEND
    # ----------------------------------------------------
    example_sizes = [50, 100, 200, 500]

    handles_area = [
        plt.Line2D(
            [], [], marker="o", linestyle="",
            markersize=np.sqrt(s/np.pi),
            markerfacecolor="gray", alpha=0.5,
            markeredgecolor="k",
            label=f"{s} m²"
        )
        for s in example_sizes
    ]

    ax.legend(
        handles=handles_area,
        title="Reference Areas",
        loc="lower left",
        frameon=True
    )

    # ----------------------------------------------------
    # TITLE
    # ----------------------------------------------------
    plt.title("Geographical distribution of validation dwellings", fontsize=16)

    # Space for legends
    plt.subplots_adjust(right=0.83)

    plt.savefig("geo_map_marine.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_marine_us_buildings(df:pd.DataFrame):

    df=df.loc[df["climate_zone"]=="marine"]
    df=df.drop_duplicates(["id"])

    lat = df[Objects.LAT].values
    lon = df[Objects.LON].values
    area = df[Objects.AREA].values
    climate = df["climate_zone"].values

    # ----------------------------------------------------
    # FIXED CLIMATE ZONE ORDER (cold → hot)
    # ----------------------------------------------------
    climate_order = [

        "marine",
    ]

    # Filter only existing categories (avoid errors)
    climate_order = [cz for cz in climate_order if cz in df["climate_zone"].unique()]

    # ----------------------------------------------------
    # CUSTOM COLORMAP: blue → neutral → red
    # ----------------------------------------------------
    climate_colors = {
        "marine": "#FAA805",      # purple / midpoint
    }

    colors = [climate_colors[z] for z in climate_order]

    zone_to_idx = {z: i for i, z in enumerate(climate_order)}
    climate_color_idx = np.array([zone_to_idx[z] for z in climate])

    # ----------------------------------------------------
    # Jitter
    # ----------------------------------------------------
    jitter_scale = 0.5
    lat_jitter = lat + np.random.uniform(-jitter_scale, jitter_scale, size=len(lat))
    lon_jitter = lon + np.random.uniform(-jitter_scale, jitter_scale, size=len(lon))

    # ----------------------------------------------------
    # Basemap
    # ----------------------------------------------------
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)

    m = Basemap(
        projection='lcc',
        resolution='i',
        width=1.5e6, height=1e6,
        lat_0=46, lon_0=-122,
        ax=ax
    )

    m.shadedrelief(zorder=0)
    m.drawcoastlines(color="gray", linewidth=0.6, zorder=1)
    m.drawcountries(color="gray", linewidth=0.6, zorder=1)
    m.drawstates(color="gray", linewidth=0.8, zorder=2)

    # ----------------------------------------------------
    # Latitude / Longitude grid
    # ----------------------------------------------------
    parallels = np.arange(44, 48, 1)      # lat lines
    meridians = np.arange(-126, -116, 2)  # lon lines

    m.drawparallels(
        parallels,
        labels=[1, 0, 0, 0],   # left only
        fontsize=9,
        linewidth=0.5,
        color="0.6",
        dashes=[2, 2],
        zorder=3
    )

    m.drawmeridians(
        meridians,
        labels=[0, 0, 0, 1],   # bottom only
        fontsize=9,
        linewidth=0.5,
        color="0.6",
        dashes=[2, 2],
        zorder=3
    )

    # ----------------------------------------------------
    # Scatter
    # ----------------------------------------------------
    sc = m.scatter(
        lon_jitter, lat_jitter,
        latlon=True,
        c=climate_color_idx,
        cmap=ListedColormap(colors),
        s=area,
        alpha=0.75,
        edgecolor='k',
        linewidth=0.25,
        zorder=10
    )

    # ----------------------------------------------------
    # AREA LEGEND
    # ----------------------------------------------------
    example_sizes = [50, 100, 200, 500]

    handles_area = [
        plt.Line2D(
            [], [], marker="o", linestyle="",
            markersize=np.sqrt(s/np.pi),
            markerfacecolor="gray", alpha=0.5,
            markeredgecolor="k",
            label=f"{s} m²"
        )
        for s in example_sizes
    ]

    ax.legend(
        handles=handles_area,
        title="Reference Areas",
        loc="lower left",
        frameon=True
    )

    # Space for legends
    plt.subplots_adjust(right=0.83)

    plt.savefig("geo_map_marine.png", dpi=300, bbox_inches="tight")
    plt.close()

#plot_marine_us_buildings(pd.read_csv("data/validation/objects_entise.csv"))


def plot_internal_gains(df):

    climate_order = [
        "very cold",
        "cold",
        "marine",
        "mixed dry",
        "hot humid",
        "hot dry"
    ]
      
    df = df.drop_duplicates(subset="id", keep="first")

    # -------------------------------------------------------------
    # 2) Load each CSV and compute monthly means per building
    # -------------------------------------------------------------
    monthly_records = []

    for _, row in  tqdm.tqdm(df.iterrows(), total=len(df), desc="Processing buildings"):
        id  = row["id"]
        cz  = row["climate_zone"]
        csv_path = f"data/validation/internal_gains/geoma/{id}.csv"

        # Load timeseries (assume columns: datetime, solar_gain)
        ts = pd.read_csv(csv_path, parse_dates=["datetime"])

        # Ensure correct datetime index
        ts = ts.set_index("datetime")

        # Monthly mean solar gains
        monthly = ts["gains_internal[W]"].resample("M").mean()
        monthly = monthly.to_frame("gains_internal[W]")
        monthly["month"] = monthly.index.month
        monthly["climate_zone"] = cz
        monthly["id"] = id

        monthly_records.append(monthly)

    # Combine
    monthly_df = pd.concat(monthly_records).reset_index(drop=True)

    # -------------------------------------------------------------
    # 3) Aggregate over buildings per climate zone & year group
    # -------------------------------------------------------------
    agg_df = (
        monthly_df
        .groupby(["climate_zone", "month"])["gains_internal[W]"]
        .mean()
        .reset_index()
    )

    # Ensure month order
    agg_df = agg_df.sort_values("month")

    # -------------------------------------------------------------
    # 4) Plot — Facet grid: rows = climate zones, columns = year groups
    # -------------------------------------------------------------
    sns.set_theme(style="whitegrid")

    g = sns.catplot(
        data=agg_df,
        x="month",
        y="gains_internal[W]",
        col="climate_zone",
        kind="bar",
        color="dimgrey",
        col_wrap=3,
        height=3.2,
        aspect=1.3,
        sharey=True,
        col_order=climate_order
    )

    g.set_axis_labels("Month", "Average Internal Gains (W)")
    g.set_titles("{col_name}")
    g.figure.suptitle("Average monthly internal gains", y=0.98, fontsize=14)
    g.figure.subplots_adjust(top=0.87)

    agg_df.to_csv("aggregated_internal_gains.csv", index=False)
    plt.savefig("internal_gains_summary.png")

def plot_solar_gains_collapsed(df):
    climate_order = [
        "very cold", "cold", "marine",
        "mixed dry", "hot humid", "hot dry"
    ]

    palette = [
    "#FFE8A3",  # very cold  (light warm yellow)
    "#FFC85C",  # cold       (strong yellow-orange)
    "#FFA22C",  # marine     (medium orange)
    "#FF7A1A",  # mixed dry  (deep orange)
    "#E8501B",  # hot humid  (red-orange)
    "#B22222",  # hot dry    (firebrick red, very visible)
]

    # ----------------------------------
    # 1) Year groups
    # ----------------------------------
    def assign_year_group(year):
        if year < 1950:
            return "1900–1950"
        elif year < 2000:
            return "1950–2000"
        else:
            return "2000+"

    df["year_group"] = df["year"].apply(assign_year_group)

    # ----------------------------------
    # 2) Load monthly solar gains
    # ----------------------------------
    monthly_records = []
    for _, row in tqdm.tqdm(df.iterrows(), total=len(df)):
        id  = row["id"]
        cz  = row["climate_zone"]
        year = row["year"]
        group = row["year_group"]
        window_area = row["window_area[m2]"]

        ts = pd.read_csv(
            f"data/validation/solar_gains/{id}_{year}.csv",
            parse_dates=["datetime"]
        ).set_index("datetime")

        monthly = (ts["gains_solar[W]"] / window_area).resample("M").mean()
        monthly = monthly.rename("gains_solar[W]").reset_index()
        monthly["month"] = monthly["datetime"].dt.month
        monthly["climate_zone"] = cz
        monthly["year_group"] = group
        monthly["id"] = id

        monthly_records.append(monthly)

    monthly_df = pd.concat(monthly_records, ignore_index=True)

    # ----------------------------------
    # 3) Aggregate
    # ----------------------------------
    agg_df = (
        monthly_df.groupby(["year_group", "climate_zone", "month"])
        ["gains_solar[W]"].mean()
        .reset_index()
    )

    # ----------------------------------
    # 4) Sort bars within each group/month
    # ----------------------------------
    sorted_agg = []
    for group in agg_df["year_group"].unique():
        sub_g = agg_df[agg_df["year_group"] == group]
        for m in range(1, 13):
            sub_m = sub_g[sub_g["month"] == m]
            sub_m = sub_m.sort_values("gains_solar[W]", ascending=True)
            sorted_agg.append(sub_m)
    sorted_agg = pd.concat(sorted_agg)

    # ----------------------------------
    # 5) Plot (short bars now visible!)
    # ----------------------------------
    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(
        nrows=1, ncols=3,
        figsize=(18, 4.8),
        sharey=True
    )

    year_groups = sorted(df["year_group"].unique())

    for ax, group in zip(axes, year_groups):

        sub = sorted_agg[sorted_agg["year_group"] == group]

        # use tiny dodge, not 0
        sns.barplot(
            data=sub,
            x="month",
            y="gains_solar[W]",
            hue="climate_zone",
            dodge=0.15,                # <-- key: small offset reveals short bars
            palette=palette,
            ax=ax,
            linewidth=0.5, edgecolor=".5",
        )

        ax.set_title(group, fontsize=13)
        ax.set_xlabel("Month")

        # remove per-axes legends
        if ax.get_legend() is not None:
            ax.get_legend().remove()

    axes[0].set_ylabel("Avg. Solar Gains (W/m2)")

    # ----------------------------------
    # 6) Single external legend
    # ----------------------------------
    fig.legend(
        labels=climate_order,
        handles=[plt.Rectangle((0,0),1,1,color=c) for c in palette],
        loc="center right",
        bbox_to_anchor=(0.99, 0.68)
    )

    fig.suptitle(
        "Monthly solar gains by construction year",
        fontsize=16
    )

    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.savefig("solar_gains_overlay_sorted.png", dpi=300)

def plot_ventilation_rate(df):

    climate_order = [
        "very cold",
        "cold",
        "marine",
        "mixed dry",
        "hot dry",
        "hot humid",
    ]

    # -------------------------------------------------------------
    # 2) Load each CSV and compute monthly means per building
    # -------------------------------------------------------------
    monthly_records = []
    df = df.drop_duplicates(subset="id", keep="first")
    for _, row in  tqdm.tqdm(df.iterrows(), total=len(df), desc="Processing buildings"):
        id  = row["id"]
        cz  = row["climate_zone"]
        csv_path = f"data/validation/ventilation/{id}.csv"

        # Load timeseries (assume columns: datetime, solar_gain)
        ts = pd.read_csv(csv_path, parse_dates=["datetime"])

        # Ensure correct datetime index
        ts = ts.set_index("datetime")

        # Monthly mean ventilation rate
        monthly = (ts["ventilation[W K-1]"]).resample("M").mean()
        monthly = monthly.to_frame("ventilation[W K-1]")
        monthly["month"] = monthly.index.month
        monthly["climate_zone"] = cz
        monthly["id"] = id

        monthly_records.append(monthly)

    # Combine
    monthly_df = pd.concat(monthly_records).reset_index(drop=True)

    # -------------------------------------------------------------
    # 3) Aggregate over buildings per climate zone & year group
    # -------------------------------------------------------------
    agg_df = (
        monthly_df
        .groupby(["climate_zone", "month"])["ventilation[W K-1]"]
        .mean()
        .reset_index()
    )

    # Ensure month order
    agg_df = agg_df.sort_values("month")

    # -------------------------------------------------------------
    # 4) Plot — Facet grid: rows = climate zones, columns = year groups
    # -------------------------------------------------------------
    sns.set_theme(style="whitegrid")

    g = sns.catplot(
        data=agg_df,
        x="month",
        y="ventilation[W K-1]",
        col="climate_zone",
        kind="bar",
        color="darkblue",
        col_order=climate_order,
        col_wrap=3,
        height=3.5,
        aspect=1.3,
        sharey=True
    )

    g.set_axis_labels("Month", "Average Ventilation Rate (W/K)")
    g.set_titles("{col_name}")
    g.figure.suptitle("Average monthly ventilation rate", y=0.98, fontsize=18)
    g.figure.subplots_adjust(top=0.83)
    #g.figure.subplots_adjust(top=0.85, right=0.95, left=0.25, bottom=0.15)
    agg_df.to_csv("aggregated_ventilation_rate.csv", index=False)
    plt.savefig("aggregated_ventilation_rate.png")

def dwelling_area_distribution(df):

    sns.set_theme(style="whitegrid")

    climate_order = [
        "very cold",
        "cold",
        "marine",
        "mixed dry",
        "hot dry",
        "hot humid",
    ]

    g = sns.catplot(
        data=df,
        y="climate_zone",
        x="area[m2]",
        kind="violin",
        height=4,
        color="tab:cyan",
        aspect=1.3,
        order=climate_order,     # <-- Correct ordering parameter
    )

    # Set axis labels
    g.set_axis_labels("Dwelling Area (m²)", "Climate Zone")

    # Increase grid resolution
    ax = g.ax
    ax.xaxis.set_major_locator(plt.MaxNLocator(10))
    ax.grid(True, which="major", linewidth=0.6)

    # Title + spacing
    g.figure.suptitle("Dwellings Area Distribution", y=0.98, fontsize=14)
    g.figure.subplots_adjust(top=0.85, right=0.95, left=0.25, bottom=0.15)
    plt.tight_layout()

    plt.savefig("dwelling_area_distribution.png", dpi=300)

def windows_area_distribution(df):

    # -------------------------------
    # 1) Define building year groups
    # -------------------------------
    def assign_year_group(year):
        if year < 1950:
            return "1900–1950"
        elif year < 2000:
            return "1950–2000"
        else:
            return "2000+"

    df["year_group"] = df["year"].apply(assign_year_group)

    sns.set_theme(style="whitegrid")

    # -------------------------------
    # 2) Create violin plot
    # -------------------------------
    g = sns.catplot(
        data=df,
        y="year_group",
        x="window_area[m2]",
        kind="violin",
        palette=["lightblue"],
        height=4,
        aspect=1.3
    )

    # Set axis labels
    g.set_axis_labels("Window Area (m²)", "Building Year Category")



    # Increase grid resolution (more ticks)
    ax = g.ax
    ax.xaxis.set_major_locator(plt.MaxNLocator(10))  # Increase number of gridlines
    ax.grid(True, which="major", linewidth=0.6)

    # Title + spacing
    g.figure.suptitle("Window Area Distribution by Building Year Category", 
                      y=0.98, fontsize=14)
    g.figure.subplots_adjust(top=0.85, right=0.95, left=0.25, bottom=0.15)
    plt.tight_layout()

    plt.savefig("window_area_distribution.png", dpi=300)

def occupancy_detected_distribution(df):
    climate_order = [
        "very cold",
        "cold",
        "marine",
        "mixed dry",
        "hot humid",
        "hot dry"
    ]

    sns.set_theme(style="whitegrid")

    # -------------------------------
    # 2) Create violin plot
    # -------------------------------
    g = sns.catplot(
        data=df,
        y="climate_zone",
        x="average_occupancy",
        kind="box",
        color="dimgrey",
        order=climate_order,
        height=4,
        aspect=1.3,
    )

    # Set axis labels
    g.set_axis_labels("% of yearly occupancy", "Climate Zone")



    # Increase grid resolution (more ticks)
    ax = g.ax
    ax.xaxis.set_major_locator(plt.MaxNLocator(10))  # Increase number of gridlines
    ax.grid(True, which="major", linewidth=0.6)

    # Title + spacing
    g.figure.suptitle("Yearly average of detected occupancy via GeoMA", 
                      y=0.98, fontsize=14)
    g.figure.subplots_adjust(top=0.85, right=0.95, left=0.25, bottom=0.15)
    plt.tight_layout()

    plt.savefig("occupancy_distribution.png", dpi=300)

def boxplot_hvac_demand():
    df = pd.read_csv("results/hvac_summary_geoma.csv")
    df = df[[
        "id",
        "climate_zone",
        "heating_demand_simulated[kWh]/area[m2]",
        "cooling_demand_simulated[kWh]/area[m2]"
    ]]
    df = df.drop_duplicates(subset="id", keep="first")

    climate_order = [
        "very cold",
        "cold",
        "marine",
        "mixed dry",
        "hot humid",
        "hot dry"
    ]

    # Melt
    df_melted = df.melt(
        id_vars=["climate_zone"],
        value_vars=[
            "heating_demand_simulated[kWh]/area[m2]",
            "cooling_demand_simulated[kWh]/area[m2]"
        ],
        var_name="metric",
        value_name="kWh/m2"
    )

    # Label heating vs cooling
    df_melted["Type"] = df_melted["metric"].apply(
        lambda x: "Heating" if "heating" in x.lower()
        else "Cooling"
    )

    # Correct palette
    palette = {
        "Heating": "#7F0D00",  # red
        "Cooling": "#004D81",  # blue
    }

    # Subplots (one per climate zone)
    n_cols = len(climate_order)
    fig, axes = plt.subplots(
        nrows=1,
        ncols=n_cols,
        figsize=(3 * n_cols, 5),
        sharey=True
    )

    if n_cols == 1:
        axes = [axes]

    for ax, cz in zip(axes, climate_order):
        subset = df_melted[df_melted["climate_zone"] == cz]

        sns.boxplot(
            data=subset,
            x="Type",
            y="kWh/m2",       
            palette=palette,
            ax=ax,
            showfliers=False
        )

        ax.set_title(cz)
        ax.set_xlabel("")
        ax.set_xticklabels([])

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=palette[name], label=name)
        for name in palette
    ]

    fig.legend(
        handles=legend_elements,
        title="HVAC Type",
        loc="center right",
        bbox_to_anchor=(0.99, 0.75),
        frameon=True
    )

    fig.suptitle("Distribution of total real HVAC demand for 2018", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 1])

    plt.savefig("real_hvac_demand.png", dpi=300)
    plt.close()

def boxplot_hvac_demand_marine():

    # ----------------------------------------------------
    # Load & filter
    # ----------------------------------------------------
    df = pd.read_csv("results/hvac_summary_geoma.csv")
    df = df[[
        "id",
        "climate_zone",
        "heating_demand_real[kWh]/area[m2]",
        "cooling_demand_real[kWh]/area[m2]"
    ]]
    df = df.drop_duplicates(subset="id", keep="first")
    df = df.loc[df["climate_zone"] == "marine"]

    # ----------------------------------------------------
    # Melt
    # ----------------------------------------------------
    df_melted = df.melt(
        id_vars=["climate_zone"],
        value_vars=[
            "heating_demand_real[kWh]/area[m2]",
            "cooling_demand_real[kWh]/area[m2]"
        ],
        var_name="metric",
        value_name="kWh/m²"
    )

    # Label heating vs cooling
    df_melted["Type"] = df_melted["metric"].apply(
        lambda x: "Heating" if "heating" in x.lower() else "Cooling"
    )

    # ----------------------------------------------------
    # Figure
    # ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 5))

    palette = {
        "Heating": "#C43B2B",  # red
        "Cooling": "#0672BA",  # blue
    }

    sns.boxplot(
        data=df_melted,
        x="Type",
        y="kWh/m²",
        order=["Heating", "Cooling"],
        palette=palette,
        showfliers=False,
        width=0.5,
        ax=ax
    )

    ax.set_xlabel("")
    ax.set_ylabel("Annual demand [kWh/m²]")

    plt.tight_layout()
    plt.savefig("real_hvac_demand_marine.png", dpi=300)
    plt.close()
boxplot_hvac_demand_marine()

def entise_vs_dc(objects=pd.read_csv("results/best_fit_score_geoma_vent_optimal_wshade_1_warea_1_gpp_65_capac_1_resis_1.csv")):

    objects=objects.copy()

    for idx,obj in tqdm.tqdm(objects.iterrows(), total=len(objects), desc="Plotting HVAC timeseries"):
        
        id=obj[Objects.ID]

        if not id==265047:
            continue
        
        df_weather=pd.read_csv(f"data/validation/weather/cleaned/marine.csv")
        df_weather[Columns.DATETIME] = pd.to_datetime(df_weather[Columns.DATETIME])

        df_solar_gains=pd.read_csv(f"data/validation/solar_gains/265047_2002.csv")
        df_solar_gains["datetime"]=pd.to_datetime(df_solar_gains["datetime"])

        df_internal_gains=pd.read_csv(f"data/validation/internal_gains/geoma/265047.csv")
        df_internal_gains[Columns.DATETIME] = pd.to_datetime(df_internal_gains[Columns.DATETIME])

        df_hvac_dc= pd.read_csv(f"data/validation/hvac/geoma/265047_2002.csv", parse_dates=[Columns.DATETIME])

        df_ventilation= pd.read_csv(f"data/validation/ventilation/265047.csv", parse_dates=[Columns.DATETIME])

        df_hvac_entise = pd.read_csv(f"entise_hvac_265047.csv", parse_dates=[Columns.DATETIME])
        
        day_of_the_week=[0,1,2]
        week=[13]
        hour=[12,13,14,15,16,17,18,19,20,21,22,23]
        df_hvac_dc = df_hvac_dc.loc[
            #(df_hvac_dc[Columns.DATETIME].dt.hour.isin([10,11,12,13]))&
            (df_hvac_dc[Columns.DATETIME].dt.day_of_week.isin(day_of_the_week))&
            (df_hvac_dc[Columns.DATETIME].dt.isocalendar().week.isin(week))&
            (df_hvac_dc[Columns.DATETIME].dt.hour.isin(hour))]
        
        df_hvac_entise = df_hvac_entise.loc[
            #df_hvac_entise[Columns.DATETIME].dt.hour.isin([10,11,12,13]) &
            (df_hvac_entise[Columns.DATETIME].dt.day_of_week.isin(day_of_the_week))&
            (df_hvac_entise[Columns.DATETIME].dt.isocalendar().week.isin(week))&
            (df_hvac_entise[Columns.DATETIME].dt.hour.isin(hour))]
        
        #df_solar_gains = df_solar_gains.loc[
        #    #df_hvac_entise[Columns.DATETIME].dt.hour.isin([10,11,12,13]) &
        #    (df_solar_gains[Columns.DATETIME].dt.day_of_week.isin(day_of_the_week))&
        #    (df_solar_gains[Columns.DATETIME].dt.isocalendar().week.isin(week))]
        
        df_internal_gains = df_internal_gains.loc[
            #df_hvac_entise[Columns.DATETIME].dt.hour.isin([10,11,12,13]) &
            (df_internal_gains[Columns.DATETIME].dt.day_of_week.isin(day_of_the_week))&
            (df_internal_gains[Columns.DATETIME].dt.isocalendar().week.isin(week))&
            (df_internal_gains[Columns.DATETIME].dt.hour.isin(hour))]

        #df_ventilation = df_ventilation.loc[
        #    #df_hvac_entise[Columns.DATETIME].dt.hour.isin([10,11,12,13]) &
        #    (df_ventilation[Columns.DATETIME].dt.day_of_week.isin(day_of_the_week))&
        #    (df_ventilation[Columns.DATETIME].dt.isocalendar().week.isin(week))]


        fig, ax1 = plt.subplots(figsize=(14, 5))

        ax1.grid(True)

        ax1.plot(df_internal_gains[Columns.DATETIME],
                 df_internal_gains[Objects.GAINS_INTERNAL],
                 color="black",
                 label="Internal Gains",
                 alpha=0.5,
                 zorder=0)
        
        #ax1.plot(df_ventilation[Columns.DATETIME],
        #         df_ventilation[Objects.VENTILATION],
        #         color="darkblue",
        #         label="Ventilation",
        #         alpha=0.5,
        #         zorder=0)

        #ax1.plot(df_solar_gains[Columns.DATETIME],
        #         df_solar_gains[Objects.GAINS_SOLAR],
        #         color="orange",
        #         label="Solar Gains",
        #         alpha=1,
        #         zorder=1,
        #         linewidth=1.5)


        
        ax1.plot(df_hvac_dc[Columns.DATETIME],
                 df_hvac_dc[f"{Types.HEATING}_{Columns.DEMAND}[W]"],
                 color="darkred",
                 label=f"Heating DC",
                 alpha=1,
                 zorder=1,
                 linewidth=2)
        
        ax1.plot(df_hvac_entise[Columns.DATETIME],
                 df_hvac_entise[f"{Types.HEATING}_{Columns.DEMAND}[W]"],
                 color="tab:red",
                 label=f"Heating EnTiSe",
                 alpha=0.7,
                 zorder=3,
                 
                )
        

        
        
        ax1.set_ylabel("Load [W]")
        lines, labels = ax1.get_legend_handles_labels()

        # draw legends manually after layout
        leg1 = ax1.legend(lines, labels, loc="upper left", frameon=True)

        # Improve layout
        plt.title(f"Derive and Couple vs EnTiSe plain 1R1C Building ID: {id}")
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Save figure
        plt.savefig(f"hvac_comparison.png", dpi=100)
        plt.close
