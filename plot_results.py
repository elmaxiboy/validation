import os
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from entise.constants import Types
from entise.constants.columns import Columns
from entise.constants.objects import Objects
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

def plot_bar_plot_resistance_capacitance():
    df = pd.read_csv("results/hvac_summary_geoma.csv")
    df["climate_zone"] = df["filename"].str.replace(".csv", "", regex=False).str.split("_").str[1:].str.join(" ")


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


def distribution_thermal_props(thermal_prop=Objects.RESISTANCE,mode:str="box"):

    df = pd.read_csv("results/hvac_summary_geoma.csv")
    
    df =df.loc[df[Objects.AREA]<500] #remove mansions
    
    df["climate_zone"] = df["filename"].str.replace(".csv", "", regex=False).str.split("_").str[1:].str.join(" ")

    # Define desired order
    climate_order = ["very cold", "cold", "marine", "hot humid", "mixed dry","hot dry"]

    # Convert to categorical with order
    df["climate_zone"] = pd.Categorical(df["climate_zone"], categories=climate_order, ordered=True)
    
    # Create faceted boxplots by year
    g = sns.catplot(
        data=df,
        x="climate_zone",
        y=thermal_prop,
        hue="stories",
        col="year",
        kind=mode,
        height=4,
        aspect=1.2
    )
    
    for ax in g.axes.flatten():
      ax.set_xticklabels(climate_order, rotation=45, ha="right")
      ax.set_xlabel("Climate Zone")
      ax.set_ylabel(thermal_prop)

    g.figure.subplots_adjust(top=0.9)
    g.figure.suptitle(f"{thermal_prop} by Climate Zone and Year")
    
    # Save the figure
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


#distribution_area()

#distribution_thermal_props(thermal_prop=Objects.RESISTANCE)
#distribution_thermal_props(thermal_prop=Objects.CAPACITANCE)

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

#scatterplot_resistance_capacitance()

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
    

#plot_box_plot_demand_by_area(Columns.OCCUPANCY_GEOMA)

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

    climate_order = (
        df.groupby("climate_zone")["fit_score"]
          .mean()
          .sort_values(ascending=True)
          .index.to_list()
    )

    df["climate_zone"] = pd.Categorical(df["climate_zone"], categories=climate_order, ordered=True)

    df["fit_score_log"] = (df["fit_score"])*100

    plt.figure(figsize=(8, 5))
    g = sns.barplot(
        data=df,
        x="climate_zone",
        y="fit_score_log",
        hue="year",
        order=climate_order)
    
    g.set_xlabel("Climate Zone")
    g.set_ylabel("Relative Error %")
    g.set_title(f"Overall Fit Score by Climate Zone — {method}")
    plt.legend(title="Building year")
    plt.tight_layout()
    plt.savefig(f"results/images/barplots/fit_score_{method}.png", dpi=300)
    plt.close()

################## PER ERROR COMPONENT ###########################


    # Melt and prepare dataframe (same as before)
    rel_error_cols = ["heating_demand_rel_error", "cooling_demand_rel_error"]

    df_melted = df.melt(
        id_vars=["climate_zone"],
        value_vars=rel_error_cols,
        var_name="metric",
        value_name="relative_error"
    )

    df_melted["Relative Error (%)"] = df_melted["relative_error"].abs()*100

    df_melted["type"] = df_melted["metric"].apply(
        lambda x: "Heating [kWh]" if "heating" in x.lower() else "Cooling [kWh]"
    )

    palette = {"Heating [kWh]": "#E74C3C", "Cooling [kWh]": "#3498DB"}

    # Create FacetGrid
    g = sns.FacetGrid(
        df_melted,
        row="climate_zone",
        row_order=climate_order,
        sharex=True,
        height=2.2,
        aspect=2,
        despine=False
    )

    # Draw boxplots
    g.map_dataframe(
        sns.boxplot,
        y="type",  # Use type as y for boxplot orientation
        x="Relative Error (%)",
        hue="type",
        palette=palette,
        orient="h",
        showfliers=False
    )

    # Remove y-axis ticks and labels to clean up
    for ax in g.axes.flat:
        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        ax.set_ylabel("")
        ax.set_yticks([])  # hide y-axis ticks
        ax.legend_.remove() if ax.get_legend() else None  # remove individual legends

    # Create a proper unified legend manually
    from matplotlib.patches import Patch
    # Set main title first
    g.figure.suptitle("Heating and Cooling Errors", fontsize=16)

    # Adjust layout to leave space for legend and title
    plt.tight_layout(rect=[0, 0, 1, 0.88])  # leave top ~12% for title + legend

    # Create a proper unified legend above the plots, below the title
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=palette[name], label=name) for name in palette]
    g.figure.legend(
        handles=legend_elements,
        title="HVAC Type",
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.95),  # slightly below suptitle
        frameon=False
    )

    plt.savefig(f"results/images/barplots/relative_error_{method}.png", dpi=300)
    plt.close()


    ################## BEST HEATING DEMAND ###########################

    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    
    df = pd.read_csv(f"results/best_heating_score_{method}_{name}.csv")

    climate_order = (
        df.groupby("climate_zone")["heating_demand_rel_error"]
          .mean()
          .sort_values(ascending=True)
          .index.to_list()
    )
    
    df["climate_zone"] = pd.Categorical(df["climate_zone"], categories=climate_order, ordered=True)

    df["heating_demand_rel_error"] = df["heating_demand_rel_error"]*100

    n_colors = df["year"].nunique()  # number of bars per climate_zone
    cmap = cm.get_cmap("Reds")  # full Reds colormap

    # Take a range avoiding the very lightest values (0.3 to 1.0)
    colors = [mcolors.rgb2hex(cmap(x)) for x in np.linspace(0.3, 1, n_colors)]
    plt.figure(figsize=(8, 5))
    g = sns.barplot(
        data=df,
        x="climate_zone",
        y="heating_demand_rel_error",
        hue="year",
        order=climate_order,
        palette=colors)
    
    g.set_xlabel("Climate Zone")
    g.set_ylabel("Relative Error (%)")
    g.set_title(f"Heating Score by Climate Zone — {method}")
    plt.legend(title="Building year")
    plt.tight_layout()
    plt.savefig(f"results/images/barplots/heating_score_{method}.png", dpi=300)
    plt.close()

        ################## BEST COOLING DEMAND ###########################

    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    
    df = pd.read_csv(f"results/best_cooling_score_{method}_{name}.csv")

    climate_order = (
        df.groupby("climate_zone")["cooling_demand_rel_error"]
          .mean()
          .sort_values(ascending=True)
          .index.to_list()
    )
    
    df["climate_zone"] = pd.Categorical(df["climate_zone"], categories=climate_order, ordered=True)

    df["cooling_demand_rel_error"] = df["cooling_demand_rel_error"]*100

    n_colors = df["year"].nunique()  # number of bars per climate_zone
    cmap = cm.get_cmap("Blues") 

    # Take a range avoiding the very lightest values (0.3 to 1.0)
    colors = [mcolors.rgb2hex(cmap(x)) for x in np.linspace(0.3, 1, n_colors)]
    plt.figure(figsize=(8, 5))
    g = sns.barplot(
        data=df,
        x="climate_zone",
        y="cooling_demand_rel_error",
        hue="year",
        order=climate_order,
        palette=colors)
    
    g.set_xlabel("Climate Zone")
    g.set_ylabel("Relative Error (%)")
    g.set_title(f"Cooling Score by Climate Zone — {method}")
    plt.legend(title="Building year")
    plt.tight_layout()
    plt.savefig(f"results/images/barplots/cooling_score_{method}.png", dpi=300)
    plt.close()

def hvac_loads_comparison(objects,res_factor,cap_factor,solar_gains_factor,method= Columns.OCCUPANCY_GEOMA):
    print(f"Plotting HVAC timeseries {method}")
    objects=objects.copy()

    for idx,obj in objects.iterrows():
        
        id   =obj[Objects.ID]
        year =obj["year"]

        print(f"Processing ID:{id}, year:{year}")

        climate_zone=objects.loc[objects[Objects.ID]==id,"climate_zone"].iloc[0]
        df_hvac_real= pd.read_csv(f"data/validation/demand/{id}.csv", parse_dates=[Columns.DATETIME])
        upper_clip=df_hvac_real[f"{Types.HEATING}_{Columns.DEMAND}[W]"].quantile(0.999)
        df_hvac_real[f"{Types.HEATING}_{Columns.DEMAND}[W]"]=df_hvac_real[f"{Types.HEATING}_{Columns.DEMAND}[W]"].clip(upper=upper_clip)

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
        ax1.plot(df_hvac_sim[Columns.DATETIME], df_hvac_sim[f"{Types.HEATING}_{Columns.DEMAND}[W]"]/1000, color="red", label="Simulated Heating Load",alpha=0.8,zorder=2)
        ax1.plot(df_hvac_sim[Columns.DATETIME], df_hvac_sim[f"{Types.COOLING}_{Columns.DEMAND}[W]"]/1000, color="deepskyblue", label="Simulated Cooling Load",alpha=0.8,zorder=2)
        
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

def hvac_load_real(method:str=Columns.OCCUPANCY_GEOMA,climate_zone:str="marine"):

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

def normalized_boxplot():

    """Creates a plot that compares the simulated and original heating demand over time."""
    import os
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    # # Generate datetime index for one year in 1-hour intervals
    # start_date = pd.to_datetime('2018-01-01')
    # end_date = start_date + pd.DateOffset(days=365)
    # datetime_index = pd.date_range(start=start_date, end=end_date, freq='1H')
    # datetime_index = datetime_index[:-1]


    # # Create simulated and original data for 30 time series
    # n_timeseries = 30
    # n_rows = len(datetime_index)

    # simulated_data = np.random.normal(loc=1000, scale=3000, size=(n_rows, n_timeseries))
    # original_data = np.random.normal(loc=1000, scale=3000, size=(n_rows, n_timeseries))

    # # Create DataFrame
    # simulated_df = pd.DataFrame(simulated_data, index=datetime_index)
    # original_df = pd.DataFrame(original_data, index=datetime_index)

    # # Set all values below 0 to 0
    # simulated_df[simulated_df < 0] = 0
    # original_df[original_df < 0] = 0

    path = os.path.join(os.path.abspath('./validation/counties'), 'G3500090_Mixed-Dry', 'households', 'G3500090_Mixed-Dry_results.csv')
    df = pd.read_csv(path, index_col=0, parse_dates=True)

    # Make each row the sum of the previous rows
    df = df.cumsum(axis=0)
    print(df)
    exit()

    #Next steps: Continue creating this plot based on the dummy data.

    # Add a ratio column for each column and group them under the correct first level column
    for column in data.columns.levels[0]:
        data[(column, 'Ratio')] = data[(column, 'Simulated')] / data[(column, 'Original')]

    # Get the maximum value of each 'Original' column
    max_values = data.xs('Original', level=1, axis=1).max(axis=0)

    print(data.iloc[:5, :].to_string())
    print(data.iloc[-5:, :].to_string())

    # Divide both 'Original' and 'Simulated' columns by the corresponding maximum values
    for column in data.columns.levels[0]:
        data[(column, 'Simulated')] /= max_values[column]
        data[(column, 'Original')] /= max_values[column]

    print(data.iloc[:5, :].to_string())
    print(data.iloc[-5:, :].to_string())

    # data.columns = pd.MultiIndex.from_product([data.columns.levels[0], ['Original', 'Simulated', 'Ratio']])


    # Use the ratio columns to calculate the median, upper quartile, lower quartile, upper 95%, and lower 5%
    # Calculate the statistics for each row using the ratio columns
    data[('Stats', 'Lower 5%')] = data.xs('Ratio', level=1, axis=1).quantile(0.05, axis=1)
    data[('Stats', 'Lower Quartile')] = data.xs('Ratio', level=1, axis=1).quantile(0.25, axis=1)
    data[('Stats', 'Median')] = data.xs('Ratio', level=1, axis=1).median(axis=1)
    data[('Stats', 'Upper Quartile')] = data.xs('Ratio', level=1, axis=1).quantile(0.75, axis=1)
    data[('Stats', 'Upper 95%')] = data.xs('Ratio', level=1, axis=1).quantile(0.95, axis=1)

    # Calculate the statistics for each row based on the 'Original' and 'Simulated' columns
    data[('Stats', 'Lower 5% (Original)')] = data.xs('Original', level=1, axis=1).quantile(0.05, axis=1)
    data[('Stats', 'Lower Quartile (Original)')] = data.xs('Original', level=1, axis=1).quantile(0.25, axis=1)
    data[('Stats', 'Median (Original)')] = data.xs('Original', level=1, axis=1).median(axis=1)
    data[('Stats', 'Upper Quartile (Original)')] = data.xs('Original', level=1, axis=1).quantile(0.75, axis=1)
    data[('Stats', 'Upper 95% (Original)')] = data.xs('Original', level=1, axis=1).quantile(0.95, axis=1)

    data[('Stats', 'Lower 5% (Simulated)')] = data.xs('Simulated', level=1, axis=1).quantile(0.05, axis=1)
    data[('Stats', 'Lower Quartile (Simulated)')] = data.xs('Simulated', level=1, axis=1).quantile(0.25, axis=1)
    data[('Stats', 'Median (Simulated)')] = data.xs('Simulated', level=1, axis=1).median(axis=1)
    data[('Stats', 'Upper Quartile (Simulated)')] = data.xs('Simulated', level=1, axis=1).quantile(0.75, axis=1)
    data[('Stats', 'Upper 95% (Simulated)')] = data.xs('Simulated', level=1, axis=1).quantile(0.95, axis=1)

    # Fill NaN values with 0
    data = data.fillna(0)

    print(data.iloc[:5, :].to_string())
    print(data.iloc[-5:, :].to_string())

    # Create the figure and axis
    plt.figure(figsize=(10, 10))
    ax = plt.gca()

    # Define a colormap with different shades of orange
    n_shades = 4
    cmap = plt.get_cmap('Oranges', n_shades)

    # Plot the custom boxplot-like representation
    stats = ['Lower 5%', 'Lower Quartile', 'Median', 'Upper Quartile', 'Upper 95%']
    for i, stat in enumerate(stats):
        color = cmap(i / (n_shades - 1))  # Get the color from the colormap
        plt.plot(data[('Stats', f'{stat} (Original)')], data[('Stats', f'{stat} (Simulated)')], color='black', linewidth=1, label=stat)
        if i < n_shades:
            if 1 < i < 3:
                alpha = 1
            else:
                alpha = 0.5
            plt.fill_between(data[('Stats', f'{stat} (Original)')], data[('Stats', f'{stats[i]} (Simulated)')], data[('Stats', f'{stats[i + 1]} (Simulated)')], color='Orange', alpha=alpha)

    # Plot the line from (0,0) to (1,1) for the ratio
    plt.plot([0, 1], [0, 1], color='black', linestyle='dashed', label='Ratio of 1', linewidth=2)

    # Set axis labels and title
    plt.xlabel('Normalized Heating Demand (Original)')
    plt.ylabel('Ratio (Simulated / Original)')
    plt.title('Comparison of Simulated and Original Heating Ratios over Time')

    # Set x-axis and y-axis limits
    plt.xlim(0, 1.1)
    plt.ylim(0, 1.1)

    # Add a legend
    plt.legend()

    # Show the grid
    plt.grid(True)

    # Display the plot
    plt.tight_layout()
    plt.show()