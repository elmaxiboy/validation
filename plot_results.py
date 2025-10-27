from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import plotly.express as px
from entise.constants import Types
from entise.constants.columns import Columns
from entise.constants.objects import Objects


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

def nrel_vs_entise():
    df = pd.read_csv("data/validation/hvac/geoma/24083_2009.csv", parse_dates=[Columns.DATETIME])


    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(df[Columns.DATETIME], df[f"{Types.COOLING}_{Columns.DEMAND}[W]"], color="tab:blue", label="Cooling Load [W]")
    ax1.plot(df[Columns.DATETIME], df[f"{Types.HEATING}_{Columns.DEMAND}[W]"], color="tab:red", label="Heating Load [W]")
    
    ax1.set_ylabel("Load [W]", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    # Combine legends
    lines, labels = ax1.get_legend_handles_labels()
    ax1.legend(lines, labels , loc="upper right")

    # Improve layout
    plt.title("HVAC Loads Over Time")
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save figure
    plt.savefig("timeseries_temperature_loads.png", dpi=300)

nrel_vs_entise()