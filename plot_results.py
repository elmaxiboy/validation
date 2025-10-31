import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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

def plot_hvac_loads(method):
    print(f"Plotting HVAC timeseries {method}")
    objects = pd.read_csv(os.path.join(".", "data/validation/objects_entise.csv"))

    for idx,obj in objects.iterrows():
        
        id   =obj[Objects.ID]
        year =obj["year"]

        print(f"Processing ID:{id}, year:{year}")

        climate_zone=objects.loc[objects[Objects.ID]==id,"climate_zone"].iloc[0]
        df_hvac = pd.read_csv(f"data/validation/hvac/{method}/{id}_{year}.csv", parse_dates=[Columns.DATETIME])
        df_internal_gains=pd.read_csv(f"data/validation/internal_gains/{method}/{id}.csv")
        df_solar_gains=pd.read_csv(f"data/validation/solar_gains/{id}.csv")
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


        ax1.set_ylabel("Load [kW]")
        ax2.set_ylabel("Temperature [°C]")


        # Combine legends
        lines, labels = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()

        ax1.legend(lines, labels , loc="upper left")
        ax2.legend(lines_2, labels_2 , loc="upper right")


        # Improve layout
        plt.title(f"Loads Over Time, climate zone:{climate_zone}")
        plt.xticks(rotation=45)
        plt.tight_layout()

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
            f"{Types.HEATING}_{Columns.DEMAND}[kWh]/{Objects.AREA}",
            f"{Types.COOLING}_{Columns.DEMAND}[kWh]/{Objects.AREA}"
        ],
        var_name="hvac_type",
        value_name="demand_per_m2"
    )

    palette = {
        "Heating": "#d73027",   
        "Cooling": "#4575b4"    
    }

    df_melted["hvac_type"] = df_melted["hvac_type"].str.replace("_demand\\[kWh\\]/area\\[m2\\]", "", regex=True)
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
        y=f"{Types.HEATING}_{Columns.DEMAND}[kWh]/{Objects.AREA}",
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
        y=f"{Types.COOLING}_{Columns.DEMAND}[kWh]/{Objects.AREA}",
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