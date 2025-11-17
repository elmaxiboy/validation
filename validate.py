
import logging
import os
import numpy
import pandas as pd

# Import the new TimeSeriesGenerator
from entise.constants import Types
from entise.constants.columns import Columns
from entise.constants.objects import Objects
from entise.constants.constants import Constants
from entise.core.generator import TimeSeriesGenerator
from entise.methods.auxiliary.internal.strategies import InternalOccupancy
from entise.methods.auxiliary.ventilation.strategies import VentilationTimeSeries,VentilationConstant
from entise.methods.hvac.R1C1 import calculate_timeseries 
from entise.methods.auxiliary.solar.strategies import SolarGainsPVLib
import pvlib
import tqdm

cwd="."
solar_gains_folder="data/validation/solar_gains"
demand_folder="data/validation/demand"
occupancy_folder=f"data/validation/{Types.OCCUPANCY}"
hvac_folder=f"data/validation/hvac"
weather_folder="data/validation/weather"
real_demand_folder="C:/Users/escobarm/Desktop/thesis/validation_data/counties"
temperature_setpoints_folder="data/validation/temperature_setpoints"



#if not obj["climate_zone"]=="marine":
#    continue
#if not int(obj["id"])==(77361):
#    continue

def generate_min_max_temperatures_schedule():
    global objects
    def shift_period(period_str, default_periods):
        """
        Returns a list of tuples (start_hour, end_hour) shifted by +/- hours.
        """
        if pd.isna(period_str):
            return []

        # Split period and shift
        if '+' in period_str:
            period_base, shift = period_str.rsplit('+', 1)
            shift_hours = int(shift.replace('h', ''))
        elif '-' in period_str:
            period_base, shift = period_str.rsplit('-', 1)
            shift_hours = -int(shift.replace('h', ''))
        else:
            period_base = period_str
            shift_hours = 0

        period_base = period_base.strip()

        periods = default_periods.get(period_base)
        if periods is None:
            return []

        # normalize to list of tuples
        if isinstance(periods[0], tuple):
            periods_list = periods
        else:
            periods_list = [periods]

        # shift hours
        shifted = []
        for start, end in periods_list:
            start_shifted = (start + shift_hours) % 24
            end_shifted = (end + shift_hours) % 24
            shifted.append((start_shifted, end_shifted))
        return shifted

    def in_period(hour, start, end):
        """Check if a given hour is in start-end, handling overnight periods."""
        if start < end:
            return start <= hour < end
        else:
            # overnight
            return hour >= start or hour < end

    # Generate timeseries for full year of 2018, 15-min intervals
    datetime_index = pd.date_range("2018-01-01", "2018-12-31 23:45", freq="15T")

    heating_periods = {
        "Day": (9, 17),
        "Night": (22, 7),
        "Day and Night": [(9, 17), (22, 7)],
    }

    cooling_periods = {
        "Day Setup": (9, 17),
        "Night Setback": (22, 7),
        "Night Setup": (22, 7),
        "Day and Night Setup": [(9, 17), (22, 7)],
        "Day Setup and Night Setback": [(9, 17), (22, 7)],
    }
    processed_ids = []
    

    for _, row in objects.iterrows():
        if row["id"] in processed_ids:
            continue

        df = pd.DataFrame({"datetime": datetime_index})
        df["id"] = row["id"]

        # === Heating (subtract offset to create setback) ===
        heating_periods_shifted = shift_period(row["heating_offset_period"], heating_periods)
        heating_setpoint = []
        for dt in df["datetime"]:
            hour = dt.hour + dt.minute / 60
            temp = row["min_temperature[C]"]
            for start, end in heating_periods_shifted:
                if in_period(hour, start, end):
                    temp -= row["offset_min_temperature[C]"]  #subtract offset
                    break
            heating_setpoint.append(temp)
        df["min_temperature[C]"] = heating_setpoint

        # === Cooling (add offset to create setup) ===
        cooling_periods_shifted = shift_period(row["cooling_offset_period"], cooling_periods)
        cooling_setpoint = []
        for dt in df["datetime"]:
            hour = dt.hour + dt.minute / 60
            temp = row["max_temperature[C]"]
            for start, end in cooling_periods_shifted:
                if in_period(hour, start, end):
                    temp += row["offset_max_temperature[C]"]  #add offset
                    break
            cooling_setpoint.append(temp)
        df["max_temperature[C]"] = cooling_setpoint
        df["max_temperature[C]"]=df["max_temperature[C]"].round(2)
        df["min_temperature[C]"]=df["min_temperature[C]"].round(2)
        df=df.drop(columns="id")
        df.to_csv(f"data/validation/temperature_setpoints/{row["id"]}.csv",index=False)
        
        processed_ids.append(row["id"])


#DERIVE OCCUPANCY SCHEDULE
def derive_occupancy_schedule(objects:pd.DataFrame):

    df_summary=pd.DataFrame(columns=[Objects.ID,"method",f"average_{Types.OCCUPANCY}"])
    
    #Discard year,resistance and capacitance
    objects=objects.drop(columns=["year",Objects.RESISTANCE,Objects.CAPACITANCE])
    objects=objects.drop_duplicates()


    processed_ids = set()

    for idx,obj in objects.iterrows():
        data = {}

        obj_id = str(obj[Objects.ID])

        #if not obj_id=="80450":
        #    continue
    
        if obj_id in processed_ids:
            continue

        matching_files = [f for f in os.listdir(os.path.join(cwd, demand_folder)) if f.startswith(obj_id)]

        if not matching_files:
            continue
        
        file = matching_files[0]  # only one match
        
        df_demand = pd.read_csv(os.path.join(cwd, demand_folder, file), parse_dates=True)

        data[Columns.DEMAND]=df_demand

        print("Loaded data keys:", list(data.keys()))

        # Instantiate and configure the generator
        gen = TimeSeriesGenerator(logging_level=logging.WARNING)

        #GEOMA
        obj[Types.OCCUPANCY]=Columns.OCCUPANCY_GEOMA
        obj[Objects.APPLY_NIGHT_SCHEDULE]=True
        obj[Objects.ID]=Columns.OCCUPANCY_GEOMA
        gen.add_objects(obj.to_dict())

        #PHT
        obj[Types.OCCUPANCY]=Columns.OCCUPANCY_PHT
        obj[Objects.APPLY_NIGHT_SCHEDULE]=True
        obj[Objects.ID]=Columns.OCCUPANCY_PHT
        gen.add_objects(obj.to_dict())

        # Generate Occupancy time series
        summary, df = gen.generate(data, workers=1)

        # Print summary
        print(f"Summary occupancy: {obj_id}")
        print(summary)

        for idx,row in summary.iterrows():

            df_summary.loc[len(df_summary)] = {
                Objects.ID:obj_id,
                "method":idx,
                f"average_{Types.OCCUPANCY}":row["average_occupation"]
            }

        df[Columns.OCCUPANCY_GEOMA][Types.OCCUPANCY].to_csv(f"data/validation/{Types.OCCUPANCY}/{Columns.OCCUPANCY_GEOMA}/{obj_id}.csv")
        df[Columns.OCCUPANCY_PHT][Types.OCCUPANCY].to_csv(f"data/validation/{Types.OCCUPANCY}/{Columns.OCCUPANCY_PHT}/{obj_id}.csv")
    
        processed_ids.add(obj_id)
    
    df_summary.to_csv(f"data/validation/{Types.OCCUPANCY}/summary_{Types.OCCUPANCY}.csv",index=False)


def derive_ventilation(obj,data):



    if os.path.exists(f"data/validation/ventilation/{obj[Columns.ID]}.csv"):
        df_ventilation=pd.read_csv(f"data/validation/ventilation/{obj[Columns.ID]}.csv")
        df_ventilation[Columns.DATETIME]=pd.to_datetime(df_ventilation[Columns.DATETIME],
                                                          utc=True,
                                                          errors="coerce").dt.tz_localize(None)
        df_ventilation=df_ventilation.set_index(Columns.DATETIME)


    else:

        ventilation_generator = VentilationTimeSeries()

        df_air_exchange=pd.read_csv(f"data/validation/tipology/ventilation/{str(obj["climate_zone"]).replace(" ","_")}_{obj[Objects.VENTILATION_COL]}.csv")

        df_air_exchange[Columns.DATETIME] = pd.to_datetime(df_air_exchange[Columns.DATETIME],
                                                          utc=True,
                                                          errors="coerce").dt.tz_localize(None)

        df_air_exchange = df_air_exchange.set_index(Columns.DATETIME)

        data[Objects.VENTILATION]=df_air_exchange
        df_ventilation=ventilation_generator.generate(obj,data)

        df_ventilation.to_csv(f"data/validation/ventilation/{obj[Columns.ID]}.csv")

    data[Objects.VENTILATION]=df_ventilation
    
    
    return obj,data

def derive_internal_gains(objects,gains_per_person):

    objects=objects.copy()

    objects=objects.drop(columns=["year",Objects.RESISTANCE,Objects.CAPACITANCE])
    objects=objects.drop_duplicates()

    objects[Objects.GAINS_INTERNAL_PER_PERSON]=gains_per_person #FISCHER NUMBER

    for idx,obj in objects.iterrows():
        data = {}
        obj_id = str(obj["id"])
        inhabitants=obj[Objects.INHABITANTS]

        print(f"Calculating internal gains for building:{obj_id}. {inhabitants} inhabitants.")

        #GEOMA-DERIVED OCCUPANCY
        data[Types.OCCUPANCY]=pd.read_csv(f"{occupancy_folder}/{Columns.OCCUPANCY_GEOMA}/{obj_id}.csv")
        data[Types.OCCUPANCY].set_index(Columns.DATETIME,inplace=True)

        df=InternalOccupancy().generate(obj=obj.to_dict(),data=data)

        df.to_csv(f"data/validation/internal_gains/{Columns.OCCUPANCY_GEOMA}/{obj_id}.csv")

        #PHT-DERIVED OCCUPANCY
        data[Types.OCCUPANCY]=pd.read_csv(f"{occupancy_folder}/{Columns.OCCUPANCY_PHT}/{obj_id}.csv")
        data[Types.OCCUPANCY].set_index(Columns.DATETIME,inplace=True)

        df=InternalOccupancy().generate(obj=obj.to_dict(),data=data)

        df.to_csv(f"data/validation/internal_gains/{Columns.OCCUPANCY_PHT}/{obj_id}.csv")

def derive_solar_gains(objects, shading_factor):
    print("Solar Gains")
    objects = objects.copy()
    solar_gains_generator = SolarGainsPVLib()

    # Month groups
    hot_months = [5, 6, 7, 8, 9, 10]
    cold_months = [1, 2, 3, 4, 11, 12]

    # Base scaling curve for hot months
    x = numpy.linspace(0, numpy.pi, len(hot_months))
    base_curve = 1 - (1 - shading_factor) * numpy.sin(x)

    # Opposite seasonal effect for cold months:
    # If summer reduces gains, winter increases them correspondingly.
    cold_curve = 1 + (1 - base_curve)

    # Climate groups
    colder_zones = {"cold", "very cold"}
    warmer_zones = {"mixed dry", "hot dry", "hot humid"}
    neutral_zones = {"marine"}

    for idx, obj in objects.iterrows():

        cz = str(obj["climate_zone"]).lower()

        print(f"Processing ID:{obj[Columns.ID]}, year:{obj['year']}, CZ:{cz}")

        # -------------------------------------------------------
        # Build climate-zone-dependent curves
        # -------------------------------------------------------
        if shading_factor == 0:
            # No adjustment → all months scale = 1
            hot_final = numpy.ones_like(base_curve)
            cold_final = numpy.ones_like(cold_curve)

        else:
            hot_final = base_curve.copy()
            cold_final = cold_curve.copy()

            if cz in colder_zones:
                # Cold zones → rely more on winter sun, allow more gain year-round
                hot_final *= 1        # even higher summer gain
                cold_final *= 1        # also enhance winter solar contribution

            elif cz in warmer_zones:
                # Warm zones → restrict summer gain, allow winter gain
                hot_final *= 1           # eliminate summer gains entirely
                cold_final *= 1        # strong winter solar contribution allowed

            elif cz in neutral_zones:
                # Marine → mild correction
                hot_final *= 1
                cold_final *= 1

            else:
                print(f"⚠ Unknown climate zone '{cz}'. Using base curve.")

        # Mapping each month to its scale
        hot_scale = dict(zip(hot_months, hot_final))
        cold_scale = dict(zip(cold_months, cold_final))

        # -------------------------------------------------------
        # Load input data
        # -------------------------------------------------------
        data = {}

        df_weather = get_weather_timeseries(obj)
        df_weather.set_index(Columns.DATETIME, inplace=True)
        data[Objects.WEATHER] = df_weather

        df_windows = get_windows(obj)
        data[Objects.WINDOWS] = df_windows

        # -------------------------------------------------------
        # Compute solar gains
        # -------------------------------------------------------
        df_solar_gains = solar_gains_generator.generate(obj=obj, data=data)
        df_solar_gains = df_solar_gains.reset_index()

        # -------------------------------------------------------
        # Apply hot-month scaling
        # -------------------------------------------------------
        mask_hot = df_solar_gains["datetime"].dt.month.isin(hot_months)
        df_solar_gains.loc[mask_hot, Objects.GAINS_SOLAR] *= \
            df_solar_gains.loc[mask_hot, "datetime"].dt.month.map(hot_scale)

        # -------------------------------------------------------
        # Apply cold-month scaling (opposite effect)
        # -------------------------------------------------------
        mask_cold = df_solar_gains["datetime"].dt.month.isin(cold_months)
        df_solar_gains.loc[mask_cold, Objects.GAINS_SOLAR] *= \
            df_solar_gains.loc[mask_cold, "datetime"].dt.month.map(cold_scale)

        # -------------------------------------------------------
        # Save output
        # -------------------------------------------------------
        out_path = f"{solar_gains_folder}/{obj[Objects.ID]}_{obj['year']}.csv"
        df_solar_gains.to_csv(out_path, index=False)

        print(f"✔ Saved: {out_path}")

        

def derive_hvac(objects,capacitance_factor,resistance_factor,ventilation_mode,method:str=Columns.OCCUPANCY_GEOMA):
    print(f"HVAC calculation: {method}")
    
    objects=objects.copy()

    internal_gains_folder="data/validation/internal_gains"
    weather_folder="data/validation/weather"
    hvac_folder="data/validation/hvac"
    real_demand_folder="data/validation/demand"

    objects=objects[[Objects.ID,"year",
                     Objects.INHABITANTS,
                     Objects.RESISTANCE,
                     Objects.CAPACITANCE,
                     Objects.TEMP_MIN,
                     Objects.TEMP_MAX,
                     Objects.FILE,
                     Objects.AREA,
                     "stories",
                     "climate_zone"]]
    
    objects[Types.HVAC]             =   "1r1c"
    objects[Objects.GAINS_INTERNAL] =   Objects.GAINS_INTERNAL
    objects[Objects.GAINS_INTERNAL_COL] = Objects.GAINS_INTERNAL
    objects[Objects.VENTILATION]    =   Objects.VENTILATION
    objects[Objects.VENTILATION_COL] = ventilation_mode
    objects[Objects.WINDOWS]    =   Objects.WINDOWS

    #Initialize missing parameters
    objects[Objects.ACTIVE_COOLING] = True
    objects[Objects.ACTIVE_HEATING] = True
    

    full_index = pd.date_range(start="2018-01-01 00:00:00", end="2018-12-31 23:45:00", freq="15min",name=Columns.DATETIME)

    for idx,obj in tqdm.tqdm(objects.iterrows(), total=len(objects), desc="Processing buildings"):

            data = {}
            obj_id =str(obj[Objects.ID])
            obj_year=obj["year"]

            obj[Objects.TEMP_MAX]=obj[Objects.TEMP_MAX]+273.15
            obj[Objects.TEMP_MIN]=obj[Objects.TEMP_MIN]+273.15
            obj[Objects.TEMP_INIT] = (obj[Objects.TEMP_MAX]+obj[Objects.TEMP_MIN])/2

            obj[Objects.CAPACITANCE]=obj[Objects.CAPACITANCE]*capacitance_factor
            obj[Objects.RESISTANCE]=obj[Objects.RESISTANCE]*resistance_factor

            #Detect power limits
            real_demand=pd.read_csv(f"{real_demand_folder}/{obj_id}.csv")
            obj[Objects.POWER_HEATING]  = detect_upper_limit(real_demand[f"{Types.HEATING}_{Columns.DEMAND}[W]"])
            obj[Objects.POWER_COOLING]  = detect_upper_limit(real_demand[f"{Types.COOLING}_{Columns.DEMAND}[W]"])
            
            
            #print(f"Processing ID:{obj_id}, year:{obj_year}")

            #Add Temperature Setpoints
            #df_setpoints=pd.read_csv(f"{temperature_setpoints_folder}/{obj_id}.csv")
            #df_setpoints[Objects.TEMP_MIN]=df_setpoints[Objects.TEMP_MIN]+273.15
            #df_setpoints[Objects.TEMP_MAX]=df_setpoints[Objects.TEMP_MAX]+273.15
            #data[Objects.TEMP_SETPOINTS]=df_setpoints

            #Add Weather Data
            df_weather=get_weather_timeseries(obj)
            df_weather[Columns.TEMP_AIR]=df_weather[Columns.TEMP_AIR]+273.15
            data[Objects.WEATHER]=df_weather

            #Add Solar Gains Data
            df_solar_gains=pd.read_csv(f"{solar_gains_folder}/{obj_id}_{obj_year}.csv")
            df_solar_gains=set_datetime_index(df_solar_gains)
            data[Objects.GAINS_SOLAR]=df_solar_gains

            #Add Internal Gains Data
            df_internal_gains=pd.read_csv(f"{internal_gains_folder}/{method}/{obj_id}.csv")
            df_internal_gains=set_datetime_index(df_internal_gains)
            data[Objects.GAINS_INTERNAL]=df_internal_gains          

            #Add Ventilation
            obj,data=derive_ventilation(obj,data)

            # Generate HVAC time series
            temp_in,p_heat,p_cool=calculate_timeseries(obj=obj,data=data)
            df_hvac = pd.DataFrame(
                {
                    Columns.TEMP_IN: temp_in,
                    f"{Types.HEATING}_{Columns.DEMAND}[W]": p_heat,
                    f"{Types.COOLING}_{Columns.DEMAND}[W]": p_cool,
                },
                    index=full_index
                )

            df_hvac.reset_index(inplace=True)
            df_hvac[Columns.TEMP_IN]=df_hvac[Columns.TEMP_IN]-273.15
            df_hvac.to_csv(f"{hvac_folder}/{method}/{obj_id}_{obj_year}.csv",index=False)

def summarize_hvac(objects,method:str=Columns.OCCUPANCY_GEOMA):
    objects=objects.copy()
    df_summary=pd.DataFrame(columns=[
    Objects.ID,
    "year",
    "climate_zone",
    "state",
    Objects.INHABITANTS,
    Objects.RESISTANCE,
    Objects.CAPACITANCE,
    Objects.AREA,
    "stories",
    "method",
    Objects.LAT,
    Objects.LON,
    Objects.TEMP_MAX,
    Objects.TEMP_MIN
    ])

    for idx,obj in tqdm.tqdm(objects.iterrows(), total=len(objects), desc="Summaryzing HVAC"):
        
        
        #print(f"Processing ID: {obj[Columns.ID]}, year:{obj["year"]}")

        obj_id =str(obj[Objects.ID])
        obj_year=obj["year"]
    
        
        obj_stories=obj["stories"]
        obj_state=obj["state"]
        obj_climate_zone=obj["climate_zone"]
        obj_area=obj[Objects.AREA]
        obj_filename=obj[Objects.FILE]
        obj_inhabitants=obj[Objects.INHABITANTS]
        obj_resistance=obj[Objects.RESISTANCE]
        obj_capacitance=obj[Objects.CAPACITANCE]
        obj_latitude=   obj[Objects.LAT]
        obj_longitude=obj[Objects.LON]
        obj_temp_max=obj[Objects.TEMP_MAX]
        obj_temp_min=obj[Objects.TEMP_MIN]

        df_hvac=pd.read_csv(f"{hvac_folder}/{method}/{obj_id}_{obj_year}.csv")
        df_hvac_real=pd.read_csv(f"{demand_folder}/{obj_id}.csv")

        

        df_summary.at[idx,Objects.ID] = obj_id
        df_summary.at[idx, "method"] = method 
        df_summary.at[idx,"year"] = obj_year
        df_summary.at[idx,"stories"]=obj_stories
        df_summary.at[idx,"state"]=obj_state
        df_summary.at[idx,"climate_zone"]=obj_climate_zone
        df_summary.at[idx,Objects.AREA]=obj_area
        df_summary.at[idx,Objects.FILE]=obj_filename
        df_summary.at[idx,Objects.RESISTANCE]=obj_resistance
        df_summary.at[idx,Objects.CAPACITANCE]=obj_capacitance
        df_summary.at[idx,Objects.INHABITANTS]=obj_inhabitants
        df_summary.at[idx,Objects.LAT]=obj_latitude
        df_summary.at[idx,Objects.LON]=obj_longitude
        df_summary.at[idx,Objects.TEMP_MAX]=obj_temp_max
        df_summary.at[idx,Objects.TEMP_MIN]=obj_temp_min

        df_summary.at[idx,f"{Types.HEATING}_{Columns.DEMAND}_simulated[kWh]"]   = round((df_hvac[f"{Types.HEATING}_{Columns.DEMAND}[W]"].sum()/4)/1000,1)
        df_summary.at[idx,f"{Types.HEATING}_{Columns.DEMAND}_real[kWh]"]        = round((df_hvac_real[f"{Types.HEATING}_{Columns.DEMAND}[W]"].sum()/4)/1000,1)

        df_summary.at[idx,f"{Types.COOLING}_{Columns.DEMAND}_simulated[kWh]"]   = round((df_hvac[f"{Types.COOLING}_{Columns.DEMAND}[W]"].sum()/4)/1000,1)
        df_summary.at[idx,f"{Types.COOLING}_{Columns.DEMAND}_real[kWh]"]        = round((df_hvac_real[f"{Types.COOLING}_{Columns.DEMAND}[W]"].sum()/4)/1000,1)
        
        df_summary.at[idx,f"{Types.HEATING}_{Columns.DEMAND}_simulated[kWh]/{Objects.AREA}"]=   round(((df_hvac[f"{Types.HEATING}_{Columns.DEMAND}[W]"].sum()/4)/1000)/obj_area,1)
        df_summary.at[idx,f"{Types.HEATING}_{Columns.DEMAND}_real[kWh]/{Objects.AREA}"]     =   round((( df_hvac_real[f"{Types.HEATING}_{Columns.DEMAND}[W]"].sum()/4)/1000)/obj_area,1)
        
        df_summary.at[idx,f"{Types.COOLING}_{Columns.DEMAND}_simulated[kWh]/{Objects.AREA}"]=   round(((df_hvac[f"{Types.COOLING}_{Columns.DEMAND}[W]"].sum()/4)/1000)/obj_area,1)
        df_summary.at[idx,f"{Types.COOLING}_{Columns.DEMAND}_real[kWh]/{Objects.AREA}"]     =   round(((df_hvac_real[f"{Types.COOLING}_{Columns.DEMAND}[W]"].sum()/4)/1000)/obj_area,1)

        
    #df_summary.to_csv(f"results/hvac_summary_{method}.csv",index=False)
    return df_summary


def get_weather_timeseries(obj):
    weather_folder="data/validation/weather"
    df_weather=pd.read_csv(f"{weather_folder}/cleaned/{obj["climate_zone"]}.csv")
    df_weather[Columns.DATETIME]=pd.to_datetime(df_weather[Columns.DATETIME])
    return df_weather

def format_weather_timeseries():
    raw_weather_folder=os.path.join(weather_folder,"raw")
    files=os.listdir(raw_weather_folder)

    for file in files:
        df=df=pd.read_csv(f"{raw_weather_folder}/{file}")
        df.columns = df.columns.str.strip()
        df=df.rename(columns={"date_time":                          Columns.DATETIME,
                           "Dry Bulb Temperature [°C]":             Columns.TEMP_AIR,
                           "Global Horizontal Radiation [W/m2]":    Columns.SOLAR_GHI,
                           "Direct Normal Radiation [W/m2]":        Columns.SOLAR_DNI,
                           "Diffuse Horizontal Radiation [W/m2]":   Columns.SOLAR_DHI})
        df=df[[Columns.DATETIME,
                Columns.TEMP_AIR,
                Columns.SOLAR_GHI,
                Columns.SOLAR_DNI,
                Columns.SOLAR_DHI]]
        
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

        df=df.set_index(Columns.DATETIME)
        df=df.resample("15min").interpolate('linear')
        df=df.reset_index()
        df.to_csv(f"{weather_folder}/cleaned/{file}",index=False)

def get_windows(obj):
    df_windows=pd.read_csv(f"data/validation/tipology/windows.csv")
    mask = (
        (df_windows["year"].astype(int) == int(obj["year"])) &
        (df_windows[Objects.ID].astype(str) == str(obj[Objects.ID]))
    )
    df_windows = df_windows.loc[mask]   
    return df_windows


def calculate_dni(df_weather: pd.DataFrame, object,method):

    solpos = pvlib.solarposition.get_solarposition(
        time=df_weather[Columns.DATETIME],
        latitude=object[Objects.LAT],
        longitude=object[Objects.LON]
    )
    #solpos = solpos.reindex(df_weather.index)
    zenith=solpos["zenith"]

    location=pvlib.location.Location(latitude=object[Objects.LAT],longitude=object[Objects.LON])

    df_weather[Columns.DATETIME] = pd.to_datetime(df_weather[Columns.DATETIME])
    df_weather.set_index(Columns.DATETIME, inplace=True)

    df_clearsky=location.get_clearsky(times=df_weather.index)

    df_clearsky = df_clearsky.reindex(df_weather.index)
    dni_clear = df_clearsky["dni"]

    dhi=df_weather[Columns.SOLAR_DHI]
    ghi=df_weather[Columns.SOLAR_GHI]
    
    match method:
        case "zeros":        
            df_weather[Columns.SOLAR_DNI]=0.0
        
        case "pvlib":
            ghi=ghi.align(dhi, join="inner")[0]
            dhi=dhi.align(ghi, join="inner")[0]
            zenith=zenith.align(ghi, join="inner")[0]
            dni_clear=dni_clear.reindex_like(ghi)
            dni = pvlib.irradiance.dni(
                ghi=ghi,
                dhi=dhi,
                zenith=zenith,
                dni_clear=dni_clear,
                clearsky_tolerance=1.1
            )
            dni = dni.clip(lower=0).fillna(0.0)
            df_weather[Columns.SOLAR_DNI]=dni

        case "erbs":
            df=pvlib.irradiance.erbs(ghi,zenith,df_weather.index)
            df_weather[Columns.SOLAR_DNI]=df["dni"]

        case "clear_sky":
            df_weather[Columns.SOLAR_DNI]=dni_clear

    df_weather=df_weather.reset_index()

    return df_weather

def calculate_dhi(df_weather: pd.DataFrame, object):

    solpos = pvlib.solarposition.get_solarposition(
        time=df_weather[Columns.DATETIME],
        latitude=object[Objects.LAT],
        longitude=object[Objects.LON]
    )

    zenith=solpos["zenith"]
    zenith=zenith.clip(upper=90)
    
    location=pvlib.location.Location(latitude=object[Objects.LAT],longitude=object[Objects.LON])

    df_weather[Columns.DATETIME] = pd.to_datetime(df_weather[Columns.DATETIME])
    df_weather.set_index(Columns.DATETIME, inplace=True)

    df_clearsky=location.get_clearsky(times=df_weather.index)

    df_clearsky = df_clearsky.reindex(df_weather.index)
    dni=df_weather[Columns.SOLAR_DNI]
    ghi=df_weather[Columns.SOLAR_GHI]

    ghi_fraction=numpy.cos(numpy.radians(zenith))

    dhi = ghi - dni*ghi_fraction

    dhi = dhi.clip(lower=0).fillna(0.0)
    
    df_weather[Columns.SOLAR_DHI]=dhi
    df_weather["zenith[°]"]=zenith
    
    df_weather=df_weather.reset_index()
    
    df_weather.to_csv(f"{weather_folder}/cleaned/{object["climate_zone"]}.csv",index=False)
    

    return df_weather



def set_datetime_index(df):
    df[Columns.DATETIME]=pd.to_datetime(df[Columns.DATETIME])
    df.set_index(Columns.DATETIME,inplace=True)
    return df

def get_real_demand_files(objects:pd.DataFrame):

    objects = objects.drop_duplicates(subset="id", keep="first")
    cz_folders=os.listdir(real_demand_folder)

    for idx,obj in objects.iterrows():
        

        cz_code=str(obj["filename"]).split("_")[0]
        cz_folder=next((f for f in cz_folders if f.split("_")[0] == cz_code), None)
        real_demand=pd.read_csv(f"{real_demand_folder}/{cz_folder}/households/{obj[Columns.ID]}/{obj[Columns.ID]}_timeseries_adjusted.csv")
        
        print(f"Processing ID: {obj[Columns.ID]}")
        
        try:
            raw_demand_parquet=pd.read_parquet(f"{real_demand_folder}/{cz_folder}/households/{obj[Columns.ID]}/{obj[Columns.ID]}_timeseries.parquet")
        except Exception as e:
                raw_demand_parquet=pd.read_parquet(f"{real_demand_folder}/{cz_folder}/households/{obj[Columns.ID]}/{obj[Columns.ID]}-10.parquet")
        raw_demand_parquet=raw_demand_parquet[["timestamp",
                                               "out.electricity.total.energy_consumption",
                                               "out.electricity.cooling_fans_pumps.energy_consumption",
                                               "out.electricity.cooling.energy_consumption"]]
        raw_demand_parquet[f"{Types.COOLING}_{Columns.DEMAND}[W]"] = (raw_demand_parquet["out.electricity.cooling_fans_pumps.energy_consumption"]+raw_demand_parquet["out.electricity.cooling.energy_consumption"])*4000
        raw_demand_parquet=raw_demand_parquet.rename(columns={"timestamp":Columns.DATETIME,
                                                              "out.electricity.total.energy_consumption":f"total_{Columns.POWER}"})
        
        #Turn into W from kWh for 15' resolution readings
        raw_demand_parquet[f"total_{Columns.POWER}"]=raw_demand_parquet[f"total_{Columns.POWER}"]*4000
        
        raw_demand_parquet[Columns.DATETIME]=pd.to_datetime(raw_demand_parquet[Columns.DATETIME], errors="coerce")

        raw_demand_parquet=raw_demand_parquet[[Columns.DATETIME,
                                                f"total_{Columns.POWER}",
                                                f"{Types.COOLING}_{Columns.DEMAND}[W]"]]
        
        real_demand=real_demand.rename(columns={"heating":f"{Types.HEATING}_{Columns.DEMAND}[W]",
                               "cooling":f"{Types.COOLING}_{Columns.DEMAND}[W]",
                               "timestamp":Columns.DATETIME,
                               "power":f"{Columns.POWER}"})
        
        real_demand=real_demand[[Columns.DATETIME,
                                 f"{Columns.POWER}",
                                 f"{Types.HEATING}_{Columns.DEMAND}[W]"]]
        
        real_demand[Columns.DATETIME]=pd.to_datetime(real_demand[Columns.DATETIME], errors="coerce")

        real_demand=real_demand.merge(
            raw_demand_parquet,
            how="outer",
            on=Columns.DATETIME
        )

        real_demand[f"total_{Columns.POWER}"]=real_demand[f"total_{Columns.POWER}"].bfill()
        real_demand[f"{Types.COOLING}_{Columns.DEMAND}[W]"]=real_demand[f"{Types.COOLING}_{Columns.DEMAND}[W]"].bfill()

        lower_heating,upper_heating = real_demand[f"{Types.HEATING}_{Columns.DEMAND}[W]"].quantile([0.05, 0.999])
        real_demand[f"{Types.HEATING}_{Columns.DEMAND}[W]"] =real_demand[f"{Types.HEATING}_{Columns.DEMAND}[W]"].clip(upper=upper_heating) 
        real_demand.drop(real_demand.tail(1).index,inplace=True)


        #REMOVE EXCESIVELY HIGH VALUES
        clip_values = {
            71009   :   6000,
            130744  :   8000,
            332001  :   8000,
            418445  :   20000,
            482749  :   7000,
            525662  :   6000,
            86716   :   10000,
            87413   :   8000,
            130744  :   10000,
            196069  :   10000,
            234774  :   10500,
            255488  :   4000,
            255879  :   4000,
            265047  :   10000,
            311843  :   10000,
            332001  :   8000,
            394422  :   25000,
            442855  :   3000,
            482749  :   8000,
            488971  :   6000,
            494227  :   6000,
            491566  :   4000,
            289249  :   20000,
            81487   :   40000,
            196069  :   8000,
            492842  :   8000,
            394422  :   20000,

        }

        
        col = f"{Types.HEATING}_{Columns.DEMAND}[W]"
        limit = clip_values.get(obj[Columns.ID])

        if limit is not None:
            series = real_demand[col].copy()

            # Find indices where the value exceeds the limit
            high_idx = series[series > limit].index

            if len(high_idx) > 0:
                print(f"⚠️ Replacing high values for building ID {obj[Columns.ID]}")

                for idx in high_idx:
                    # Find the last previous value that is below the limit
                    prev_valid = series.loc[:idx][series.loc[:idx] < limit]

                    if not prev_valid.empty:
                        replacement_val = prev_valid.iloc[-1]
                    else:
                        # If no previous value is valid, keep limit or NaN
                        replacement_val = limit

                    series.loc[idx] = replacement_val

                real_demand[col] = series
        real_demand.to_csv(f"{demand_folder}/{obj[Columns.ID]}.csv",index=False)


def calculate_fit_score(df,method=Columns.OCCUPANCY_GEOMA,name=""):
    #df=pd.read_csv(f"results/hvac_summary_{method}.csv")

    df["heating_demand_error"] = round((df["heating_demand_simulated[kWh]"] - df["heating_demand_real[kWh]"]).abs(),2)
    #df["heating_load_error"]   = (df["heating_load_simulated_max[kW]"] - df["heating_load_real_max[kW]"]).abs()
    df["cooling_demand_error"] = round((df["cooling_demand_simulated[kWh]"] - df["cooling_demand_real[kWh]"]).abs(),2)
    #df["cooling_load_error"]   = (df["cooling_load_simulated_max[kW]"] - df["cooling_load_real_max[kW]"]).abs()

    df["heating_demand_rel_error"] = round(df["heating_demand_error"] / df["heating_demand_real[kWh]"].replace(0, numpy.nan),2)
    #df["heating_load_rel_error"]   = df["heating_load_error"] / df["heating_load_real_max[kW]"].replace(0, numpy.nan)
    df["cooling_demand_rel_error"] = round(df["cooling_demand_error"] / df["cooling_demand_real[kWh]"].replace(0, numpy.nan),2)
    #df["cooling_load_rel_error"]   = df["cooling_load_error"] / df["cooling_load_real_max[kW]"].replace(0, numpy.nan)

    df["fit_score"] = df[[
    "heating_demand_rel_error",
    #"heating_load_rel_error",
    "cooling_demand_rel_error",
    #"cooling_load_rel_error"
    ]].mean(axis=1)

    df = df.sort_values("fit_score", ascending=True)
    
    df_best_fit_score= (
    df.groupby("id", as_index=False)
      .apply(lambda g: g.loc[g["fit_score"].idxmin()])
      .reset_index(drop=True)
    )

    df_best_heating_score= (
    df.groupby("id", as_index=False)
      .apply(lambda g: g.loc[g["heating_demand_rel_error"].idxmin()])
      .reset_index(drop=True)
    )

    df_best_cooling_score=(
    df.groupby("id", as_index=False)
      .apply(lambda g: g.loc[g["cooling_demand_rel_error"].idxmin()])
      .reset_index(drop=True)
    )
    
    df_best_fit_score=df_best_fit_score.sort_values(by="fit_score", ascending=True)
    df_best_fit_score.to_csv(f"results/best_fit_score_{method}_{name}.csv",index=False)
    df_best_heating_score.to_csv(f"results/best_heating_score_{method}_{name}.csv",index=False)
    df_best_cooling_score.to_csv(f"results/best_cooling_score_{method}_{name}.csv",index=False)

    fit_score=df["fit_score"].iloc[0]
    heating_demand_error=df["heating_demand_rel_error"].iloc[0]
    cooling_demand_error=df["cooling_demand_rel_error"].iloc[0]

    print(f"Best fit for this iteration: {fit_score}")
    return df_best_fit_score


def detect_upper_limit(s: pd.Series) -> float:
    # Ignore empty or all-zero series
    if s.empty or s.max() == 0:
        return numpy.inf

    max_val = s.max()
    mask = (s == max_val).astype(int)

    # Detect if any two consecutive readings are at the max value
    if ((mask.shift(1) == 1) & (mask == 1)).any():
        return max_val
    else:
        return numpy.inf
    
def add_typical_open_ventilation():
    
    df=pd.read_csv(f"data/validation/tipology/ventilation.csv")

    df["datetime"]=pd.to_datetime(df["datetime"],errors="coerce")

    # Define your cooling season (adjust dates as needed)
    cooling_start = pd.Timestamp("2018-05-01", tz="UTC")
    cooling_end = pd.Timestamp("2018-09-30", tz="UTC")

    # Copy the 'typical' column
    df["typical_open"] = df["typical"]

    # During the cooling season, set windows "open 24/7"
    # → For example, increase the ventilation fraction to 1.0 (fully open)
    mask = (df["datetime"] >= cooling_start) & (df["datetime"] <= cooling_end)
    df.loc[mask, "typical_open"] = 1.0

    df.to_csv("data/validation/tipology/ventilation.csv",index=False)


def melt_best_results(df_best:pd.DataFrame):

    demand_folder="data/validation/demand"
    hvac_folder=f"data/validation/hvac"
    dfs=[]
    for idx,obj in df_best.iterrows():

        obj_id =str(obj[Objects.ID])
        obj_year=obj["year"]
        obj_climate_zone=obj["climate_zone"]
        obj_lat=obj[Objects.LAT]
        obj_lon=obj[Objects.LON]
        obj_area=obj[Objects.AREA]
        obj_inhabitants=obj[Objects.INHABITANTS]

        df_hvac=pd.read_csv(f"{hvac_folder}/{Columns.OCCUPANCY_GEOMA}/{obj_id}_{obj_year}.csv")
        df_hvac_real=pd.read_csv(f"{demand_folder}/{obj_id}.csv")

        df_hvac[Columns.DATETIME]=pd.to_datetime(df_hvac[Columns.DATETIME])
        df_hvac_real[Columns.DATETIME]=pd.to_datetime(df_hvac_real[Columns.DATETIME])


        df_hvac_real=df_hvac_real.rename(columns={f"{Types.HEATING}_{Columns.DEMAND}[W]":f"real_{Types.HEATING}_{Columns.DEMAND}[W]"})
        df_hvac_real=df_hvac_real.rename(columns={f"{Types.COOLING}_{Columns.DEMAND}[W]":f"real_{Types.COOLING}_{Columns.DEMAND}[W]"})
        df_hvac_real=df_hvac_real[[Columns.DATETIME,f"real_{Types.HEATING}_{Columns.DEMAND}[W]",f"real_{Types.COOLING}_{Columns.DEMAND}[W]"]]



        df_hvac=df_hvac.rename(columns={f"{Types.HEATING}_{Columns.DEMAND}[W]":f"simulated_{Types.HEATING}_{Columns.DEMAND}[W]"})
        df_hvac=df_hvac.rename(columns={f"{Types.COOLING}_{Columns.DEMAND}[W]":f"simulated_{Types.COOLING}_{Columns.DEMAND}[W]"})
        df_hvac=df_hvac[[Columns.DATETIME,f"simulated_{Types.HEATING}_{Columns.DEMAND}[W]",f"simulated_{Types.COOLING}_{Columns.DEMAND}[W]"]]


        df=pd.merge(df_hvac,df_hvac_real,on=Columns.DATETIME, how="outer")

        df[Objects.ID]=obj_id

        df["year"]=obj_year

        df["climate_zone"]=obj_climate_zone
        df[Objects.LAT]=obj_lat
        df[Objects.LON]=obj_lon
        df[Objects.AREA]=obj_area
        df[Objects.INHABITANTS]=obj_inhabitants

        df=df.set_index(Columns.DATETIME)

        dfs.append(df)

    output_df=pd.concat(dfs, axis=0)
    output_df=output_df.reset_index()
    output_df.to_csv("all_hvac_objects.csv",index=False)
    return output_df


def add_average_detected_occupancy(method=Columns.OCCUPANCY_GEOMA):
    df=pd.read_csv("data/validation/objects_entise.csv")
    df_occupancy=pd.read_csv("data/validation/occupancy/summary_occupancy.csv")
    df_occupancy=df_occupancy.loc[df_occupancy["method"]==method]
    df["average_occupancy"]=0
    for idx,row in df.iterrows():
        avg_occupancy=df_occupancy.loc[(df_occupancy["id"]==row["id"]),"average_occupancy"].mean()
        df.at[idx,"average_occupancy"]=avg_occupancy
    df.to_csv("data/validation/objects_entise_occupancy.csv",index=False)    

add_average_detected_occupancy()