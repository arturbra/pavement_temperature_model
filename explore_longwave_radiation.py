import pandas as pd
import numpy as np


def calculate_saturation_vapor_pressure(T):
    """
    Calculate saturation vapor pressure using the Tetens formula.
    T is in Celsius.
    e_sat = 6.11 * 10 ** ((7.5 * T_air) / (T_air + 237.3)) * 100 [Pa]
    e     = 6.11 * 10 ** ((7.5 * T_dew) / (T_dew + 237.3)) * 100 [Pa]
    NEW: Converts the result to Pascals.
    """
    return 6.112 * np.exp((17.67 * T) / (T + 243.5)) * 100  # converting from mba to Pa


def calculate_longwave_radiation_new_method(df):
    T_air = df['AirTemperature']
    T_air_K = T_air + 273.15

    e_s = np.array([calculate_saturation_vapor_pressure(T) for T in T_air])
    e_a = (df['RelativeHumidity'] * e_s) / 1000 # converting from Pa to kPa
    emissivity_clear_sky = 1.24 * ((e_a / T_air_K) ** (1/7))

    z = 302.7 # site elevation (m)
    Ra = df['TOA']
    Rso = (0.75 + ((2 * 10 ** (-5)) * z)) * Ra
    Rs = df['SolarRadiation']

    tiny = 1e-6
    kt = np.where(Rso > tiny, Rs / np.maximum(Rso, tiny), 0.0)
    kt = np.clip(kt, 0.0, 1.0)

    emissivity_sky = emissivity_clear_sky * (1 + 0.22 * ((1 - kt) ** 2))
    emissivity_sky = np.minimum(emissivity_sky, 1)
    sigma_const = 5.67e-8
    longwave = sigma_const * emissivity_sky * (T_air_K ** 4)

    return longwave

def calculate_longwave_radiation_old_method(df):
    T_air = df['AirTemperature']
    T_air_K = T_air + 273.15

    e_s = np.array([calculate_saturation_vapor_pressure(T) for T in T_air])
    e_a = (df['RelativeHumidity'] * e_s) / 100 # converting from Pa to mbar

    CR = df['CloudCoverage']
    sigma_const = 5.67e-8

    longwave = sigma_const * (CR + 0.67 * (1 - CR) * (e_a ** 0.08)) * (T_air_K ** 4)
    return longwave



df = pd.read_csv(r"C:\Users\Artur\PycharmProjects\pavement_temperature_model\input_data\1h\open-meteo_data.csv")
longwave_new = calculate_longwave_radiation_new_method(df)
longwave_old = calculate_longwave_radiation_old_method(df)