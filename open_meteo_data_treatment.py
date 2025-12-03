import pandas as pd
import numpy as np


def calculate_saturation_vapor_pressure(T):
    return 6.112 * np.exp((17.67 * T) / (T + 243.5))


def calculate_rlds(df):
    CR = df['cloud_cover (%)'] / 100
    T_air = df['temperature_2m (°C)']
    T_air_K = T_air + 273.15
    RH = df['relative_humidity_2m (%)'] / 100

    sigma_const = 5.67e-8

    e_s = np.array([calculate_saturation_vapor_pressure(T) for T in T_air])
    e_a = e_s * RH
    rlds = sigma_const * (CR + 0.67 * (1 - CR) * (e_a ** 0.08)) * (T_air_K ** 4)
    return rlds


def calculate_tdew(T_air, RH):
    """
    Calculates the dew point temperature based on the second order fit (Equation 21 of Lawrence, 2005)
    """
    T_air_k = T_air + 273.15
    RH = RH * 100
    T_dew = T_air - ((100 - RH) / 5) * ((T_air_k / 300) ** 2) - (0.00135 * (RH - 84) ** 2) + 0.35
    return T_dew


df = pd.read_csv(r"C:\Users\Artur\PycharmProjects\pavement_temperature_model\climate_change\open-meteo-29.63N98.45W304m.csv", parse_dates=['time'])
rlds = calculate_rlds(df)
df['rlds'] = rlds

df['time'] = df['time'].dt.tz_localize('America/Chicago', ambiguous='NaT', nonexistent='shift_forward')


# df.to_csv(r"C:\Users\Artur\PycharmProjects\pavement_temperature_model\climate_change\open_meteo_data.csv", index=False)


cc_file = r"C:\Users\Artur\PycharmProjects\pavement_temperature_model\climate_change\climate_change_ssp585_compare.csv"

df = pd.read_csv(cc_file, parse_dates=['date'])
T_air = df['tas']
RH = df['hurs'] / 100

t_dew = calculate_tdew(T_air, RH)
df['dew'] = t_dew

df.to_csv(cc_file, index=False)

