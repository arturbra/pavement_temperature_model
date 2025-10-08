import os
import pandas as pd
import numpy as np
import temperature_model
import matplotlib.pyplot as plt


def aggregate_ISIMIP_data(folder_path, ssp):
    parameters = ['hurs', 'pr', 'ps', 'rlds', 'rsds', 'sfcwind', 'tas']
    combined_df = None
    for p in parameters:
        input_file = os.path.join(folder_path, f"{p}_GFDL-ESM4_{ssp}_2015-2050.csv")
        df = pd.read_csv(input_file, sep='\t', header=None)
        df.columns = ['year', 'month', 'day', 'hour', p]

        # Create datetime and convert to UTC-6
        df['datetime_utc'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
        df['datetime_utc'] = df['datetime_utc'].dt.tz_localize('UTC')
        df['date'] = df['datetime_utc'].dt.tz_convert('Etc/GMT+6')

        # Keep only the relevant columns
        df = df[['date', p]]

        # Merge all parameters into one DataFrame
        if combined_df is None:
            combined_df = df
        else:
            combined_df = pd.merge(combined_df, df, on='date', how='outer')

    # Optional: sort by date
    combined_df = combined_df.sort_values('date').reset_index(drop=True)
    return combined_df


def saturation_vapor_pressure(T_C):
    """Saturation vapor pressure in kPa using Tetens equation"""
    return 0.6108 * np.exp((17.27 * T_C) / (T_C + 237.3))

def ambient_vapor_pressure(T_C, RH):
    """Actual vapor pressure e_a from air temperature and RH (%)"""
    es = saturation_vapor_pressure(T_C)
    return (RH / 100.0) * es

def calculate_h_li(emissivity, sigma, CR, e_a, T_K):
    """Longwave radiation based on cloud cover, vapor pressure, and air temp"""
    return emissivity * sigma * (CR + 0.67 * (1 - CR) * (e_a ** 0.08)) * (T_K ** 4)


def climate_df_to_input_file(cc_file):
    climate_df = pd.read_csv(cc_file)

    # ---------- FINAL COLUMN NAMES ----------
    climate_df.rename(columns={
        'tas': 'AirTemperature',
        'hurs': 'RelativeHumidity',
        'pr': 'Rainfall',
        'rsds': 'SolarRadiation',
        'sfcwind': 'WindSpeed'
    }, inplace=True)

    climate_df['RelativeHumidity'] = climate_df['RelativeHumidity'] / 100
    climate_df['DewPoint'] = climate_df['AirTemperature'] - ((100 - climate_df['RelativeHumidity']) / 5)
    climate_df['CloudCoverage'] = 0.5
    return climate_df


cc_file_585 = r"C:\Users\Artur\OneDrive\Doutorado\UTSA\PP\PPPaper_3\data\climate_change\climate_change_ssp585.csv"
cc_file_126 = r"C:\Users\Artur\OneDrive\Doutorado\UTSA\PP\PPPaper_3\data\climate_change\climate_change_ssp126.csv"

climate_df_ssp585 = climate_df_to_input_file(cc_file_585)
climate_df_ssp126 = climate_df_to_input_file(cc_file_126)


parameters_file = r"C:\Users\Artur\PycharmProjects\pavement_temperature_model\input_data\1h\parameters_CP.ini"

model_results_ssp585 = temperature_model.model_pavement_temperature(climate_df_ssp585, parameters_file)
model_results_ssp126 = temperature_model.model_pavement_temperature(climate_df_ssp126, parameters_file)

df1 = model_results_ssp585.set_index('date')
df_zoom1 = df1['2026-01-01':'2027-01-01']

df2 = model_results_ssp126.set_index('date')
df_zoom2 = df2['2026-01-01':'2027-01-01']


plt.figure(figsize=(12,5))
plt.plot(df_zoom1.index, df_zoom1['surface_temp'], label='Surface Water Temp SSP585', linewidth=2)
plt.plot(df_zoom2.index, df_zoom2['surface_temp'], label='Surface Water Temp SSP126', linewidth=2)
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.title('Water Temperatures (Daily Average)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()