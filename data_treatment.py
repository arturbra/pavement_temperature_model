import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def compute_valid_temperature(row):
    temp1 = row['temperature_1']
    temp2 = row['temperature_2']

    if temp1 < 0 and temp2 < -0:
        return np.nan  # Both are invalid
    elif temp1 < -0:
        return temp2  # Only temp1 is invalid
    elif temp2 < -0:
        return temp1  # Only temp2 is invalid
    else:
        return (temp1 + temp2) / 2  # Average if both are valid


sdf_path = r"C:\Users\artur\OneDrive\Doutorado\UTSA\PP\Final Report\data\level\pp_data_level_2.csv"
sdf = pd.read_csv(sdf_path, parse_dates=[0], low_memory=False)
sdf['date'] = sdf['date'].dt.tz_localize('Etc/GMT+6')

T_obs = sdf[['date', 'box_da_temperature_1', 'box_da_temperature_2']]
T_obs.columns = ['date', 'temperature_1', 'temperature_2']
T_obs = T_obs.resample('1h', on='date').mean()
T_obs['PavementTemperature'] = T_obs.apply(compute_valid_temperature, axis=1)
T_obs.drop(columns=['temperature_1', 'temperature_2'], inplace=True)
T_obs = T_obs.dropna().reset_index()
T_obs['PavementTemperature'] = (T_obs['PavementTemperature'] - 32) / 1.8

meteo_file = r"C:\Users\Artur\OneDrive\Doutorado\UTSA\PP\PPPaper_3\data\open-meteo-29.63N98.45W302m.csv"
df = pd.read_csv(meteo_file, skiprows=2)
df['time'] = pd.to_datetime(df['time'], utc=True)
df['time'] = df['time'].dt.tz_convert('Etc/GMT+6')
df.columns = ['date', 'AirTemperature', 'RelativeHumidity', 'DewPoint', 'CloudCover', 'WindSpeed']
df['RelativeHumidity'] = df['RelativeHumidity'] / 100
df['CloudCover'] = df['CloudCover'] / 100

merged_df = df.merge(T_obs, on='date', how='inner')
merged_df.to_csv('input_data/permeable_asphalt.csv', index=False)


