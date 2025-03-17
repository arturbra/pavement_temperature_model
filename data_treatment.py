import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def compute_valid_temperature(row, method='average'):
    """
    Compute a valid temperature based on a chosen method.

    Parameters:
      row (pd.Series): A row with 'temperature_1' and 'temperature_2'.
      method (str): 'average' (default) will perform the full check and average
                    the two temperatures if both are valid; 'temp1' or 'temp2'
                    will simply select that temperature.

    Returns:
      float: The chosen temperature value.
    """
    temp1 = row['temperature_1']
    temp2 = row['temperature_2']

    if method == 'temp1':
        return temp1
    elif method == 'temp2':
        return temp2
    elif method == 'average':
        # If both temperatures are below 0, consider them invalid.
        if temp1 < 0 and temp2 < 0:
            return np.nan
        elif temp1 < 0:
            return temp2
        elif temp2 < 0:
            return temp1
        # If the difference is greater than 10, return the maximum value.
        elif abs(temp1 - temp2) > 10:
            return max(temp1, temp2)
        # Otherwise, use the average of both temperatures.
        else:
            return (temp1 + temp2) / 2
    else:
        raise ValueError("Invalid method. Choose 'average', 'temp1', or 'temp2'.")


def pavement_temperature_data(sdf, pavement, method='average'):
    """
    Extract and process pavement temperature data.

    Parameters:
      sdf (pd.DataFrame): The pavement data frame.
      pavement (str): One of "CP", "PICP", "PGr", "PA", or "PC".
      method (str): Method for selecting the temperature. Options are:
                    'average' (default), 'temp1', or 'temp2'.

    Returns:
      pd.DataFrame: Dataframe with datetime and the processed pavement temperature.
    """
    pavements = {"CP": "box_a", "PICP": "box_b", "PGr": "box_c", "PA": "box_da", "PC": "box_dc"}
    box = pavements[pavement]

    # Localize the date column
    sdf['date'] = sdf['date'].dt.tz_localize('Etc/GMT+6')

    # Select the two temperature columns for the given box
    T_obs = sdf[['date', f'{box}_temperature_1', f'{box}_temperature_2']]
    T_obs.columns = ['date', 'temperature_1', 'temperature_2']

    # Resample the data to hourly averages
    T_obs = T_obs.resample('1h', on='date').mean()

    # Apply the temperature selection function with the chosen method
    T_obs['PavementTemperature'] = T_obs.apply(
        lambda row: compute_valid_temperature(row, method=method), axis=1
    )

    # Clean up the dataframe and convert to Celsius if needed
    T_obs.drop(columns=['temperature_1', 'temperature_2'], inplace=True)
    T_obs = T_obs.dropna().reset_index()
    T_obs['PavementTemperature'] = (T_obs['PavementTemperature'] - 32) / 1.8
    return T_obs


def open_meteo_data(OM_file):
    OM = pd.read_csv(OM_file, skiprows=2)
    OM['time'] = pd.to_datetime(OM['time'], utc=True)
    OM['time'] = OM['time'].dt.tz_convert('Etc/GMT+6')
    OM.columns = ['date', 'AirTemperature', 'RelativeHumidity', 'DewPoint', 'CloudCoverage', 'WindSpeed', 'SolarRadiation']
    OM['RelativeHumidity'] = OM['RelativeHumidity'] / 100
    OM['CloudCoverage'] = OM['CloudCoverage'] / 100
    return OM

#
sdf_path = r"C:\Users\artur\OneDrive\Doutorado\UTSA\PP\Data\After_construction\Dataframes\pp_data_022.csv"
sdf = pd.read_csv(sdf_path, parse_dates=[0], low_memory=False)
pavement = 'CP'
T_obs = pavement_temperature_data(sdf, pavement, method="average")

OM_file = r"C:\Users\artur\OneDrive\Doutorado\UTSA\PP\PPPaper_3\data\open-meteo\open-meteo-29.63N98.45W308m.csv"
OM = open_meteo_data(OM_file)
merged_df = OM.merge(T_obs, on='date', how='inner')
merged_df.to_csv(f'input_data/input_data_{pavement}.csv', index=False)


################
#  Rainfall
################

sdf_path = r"C:\Users\artur\OneDrive\Doutorado\UTSA\PP\Data\After_construction\Dataframes\pp_data_024.csv"
sdf = pd.read_csv(sdf_path, parse_dates=[0], low_memory=False)

start_dates = ["2023-08-22 15:00", "2023-10-05 04:00", "2023-10-26 08:00", "2023-11-09 18:00", "2023-12-23 22:00",
               "2024-01-21 23:00", "2024-02-02 19:00", "2024-04-09 22:00", "2024-04-20 22:00", "2024-04-28 06:00",
               "2024-05-13 08:00"]

end_dates = ["2023-08-22 22:00", "2023-10-05 22:00", "2023-10-27 03:10", "2023-11-10 06:00", "2023-12-24 18:00",
             "2024-01-23 10:00", "2024-02-03 10:00", "2024-04-10 06:00", "2024-04-21 15:00", "2024-04-28 20:00",
             "2024-05-14 00:00"]

rain_gauges = ['box_d_rainfall_pendant', 'box_d_rainfall_pendant', 'box_d_rainfall_pendant', 'box_d_rainfall_pendant',
               'box_d_rainfall_pendant', 'box_da_rainfall_accum', 'box_b_rainfall_accum', 'box_da_rainfall_accum',
               'box_b_rainfall_accum', 'box_b_rainfall_accum', 'box_b_rainfall_accum']

# check event 6

filtered_data = []

# Iterate over the defined periods
for start, end, gauge in zip(start_dates, end_dates, rain_gauges):
    temp_df = sdf[(sdf['date'] >= start) & (sdf['date'] <= end)][['date', gauge]].copy()
    temp_df.rename(columns={gauge: 'Rainfall'}, inplace=True)  # Rename column to 'Rainfall'
    filtered_data.append(temp_df)

# Concatenate all filtered data
result_df = pd.concat(filtered_data, ignore_index=True)
result_df.dropna(inplace=True)
hourly_rainfall = result_df.resample("1h", on='date').sum().reset_index()
hourly_rainfall['date'] = hourly_rainfall['date'] = hourly_rainfall['date'].dt.tz_localize('Etc/GMT+6')
hourly_rainfall['Rainfall'] = hourly_rainfall['Rainfall'] * 25.4
input_df = pd.read_csv("input_data/input_data_CP.csv", parse_dates=[0])
merged_df = pd.merge(input_df, hourly_rainfall[['date', 'Rainfall']], on='date', how='left')

merged_df.to_csv('input_data/input_data_CP.csv', index=False)


