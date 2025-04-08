import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import temperature_model


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


def pavement_temperature_data(sdf, pavement, method='average', time_resolution="1h"):
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

    # Check if the date column is already timezone-aware
    if sdf['date'].dt.tz is None:
        sdf['date'] = sdf['date'].dt.tz_localize('Etc/GMT+6')
    else:
        sdf['date'] = sdf['date'].dt.tz_convert('Etc/GMT+6')

    # Select the two temperature columns for the given box
    T_obs = sdf[['date', f'{box}_temperature_1', f'{box}_temperature_2']]
    T_obs.columns = ['date', 'temperature_1', 'temperature_2']

    # Resample the data to hourly averages
    T_obs = T_obs.resample(time_resolution, on='date').mean()

    # Apply the temperature selection function with the chosen method
    T_obs['PavementTemperature'] = T_obs.apply(
        lambda row: compute_valid_temperature(row, method=method), axis=1
    )

    # Clean up the dataframe and convert to Celsius if needed
    T_obs.drop(columns=['temperature_1', 'temperature_2'], inplace=True)
    T_obs = T_obs.dropna().reset_index()
    T_obs['PavementTemperature'] = (T_obs['PavementTemperature'] - 32) / 1.8

    temp_copy = T_obs['PavementTemperature'].copy()
    if time_resolution == "5min":
        if pavement == "CP":
            T_obs.loc[12384:14112, 'PavementTemperature'] = temp_copy.loc[12384:14112].shift(-18)

        if pavement == "PICP":
            T_obs.loc[:4032, 'PavementTemperature'] = temp_copy.loc[:4032].shift(-24)
            T_obs.loc[4032:5032, 'PavementTemperature'] = temp_copy.loc[4032:5032].shift(36)
            T_obs.loc[5032:, 'PavementTemperature'] = temp_copy.loc[5032:].shift(-24)

        if pavement == "PGr":
            T_obs.loc[:, 'PavementTemperature'] = temp_copy.loc[:].shift(-24)

        if pavement == "PA":
            T_obs.loc[:21660, 'PavementTemperature'] = T_obs.loc[:21660, 'PavementTemperature'].shift(-12)
            T_obs.loc[46300:, 'PavementTemperature'] = T_obs.loc[46300:, 'PavementTemperature'].shift(-12)

        if pavement == "PC":
            T_obs = T_obs.iloc[576:, :]
            T_obs.loc[:6144, 'PavementTemperature'] = T_obs.loc[:6144, 'PavementTemperature'].shift(-108)
            T_obs.loc[6144:, 'PavementTemperature'] = T_obs.loc[6144:, 'PavementTemperature'].shift(-24)

    elif time_resolution == "1h":
        if pavement == "CP":
            T_obs.loc[1032:1176, 'PavementTemperature'] = temp_copy.loc[1032:1176].shift(-1)

        if pavement == "PICP":
            T_obs.loc[:336, 'PavementTemperature'] = temp_copy.loc[:336].shift(-2)
            T_obs.loc[336:419, 'PavementTemperature'] = temp_copy.loc[336:419].shift(3)
            T_obs.loc[419:, 'PavementTemperature'] = temp_copy.loc[419:].shift(-2)

        if pavement == "PGr":
            T_obs.loc[:, 'PavementTemperature'] = temp_copy.loc[:].shift(-2)

        if pavement == "PA":
            T_obs.loc[:1805, 'PavementTemperature'] = T_obs.loc[:1805, 'PavementTemperature'].shift(-1)
            T_obs.loc[3858:, 'PavementTemperature'] = T_obs.loc[3858:, 'PavementTemperature'].shift(-1)

        if pavement == "PC":
            T_obs = T_obs.iloc[48:, :]
            T_obs.loc[:512, 'PavementTemperature'] = T_obs.loc[:512, 'PavementTemperature'].shift(-9)
            T_obs.loc[512:, 'PavementTemperature'] = T_obs.loc[512:, 'PavementTemperature'].shift(-2)

    return T_obs


def pavement_bottom_temperature_data(sdf, pavement, time_resolution="1h"):
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
    pavements = {"PICP": "box_b", "PGr": "box_c", "PA": "box_da", "PC": "box_dc"}
    box = pavements[pavement]

    # Select the two temperature columns for the given box
    T_bottom = sdf[['date', f'{box}_pressure_temperature']]
    T_bottom.columns = ['date', 'BottomTemperature']

    # Resample the data to 5-minutes averages
    T_bottom = T_bottom.resample(time_resolution, on='date').mean()
    T_bottom = T_bottom.dropna().reset_index()
    T_bottom['BottomTemperature'] = (T_bottom['BottomTemperature'] - 32) / 1.8
    return T_bottom


def open_meteo_data(OM_file, time_resolution="1h"):
    OM = pd.read_csv(OM_file, skiprows=2)
    OM['time'] = pd.to_datetime(OM['time'], utc=True)
    OM['time'] = OM['time'].dt.tz_convert('Etc/GMT+6')
    OM.columns = ['date', 'AirTemperature', 'RelativeHumidity', 'DewPoint', 'CloudCoverage', 'WindSpeed', 'SolarRadiation']
    OM['RelativeHumidity'] = OM['RelativeHumidity'] / 100
    OM['CloudCoverage'] = OM['CloudCoverage'] / 100

    if time_resolution != "1h":
        OM['date'] = pd.to_datetime(OM['date'])
        OM.set_index('date', inplace=True)
        new_index = pd.date_range(start=OM.index.min(), end=OM.index.max(), freq=time_resolution)
        OM = OM.reindex(new_index)
        OM = OM.interpolate(method='linear')
        OM.reset_index(drop=False, inplace=True)
        OM.rename(columns={'index': 'date'}, inplace=True)
    return OM


def rainfall_data(sdf, time_resolution="1h"):
    start_dates = ["2023-08-22 15:00", "2023-10-05 04:00", "2023-10-26 08:00", "2023-11-09 18:00", "2023-12-23 22:00",
                   "2024-01-21 23:00", "2024-02-02 19:00", "2024-04-09 22:00", "2024-04-20 22:00", "2024-04-28 06:00",
                   "2024-05-13 08:00"]

    end_dates = ["2023-08-22 22:00", "2023-10-05 22:00", "2023-10-27 03:10", "2023-11-10 06:00", "2023-12-24 18:00",
                 "2024-01-23 10:00", "2024-02-03 10:00", "2024-04-10 06:00", "2024-04-21 15:00", "2024-04-28 20:00",
                 "2024-05-14 00:00"]

    rain_gauges = ['box_d_rainfall_pendant', 'box_d_rainfall_pendant', 'box_d_rainfall_pendant',
                   'box_d_rainfall_pendant',
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
    hourly_rainfall = result_df.resample(time_resolution, on='date').sum().reset_index()
    hourly_rainfall['Rainfall'] = hourly_rainfall['Rainfall'] * 25.4
    return hourly_rainfall


def water_temperature_data(pavement):
    pavements = {"CP": "box_a", "PICP": "box_b", "PGr": "box_c", "PA": "box_da", "PC": "box_dc"}
    box = pavements[pavement]
    water_data_path = rf"C:\Users\Artur\OneDrive\Doutorado\UTSA\PP\PPPaper_3\data\water_temperature_{box}.csv"
    df = pd.read_csv(water_data_path, parse_dates=[0])
    df = df[['date', f'temp_{box}']]
    df['date'] = df['date'].dt.tz_localize('Etc/GMT+6')
    df.dropna(inplace=True)
    df.columns = ['date', 'WaterTemperature']
    df['WaterTemperature'] = (df['WaterTemperature'] - 32) / 1.8
    return df


def plot_time_shift(input_file, parameters_file, time_resolution="1h"):
    sim_df = pd.read_csv(input_file)
    model_results = temperature_model.model_pavement_temperature(sim_df, parameters_file)

    num_rows = len(model_results)

    if time_resolution == "1h":
        rows_per_week = 168

    elif time_resolution == "5min":
        rows_per_week = 2016

    else:
        raise Exception("Valid time resolution is 1h or 5min.")

    for start in range(0, num_rows, rows_per_week):
        end = min(start + rows_per_week, num_rows)
        observed_segment = sim_df.iloc[start:end]
        simulated_segment = model_results.iloc[start:end]

        # Get the date range from the observed segment (if the 'date' column exists)
        if 'date' in observed_segment.columns and not observed_segment.empty:
            start_date = observed_segment['date'].iloc[0]
            end_date = observed_segment['date'].iloc[-1]
        else:
            start_date, end_date = start, end

        plt.figure(figsize=(12, 6))
        plt.plot(simulated_segment['surface_temp'].values, label='Modeled')
        plt.plot(observed_segment['PavementTemperature'].values, label='Observed')
        plt.xlabel('Time steps (5-min intervals)')
        plt.ylabel('Temperature (°C)')
        plt.title(f'Index {start} to {end} | Date range: {start_date} to {end_date}')
        plt.legend(loc='best')
        plt.show()


def generate_input_files(sdf_path, time_resolution, plot=True):
    sdf = pd.read_csv(sdf_path, parse_dates=[0], low_memory=False)
    pavements = ["CP", "PICP", "PGr", "PA", "PC"]

    for pavement in pavements:
        T_obs = pavement_temperature_data(sdf, pavement, method="average", time_resolution=time_resolution)
        OM_file = r"C:\Users\artur\OneDrive\Doutorado\UTSA\PP\PPPaper_3\data\open-meteo\open-meteo-29.63N98.45W308m.csv"
        OM = open_meteo_data(OM_file, time_resolution=time_resolution)
        rainfall = rainfall_data(sdf, time_resolution=time_resolution)
        merged_df = OM.merge(T_obs, on='date', how='inner')
        merged_df = merged_df.merge(rainfall, on='date', how='left')

        if pavement != "CP":
            T_bottom = pavement_bottom_temperature_data(sdf, pavement, time_resolution=time_resolution)
            water_temperature = water_temperature_data(pavement)
            merged_df = merged_df.merge(T_bottom, on='date', how='left')
            merged_df = merged_df.merge(water_temperature, on='date', how='left')

        if time_resolution == "1h":
            merged_df.to_csv(rf"input_data/1h/input_data_{pavement}.csv", index=False)
            input_file = rf"input_data\1h\input_data_{pavement}.csv"
            parameters_file = rf"input_data\1h\parameters_{pavement}.ini"
            if plot:
                plot_time_shift(input_file, parameters_file, time_resolution)


        if time_resolution == "5min":
            merged_df.to_csv(rf"input_data/input_data_{pavement}.csv", index=False)
            input_file = rf"input_data\input_data_{pavement}.csv"
            parameters_file = rf"input_data\parameters_{pavement}.ini"
            if plot:
                plot_time_shift(input_file, parameters_file, time_resolution)



# sdf_path = r"C:\Users\artur\OneDrive\Doutorado\UTSA\PP\Data\After_construction\Dataframes\pp_data_024.csv"
# time_resolution = '1h'
# generate_input_files(sdf_path, time_resolution)
#
#
# df1h = pd.read_csv(rf"input_data/1h/input_data_CP.csv", parse_dates=[0])
# df5m = pd.read_csv(rf"input_data/input_data_CP.csv", parse_dates=[0])
#
# plt.plot(df1h['date'], df1h.iloc[:, 1], marker='o')
# plt.plot(df5m['date'], df5m.iloc[:, 1], alpha=0.5)
# plt.show()


# sdf_path = r"C:\Users\artur\OneDrive\Doutorado\UTSA\PP\Data\After_construction\Dataframes\pp_data_024.csv"
# sdf = pd.read_csv(sdf_path, parse_dates=[0], low_memory=False)
# pavement = 'PC'
# time_resolution = '1h'
#
# T_obs = pavement_temperature_data(sdf, pavement, method="average", time_resolution=time_resolution)
#
# temp_copy = T_obs['PavementTemperature'].copy()
#
#
#
# # OpenMeteo File
# OM_file = r"C:\Users\artur\OneDrive\Doutorado\UTSA\PP\PPPaper_3\data\open-meteo\open-meteo-29.63N98.45W308m.csv"
# OM = open_meteo_data(OM_file, time_resolution=time_resolution)
#
# # Rainfall
# rainfall = rainfall_data(sdf, time_resolution=time_resolution)
#
# # Pressure Transducer Bottom Temperature
# T_bottom = pavement_bottom_temperature_data(sdf, pavement, time_resolution=time_resolution)
#
# # Water Temperature
# water_temperature = water_temperature_data(pavement)
#
# # Merge the dataframes and save as .csv files
# merged_df = OM.merge(T_obs, on='date', how='inner')
# merged_df = merged_df.merge(rainfall, on='date', how='left')
# merged_df = merged_df.merge(T_bottom, on='date', how='left')
# merged_df = merged_df.merge(water_temperature, on='date', how='left')
# merged_df.to_csv(rf"input_data\1h\input_data_{pavement}.csv", index=False)
#
#
# #Plot the results
# input_file = rf"input_data\1h\input_data_{pavement}.csv"
# parameters_file = rf"input_data\1h\parameters_{pavement}.ini"
#
# plot_time_shift(input_file, parameters_file, time_resolution)

