import temperature_model
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import timedelta


def generate_input_file(pavement, meteo_data, surface_temperature, rainfall, water_temperature, bottom_temperature):
    surface_temperature = surface_temperature[['date', f'{pavement}_surface_temperature']]
    merged_df = meteo_data.merge(surface_temperature, on='date', how='inner')
    merged_df = merged_df.merge(rainfall, on='date', how='left')

    if pavement != "CP":
        water_temperature = water_temperature[['date', f'{pavement}_water_temperature']]
        water_temperature.columns = ['date', 'WaterTemperature']
        bottom_temperature = bottom_temperature[['date', f'{pavement}_bottom_temperature']]
        bottom_temperature.columns = ['date', 'BottomTemperature']
        merged_df = merged_df.merge(bottom_temperature, on='date', how='left')
        merged_df = merged_df.merge(water_temperature, on='date', how='left')

    return merged_df


def plot_weekly_data_four_series(dataframe, pavement_type):
    """
    Plot weekly data with four series: rainfall, surface temperature, water temperature, and bottom temperature.

    Parameters:
    -----------
    dataframe : pandas.DataFrame
        Main dataframe containing temperature data
    pavement_type : str
        Type of pavement, used to select correct surface temperature column
    rainfall_df : pandas.DataFrame
        Dataframe containing rainfall data

    Returns:
    --------
    list
        List of matplotlib figures, one per week
    """


    # Ensure datetime
    dataframe['date'] = pd.to_datetime(dataframe['date'])
    rainfall_df = dataframe[['date', 'Rainfall']]

    # Get start and end dates
    start_date = dataframe['date'].min()
    end_date = dataframe['date'].max()

    # Calculate number of weeks
    total_days = (end_date - start_date).days
    num_weeks = total_days // 7 + 1

    figures = []

    for week in range(num_weeks):
        week_start = start_date + timedelta(days=week * 7)
        week_end = week_start + timedelta(days=7)

        # Filter for the week
        week_temp = dataframe[(dataframe['date'] >= week_start) & (dataframe['date'] < week_end)]
        week_rain = rainfall_df[(rainfall_df['date'] >= week_start) & (rainfall_df['date'] < week_end)]

        if len(week_temp) == 0:
            continue

        # Create figure with 4 subplots
        fig, axs = plt.subplots(4, 1, figsize=(20, 16), sharex=True,
                                gridspec_kw={'height_ratios': [1, 2, 2, 2]})

        # --- Rainfall subplot (top) ---
        axs[0].bar(week_rain['date'], week_rain['Rainfall'], width=0.008, color='black')
        axs[0].invert_yaxis()
        axs[0].set_ylabel('Rainfall (mm)')
        axs[0].set_title('Rainfall')

        # Optional: Annotate total rainfall
        total_rain = week_rain['Rainfall'].sum()
        mid_point = week_start + timedelta(days=3.5)
        axs[0].text(mid_point, axs[0].get_ylim()[1] * 0.9,
                    f'Total: {total_rain:.1f} mm',
                    ha='center', va='top', fontsize=10, fontweight='bold')

        # --- Surface Temperature subplot ---
        temp_column = f'{pavement_type}_surface_temperature'
        axs[1].plot(week_temp['date'], week_temp[temp_column],
                    color='red', marker='o', markersize=3, label='Surface')
        axs[1].set_ylabel('Temperature (°C)')
        axs[1].set_title(f'{pavement_type} Surface Temperature')
        axs[1].grid(True, linestyle='--', alpha=0.7)
        axs[1].legend(loc='best')

        # --- Water Temperature subplot ---
        # Handle NaN values with proper masking
        axs[2].plot(week_temp['date'], week_temp['WaterTemperature'],
                    color='blue', marker='o', markersize=3, label='Water', linestyle='-')
        axs[2].set_ylabel('Temperature (°C)')
        axs[2].set_title('Water Temperature')
        axs[2].grid(True, linestyle='--', alpha=0.7)
        axs[2].legend(loc='best')

        # --- Bottom Temperature subplot ---
        # Handle NaN values with proper masking
        axs[3].plot(week_temp['date'], week_temp['BottomTemperature'],
                    color='green', marker='o', markersize=3, label='Bottom', linestyle='-')
        axs[3].set_ylabel('Temperature (°C)')
        axs[3].set_title('Bottom Temperature')
        axs[3].grid(True, linestyle='--', alpha=0.7)
        axs[3].legend(loc='best')

        # Format x-axis
        axs[3].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        axs[3].xaxis.set_major_locator(mdates.DayLocator())
        plt.xticks(rotation=45)

        # Add overall title
        plt.suptitle(
            f'Weekly Data: {week_start.strftime("%Y-%m-%d")} to {(week_end - timedelta(days=1)).strftime("%Y-%m-%d")}',
            fontsize=16, y=0.995)

        plt.tight_layout()
        figures.append(fig)
        plt.show()
        plt.close(fig)  # Close the figure to prevent display in Jupyter notebooks

    return figures


def plot_weekly_data_with_model(dataframe, results_df, pavement_type):
    """
    Plot weekly data with observed values and model results.

    Parameters:
    -----------
    dataframe : pandas.DataFrame
        Main dataframe containing observed temperature data
    results_df : pandas.DataFrame
        Dataframe containing model results with 'date', 'surface_temp', and 'water_temp' columns
    pavement_type : str
        Type of pavement, used to select correct surface temperature column

    Returns:
    --------
    list
        List of matplotlib figures, one per week
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from datetime import timedelta

    # Ensure datetime
    dataframe['date'] = pd.to_datetime(dataframe['date'])
    results_df['date'] = pd.to_datetime(results_df['date'])
    rainfall_df = dataframe[['date', 'Rainfall']]

    # Get start and end dates
    start_date = dataframe['date'].min()
    end_date = dataframe['date'].max()

    # Calculate number of weeks
    total_days = (end_date - start_date).days
    num_weeks = total_days // 7 + 1

    figures = []

    for week in range(num_weeks):
        week_start = start_date + timedelta(days=week * 7)
        week_end = week_start + timedelta(days=7)

        # Filter for the week
        week_temp = dataframe[(dataframe['date'] >= week_start) & (dataframe['date'] < week_end)]
        week_rain = rainfall_df[(rainfall_df['date'] >= week_start) & (rainfall_df['date'] < week_end)]
        week_results = results_df[(results_df['date'] >= week_start) & (results_df['date'] < week_end)]

        if len(week_temp) == 0:
            continue

        # Create figure with 4 subplots
        fig, axs = plt.subplots(4, 1, figsize=(20, 16), sharex=True,
                                gridspec_kw={'height_ratios': [1, 2, 2, 2]})

        # --- Rainfall subplot (top) ---
        axs[0].bar(week_rain['date'], week_rain['Rainfall'], width=0.008, color='black')
        axs[0].invert_yaxis()
        axs[0].set_ylabel('Rainfall (mm)')
        axs[0].set_title('Rainfall')

        # Optional: Annotate total rainfall
        total_rain = week_rain['Rainfall'].sum()
        mid_point = week_start + timedelta(days=3.5)
        axs[0].text(mid_point, axs[0].get_ylim()[1] * 0.9,
                    f'Total: {total_rain:.1f} mm',
                    ha='center', va='top', fontsize=10, fontweight='bold')

        # --- Surface Temperature subplot ---
        temp_column = f'{pavement_type}_surface_temperature'
        # Plot observed data
        axs[1].plot(week_temp['date'], week_temp[temp_column],
                    color='red', marker='o', markersize=3, label='Observed Surface')
        # Plot model results - dashed line
        axs[1].plot(week_results['date'], week_results['surface_temp'],
                    color='red', linestyle='--', linewidth=2, label='Modeled Surface')
        axs[1].set_ylabel('Temperature (°C)')
        axs[1].set_title(f'{pavement_type} Surface Temperature')
        axs[1].grid(True, linestyle='--', alpha=0.7)
        axs[1].legend(loc='best')

        # --- Water Temperature subplot ---
        # Plot observed data
        axs[2].plot(week_temp['date'], week_temp['WaterTemperature'],
                    color='blue', marker='o', markersize=3, label='Observed Water')
        # Plot model results - dashed line
        axs[2].plot(week_results['date'], week_results['water_temp'],
                    color='blue', linestyle='--', linewidth=2, label='Modeled Water')
        axs[2].set_ylabel('Temperature (°C)')
        axs[2].set_title('Water Temperature')
        axs[2].grid(True, linestyle='--', alpha=0.7)
        axs[2].legend(loc='best')

        # --- Bottom Temperature subplot ---
        # Handle NaN values with proper masking (no model data for this)
        axs[3].plot(week_temp['date'], week_temp['BottomTemperature'],
                    color='green', marker='o', markersize=3, label='Bottom', linestyle='-')
        axs[3].set_ylabel('Temperature (°C)')
        axs[3].set_title('Bottom Temperature')
        axs[3].grid(True, linestyle='--', alpha=0.7)
        axs[3].legend(loc='best')

        # Format x-axis
        axs[3].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        axs[3].xaxis.set_major_locator(mdates.DayLocator())
        plt.xticks(rotation=45)

        # Add overall title
        plt.suptitle(
            f'Weekly Data with Model Results: {week_start.strftime("%Y-%m-%d")} to {(week_end - timedelta(days=1)).strftime("%Y-%m-%d")}',
            fontsize=16, y=0.995)

        plt.tight_layout()
        figures.append(fig)
        plt.show()  # Display the figure
        plt.close(fig)  # Close the figure after displaying to free memory

    return figures


def plot_event_data_with_model(dataframe, results_df, pavement_type, water_events):
    """
    Plot data for specific event date ranges with observed values and model results.

    Parameters:
    -----------
    dataframe : pandas.DataFrame
        Main dataframe containing observed temperature data
    results_df : pandas.DataFrame
        Dataframe containing model results with 'date', 'surface_temp', and 'water_temp' columns
    pavement_type : str
        Type of pavement, used to select correct surface temperature column
    water_events : dict
        Dictionary of event names and date ranges, where each value is a list [start_date, end_date]

    Returns:
    --------
    list
        List of matplotlib figures, one per event
    """
    # Ensure datetime
    dataframe['date'] = pd.to_datetime(dataframe['date'])
    results_df['date'] = pd.to_datetime(results_df['date'])
    rainfall_df = dataframe[['date', 'Rainfall']]

    figures = []

    # Process each event
    for event_name, event_dates in water_events.items():
        # Parse event dates
        event_start = pd.to_datetime(event_dates[0])
        event_end = pd.to_datetime(event_dates[1]) + timedelta(days=1)

        # Filter for the event period
        event_temp = dataframe[(dataframe['date'] >= event_start) & (dataframe['date'] < event_end)]
        event_rain = rainfall_df[(rainfall_df['date'] >= event_start) & (rainfall_df['date'] < event_end)]
        event_results = results_df[(results_df['date'] >= event_start) & (results_df['date'] < event_end)]

        if len(event_temp) == 0:
            print(f"No data available for {event_name}: {event_dates}")
            continue

        # Create figure with 4 subplots
        fig, axs = plt.subplots(4, 1, figsize=(20, 16), sharex=True,
                                gridspec_kw={'height_ratios': [1, 2, 2, 2]})

        # --- Rainfall subplot (top) ---
        axs[0].bar(event_rain['date'], event_rain['Rainfall'], width=0.008, color='black')
        axs[0].invert_yaxis()
        axs[0].set_ylabel('Rainfall (mm)')
        axs[0].set_title('Rainfall')

        # Optional: Annotate total rainfall
        total_rain = event_rain['Rainfall'].sum()
        mid_point = event_start + (event_end - event_start) / 2
        axs[0].text(mid_point, axs[0].get_ylim()[1] * 0.9,
                    f'Total: {total_rain:.1f} mm',
                    ha='center', va='top', fontsize=10, fontweight='bold')

        # --- Surface Temperature subplot ---
        temp_column = f'{pavement_type}_surface_temperature'
        # Plot observed data
        axs[1].plot(event_temp['date'], event_temp[temp_column],
                    color='red', marker='o', markersize=3, label='Observed Surface')
        # Plot model results - dashed line
        if not event_results.empty:
            axs[1].plot(event_results['date'], event_results['surface_temp'],
                        color='red', linestyle='--', linewidth=2, label='Modeled Surface')
        axs[1].set_ylabel('Temperature (°C)')
        axs[1].set_title(f'{pavement_type} Surface Temperature')
        axs[1].grid(True, linestyle='--', alpha=0.7)
        axs[1].legend(loc='best')

        # --- Water Temperature subplot ---
        # Plot observed data
        axs[2].plot(event_temp['date'], event_temp['WaterTemperature'],
                    color='blue', marker='o', markersize=3, label='Observed Water')
        # Plot model results - dashed line
        if not event_results.empty:
            axs[2].plot(event_results['date'], event_results['water_temp'],
                        color='blue', linestyle='--', linewidth=2, label='Modeled Water')
        axs[2].set_ylabel('Temperature (°C)')
        axs[2].set_title('Water Temperature')
        axs[2].grid(True, linestyle='--', alpha=0.7)
        axs[2].legend(loc='best')

        # --- Bottom Temperature subplot ---
        # Handle NaN values with proper masking (no model data for this)
        axs[3].plot(event_temp['date'], event_temp['BottomTemperature'],
                    color='green', marker='o', markersize=3, label='Bottom', linestyle='-')
        axs[3].set_ylabel('Temperature (°C)')
        axs[3].set_title('Bottom Temperature')
        axs[3].grid(True, linestyle='--', alpha=0.7)
        axs[3].legend(loc='best')

        # Format x-axis
        axs[3].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))

        # Adjust x-axis tick locator based on event duration
        days_in_event = (event_end - event_start).days
        if days_in_event <= 2:
            axs[3].xaxis.set_major_locator(mdates.HourLocator(interval=6))  # Every 6 hours for short events
        else:
            axs[3].xaxis.set_major_locator(mdates.DayLocator())  # Daily for longer events

        plt.xticks(rotation=45)

        # Add overall title
        plt.suptitle(
            f'Event Data: {event_name} ({event_start.strftime("%Y-%m-%d")} to {(event_end - timedelta(days=1)).strftime("%Y-%m-%d")})',
            fontsize=16, y=0.995)

        plt.tight_layout()
        figures.append(fig)
        plt.show()  # Display the figure
        plt.close(fig)  # Close the figure after displaying to free memory

    return figures


pavement = 'PA'
surface_temperature = pd.read_csv(r'input_data\surface_temperature.csv', parse_dates=[0])
rainfall = pd.read_csv(r'input_data\rainfall.csv', parse_dates=[0])
water_temperature = pd.read_csv(r'input_data\water_temperature.csv', parse_dates=[0])
meteo_data = pd.read_csv(r'input_data\meteo_data.csv', parse_dates=[0])
bottom_temperature = pd.read_csv(r'input_data\bottom_temperature.csv', parse_dates=[0])

input_df = generate_input_file(pavement, meteo_data, surface_temperature, rainfall, water_temperature, bottom_temperature)
parameters_file = rf"C:\Users\Artur\PycharmProjects\data_treatment_pavement_temperature\input_data\parameters_{pavement}.ini"

results = temperature_model.model_pavement_temperature(input_df, parameters_file)

# figures = plot_weekly_data_four_series(input_df, pavement)
# figures = plot_weekly_data_with_model(input_df, results, pavement)


water_events = {
    'event_1': ['2023-10-05 00:00:00-06:00', '2023-10-06 00:00:00-06:00'],
    'event_2': ['2023-10-26 00:00:00-06:00', '2023-10-27 00:00:00-06:00'],
    'event_3': ['2023-11-09 00:00:00-06:00', '2023-11-10 00:00:00-06:00'],
    'event_4': ['2024-01-19 00:00:00-06:00', '2024-01-23 00:00:00-06:00'],
    'event_5': ['2024-01-30 00:00:00-06:00', '2024-02-04 00:00:00-06:00'],
    'event_6': ['2024-04-10 00:00:00-06:00', '2024-04-11 00:00:00-06:00'],
    'event_7': ['2024-04-21 00:00:00-06:00', '2024-04-22 00:00:00-06:00'],
    'event_8': ['2024-04-28 00:00:00-06:00', '2024-04-29 00:00:00-06:00'],
    'event_9': ['2024-05-13 00:00:00-06:00', '2024-05-14 00:00:00-06:00']
}
figures = plot_event_data_with_model(input_df, results, pavement, water_events)
