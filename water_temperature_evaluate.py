import temperature_model
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import timedelta
import numpy as np
import os


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


def plot_event_data_with_model(dataframe, results_df, pavement_type, water_events, avg_temps_df):
    """
    Plot data for specific event date ranges with observed values and model results.
    Model results are only plotted where observed data exists (using observed as a mask).

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
    avg_temps_df : pandas.DataFrame
        Dataframe containing average temperatures for each event with 'model_avg_temp' and 'observed_avg_temp' columns
        The index should match the event names in water_events dictionary

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

    # Set font sizes
    TITLE_SIZE = 16
    LABEL_SIZE = 16
    LEGEND_SIZE = 16
    TICK_SIZE = 14

    # Process each event
    for event_name, event_dates in water_events.items():
        # Parse event dates
        event_start = pd.to_datetime(event_dates[0])
        event_end = pd.to_datetime(event_dates[1])

        # Filter for the event period
        event_temp = dataframe[(dataframe['date'] >= event_start) & (dataframe['date'] < event_end)]
        event_rain = rainfall_df[(rainfall_df['date'] >= event_start) & (rainfall_df['date'] < event_end)]
        event_results = results_df[(results_df['date'] >= event_start) & (results_df['date'] < event_end)]

        if len(event_temp) == 0:
            print(f"No data available for {event_name}: {event_dates}")
            continue

        # Get average temperatures for this event
        # Check if event_name exists in the 'event' column of avg_temps_df
        if 'event' in avg_temps_df.columns and event_name in avg_temps_df['event'].values:
            # Get the row for this event
            event_row = avg_temps_df[avg_temps_df['event'] == event_name].iloc[0]
            model_avg = event_row['model_avg_temp']
            observed_avg = event_row['observed_avg_temp']
            print(f"Event: {event_name}, Model Avg: {model_avg:.2f}, Observed Avg: {observed_avg:.2f}")
        else:
            model_avg = None
            observed_avg = None
            print(f"Warning: Event {event_name} not found in average temperatures dataframe")

        # Total rainfall for annotation
        total_rain = event_rain['Rainfall'].sum()

        # Create figure with 3 subplots (rainfall, surface temp, water temp)
        fig, axs = plt.subplots(3, 1, figsize=(20, 12), sharex=True,
                                gridspec_kw={'height_ratios': [1, 2, 2]})

        # --- Rainfall subplot (top) ---
        axs[0].bar(event_rain['date'], event_rain['Rainfall'], width=0.008, color='black')
        axs[0].invert_yaxis()
        axs[0].set_ylabel('Rainfall (mm)', fontsize=LABEL_SIZE)
        axs[0].set_title('Rainfall', fontsize=TITLE_SIZE)
        axs[0].tick_params(axis='both', which='major', labelsize=TICK_SIZE)

        # Add total rainfall annotation to the rainfall subplot
        total_rain = event_rain['Rainfall'].sum()
        axs[0].annotate(f'Total: {total_rain:.1f} mm',
                        xy=(0.95, 0.8), xycoords='axes fraction',
                        ha='right', va='top', fontsize=16, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.7))

        # --- Surface Temperature subplot ---
        temp_column = f'{pavement_type}_surface_temperature'

        # Plot observed data without average temperature in the label
        axs[1].plot(event_temp['date'], event_temp[temp_column],
                    color='red', marker='o', markersize=3, label='Observed Surface')

        # Plot model results - dashed line
        if not event_results.empty:
            axs[1].plot(event_results['date'], event_results['surface_temp'],
                        color='red', linestyle='--', linewidth=2, label='Modeled Surface')

        axs[1].set_ylabel('Temperature (°C)', fontsize=LABEL_SIZE)
        axs[1].set_title(f'{pavement_type} Surface Temperature', fontsize=TITLE_SIZE)
        axs[1].grid(True, linestyle='--', alpha=0.7)
        axs[1].legend(loc='best', fontsize=LEGEND_SIZE)
        axs[1].tick_params(axis='both', which='major', labelsize=TICK_SIZE)
        axs[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))

        # --- Water Temperature subplot ---
        # Get observed water temperature data, dropping NaN values
        observed_water_data = event_temp[['date', 'WaterTemperature']].dropna()

        # Plot observed data with average temperature in the label
        water_obs_label = f'Observed Water (Avg: {observed_avg:.2f}°C)' if observed_avg is not None else 'Observed Water'
        axs[2].plot(observed_water_data['date'], observed_water_data['WaterTemperature'],
                    color='blue', marker='o', markersize=3, label=water_obs_label)

        # Only plot model results for timestamps where we have observed data
        if not event_results.empty and not observed_water_data.empty:
            # Filter model results to only include timestamps where observed data exists
            common_dates = observed_water_data['date']
            filtered_results = event_results[event_results['date'].isin(common_dates)]

            if not filtered_results.empty:
                water_model_label = f'Modeled Water (Avg: {model_avg:.2f}°C)' if model_avg is not None else 'Modeled Water'
                axs[2].plot(filtered_results['date'], filtered_results['water_temp_infil'],
                            color='blue', linestyle='--', linewidth=2, label=water_model_label)

        # Calculate y-axis limits for water temperature subplot
        water_data = observed_water_data['WaterTemperature'].tolist()
        if not event_results.empty and not filtered_results.empty:
            water_data.extend(filtered_results['water_temp_infil'].tolist())

        # Determine min and max with some padding
        min_temp = 0
        max_temp = max(water_data)

        # Add padding (e.g., 15% on each side)
        y_min = min_temp
        y_max = max_temp + 15

        # Explicitly set y-axis limits
        axs[2].set_ylim(y_min, y_max)

        axs[2].set_ylabel('Temperature (°C)', fontsize=LABEL_SIZE)
        axs[2].set_title(f'{pavement_type} Water Temperature', fontsize=TITLE_SIZE)
        axs[2].grid(True, linestyle='--', alpha=0.7)
        axs[2].legend(loc='best', fontsize=LEGEND_SIZE)
        axs[2].tick_params(axis='both', which='major', labelsize=TICK_SIZE)
        axs[2].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))

        # Format x-axis
        axs[2].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

        # Adjust x-axis tick locator based on event duration
        days_in_event = (event_end - event_start).days
        if days_in_event <= 2:
            axs[2].xaxis.set_major_locator(mdates.HourLocator(interval=6))  # Every 6 hours for short events
        else:
            axs[2].xaxis.set_major_locator(mdates.DayLocator())  # Daily for longer events

        plt.xticks(rotation=0, fontsize=TICK_SIZE)

        plt.suptitle(
            f'Rainfall Event on {event_start.strftime("%Y-%m-%d")}',
            fontsize=16, y=0.995)

        plt.tight_layout()
        figures.append(fig)
        plt.show()  # Display the figure
        plt.close(fig)  # Close the figure after displaying to free memory

    return figures


def generate_averages_water_temperature_df(water_events, results, input_df):
    # Convert event times to datetime
    water_events_dt = {
        event: [pd.to_datetime(start), pd.to_datetime(end)]
        for event, (start, end) in water_events.items()
    }

    modeled_averages = {}
    observed_averages = {}

    for event, (start, end) in water_events_dt.items():
        # Get observed data within the event window
        observed_mask = (input_df['date'] >= start) & (input_df['date'] <= end)
        observed_data = input_df.loc[observed_mask, ['date', 'WaterTemperature']].dropna()

        if observed_data.empty:
            modeled_averages[event] = None
            observed_averages[event] = None
            continue

        # Filter modeled data to the timestamps where we have observed data
        common_dates = observed_data['date']
        modeled_data = results[results['date'].isin(common_dates)]

        # Calculate averages using aligned timestamps
        observed_avg = observed_data['WaterTemperature'].mean().round(2)
        modeled_avg = modeled_data['water_temp_infil'].mean().round(2)

        modeled_averages[event] = modeled_avg
        observed_averages[event] = observed_avg

    # Combine into a single DataFrame
    model_df = pd.DataFrame.from_dict(modeled_averages, orient='index', columns=['model_avg_temp'])
    obs_df = pd.DataFrame.from_dict(observed_averages, orient='index', columns=['observed_avg_temp'])
    combined_df = model_df.join(obs_df)

    combined_df.index.name = 'event'
    combined_df.reset_index(inplace=True)

    return combined_df


def plot_event_data_impermeable(dataframe, results_df, pavement_type, water_events):
    """
    Plot data for specific event date ranges with observed values and model results
    for impermeable pavements.

    Parameters:
    -----------
    dataframe : pandas.DataFrame
        Main dataframe containing observed temperature data
    results_df : pandas.DataFrame
        Dataframe containing model results with 'date', 'surface_temp', and 'water_temp_surface' columns
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

    # Set font sizes
    TITLE_SIZE = 16
    LABEL_SIZE = 16
    LEGEND_SIZE = 16
    TICK_SIZE = 14

    # Process each event
    for event_name, event_dates in water_events.items():
        # Parse event dates
        event_start = pd.to_datetime(event_dates[0])
        event_end = pd.to_datetime(event_dates[1])

        # Filter for the event period
        event_temp = dataframe[(dataframe['date'] >= event_start) & (dataframe['date'] < event_end)]
        event_rain = rainfall_df[(rainfall_df['date'] >= event_start) & (rainfall_df['date'] < event_end)]
        event_results = results_df[(results_df['date'] >= event_start) & (results_df['date'] < event_end)]

        if len(event_temp) == 0:
            print(f"No data available for {event_name}: {event_dates}")
            continue

        # Total rainfall for annotation
        total_rain = event_rain['Rainfall'].sum()

        # Create figure with 3 subplots (rainfall, surface temp, water temp)
        fig, axs = plt.subplots(3, 1, figsize=(20, 12), sharex=True,
                                gridspec_kw={'height_ratios': [1, 2, 2]})

        # --- Rainfall subplot (top) ---
        axs[0].bar(event_rain['date'], event_rain['Rainfall'], width=0.008, color='black')
        axs[0].invert_yaxis()
        axs[0].set_ylabel('Rainfall (mm)', fontsize=LABEL_SIZE)
        axs[0].set_title('Rainfall', fontsize=TITLE_SIZE)
        axs[0].tick_params(axis='both', which='major', labelsize=TICK_SIZE)

        # Add total rainfall annotation to the rainfall subplot
        axs[0].annotate(f'Total: {total_rain:.1f} mm',
                        xy=(0.95, 0.8), xycoords='axes fraction',
                        ha='right', va='top', fontsize=16, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.7))

        # --- Surface Temperature subplot ---
        temp_column = f'{pavement_type}_surface_temperature'

        # Plot observed data
        axs[1].plot(event_temp['date'], event_temp[temp_column],
                    color='red', marker='o', markersize=3, label='Observed Surface')

        # Plot model results - dashed line
        if not event_results.empty:
            axs[1].plot(event_results['date'], event_results['surface_temp'],
                        color='red', linestyle='--', linewidth=2, label='Modeled Surface')

        axs[1].set_ylabel('Temperature (°C)', fontsize=LABEL_SIZE)
        axs[1].set_title(f'{pavement_type} Surface Temperature', fontsize=TITLE_SIZE)
        axs[1].grid(True, linestyle='--', alpha=0.7)
        axs[1].legend(loc='best', fontsize=LEGEND_SIZE)
        axs[1].tick_params(axis='both', which='major', labelsize=TICK_SIZE)

        # --- Water Temperature subplot ---
        # For impermeable pavement, we only have modeled water temperature (no observed)
        if not event_results.empty:
            # Calculate average water temperature for this event
            model_avg_water_temp = event_results['water_temp_surface'].mean()

            # Plot with 'x' marks and with the average in the label
            axs[2].plot(event_results['date'], event_results['water_temp_surface'],
                        color='blue', marker='x', markersize=6, linestyle='',
                        label=f'Modeled Surface Water (Avg: {model_avg_water_temp:.2f}°C)')

        axs[2].set_ylabel('Temperature (°C)', fontsize=LABEL_SIZE)
        axs[2].set_title('Surface Water Temperature', fontsize=TITLE_SIZE)
        axs[2].grid(True, linestyle='--', alpha=0.7)
        axs[2].legend(loc='best', fontsize=LEGEND_SIZE)
        axs[2].tick_params(axis='both', which='major', labelsize=TICK_SIZE)

        # Format x-axis
        axs[2].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))

        # Adjust x-axis tick locator based on event duration
        days_in_event = (event_end - event_start).days
        if days_in_event <= 2:
            axs[2].xaxis.set_major_locator(mdates.HourLocator(interval=6))  # Every 6 hours for short events
        else:
            axs[2].xaxis.set_major_locator(mdates.DayLocator())  # Daily for longer events

        plt.xticks(rotation=0, fontsize=TICK_SIZE)

        plt.suptitle(
            f'Event {event_start.strftime("%Y-%m-%d")}',
            fontsize=16, y=0.995)

        plt.tight_layout()
        figures.append(fig)
        plt.show()  # Display the figure
        plt.close(fig)  # Close the figure after displaying to free memory

    return figures


def plot_all_pavement_surface_temps(observed_data, model_results_dict, start_date, end_date, save_path=None):
    """
    Plot surface temperatures for all pavement types with one subplot per pavement,
    comparing observed values with model results for a specified date range.

    Parameters:
    -----------
    observed_data : pandas.DataFrame
        Main dataframe containing observed temperature data for all pavement types
        Expected columns: 'date', '{pavement}_surface_temperature' for each pavement type
    model_results_dict : dict
        Dictionary where keys are pavement types and values are dataframes containing model results
        Each dataframe should have 'date' and 'surface_temp' columns
    start_date : str or datetime
        Start date for the comparison period
    end_date : str or datetime
        End date for the comparison period
    save_path : str, optional
        Path to save the figure. If None, the figure is not saved

    Returns:
    --------
    matplotlib.figure.Figure
        The generated figure
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import numpy as np

    # Custom color mapping for each pavement type
    color_mapping = {
        'CP': (0.12156862745098039, 0.4666666666666667, 0.7058823529411765),  # Conventional
        'PICP': (1.0, 0.4980392156862745, 0.054901960784313725),  # PICP
        'PGr': (0.17254901960784313, 0.6274509803921569, 0.17254901960784313),  # Grid
        'PA': (0.5803921568627451, 0.403921568627451, 0.7411764705882353),  # Asphalt
        'PC': (0.8392156862745098, 0.15294117647058825, 0.1568627450980392),  # Concrete
    }

    # Ensure datetime
    observed_data['date'] = pd.to_datetime(observed_data['date'])
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Filter data for the specified date range
    observed_filter = (observed_data['date'] >= start_date) & (observed_data['date'] < end_date)
    observed_period = observed_data[observed_filter].copy()

    if len(observed_period) == 0:
        print(f"No observed data available for period: {start_date} to {end_date}")
        return None

    # List of pavement types
    pavements = ['CP', 'PICP', 'PGr', 'PA', 'PC']

    # Set font sizes
    TITLE_SIZE = 14
    LABEL_SIZE = 12
    TICK_SIZE = 10

    # Create figure with subplots (one per pavement)
    fig, axs = plt.subplots(len(pavements), 1, figsize=(16, 16), sharex=True)

    # Plot each pavement type in its own subplot
    for i, pavement in enumerate(pavements):
        ax = axs[i]
        temp_column = f'{pavement}_surface_temperature'

        # Get color for this pavement (default to black if not in mapping)
        color = color_mapping.get(pavement, (0, 0, 0))

        # Check if this pavement has observed data
        if temp_column in observed_period.columns:
            # Plot observed data (solid line with markers)
            temp_data = observed_period[['date', temp_column]].dropna()
            if not temp_data.empty:
                ax.plot(temp_data['date'], temp_data[temp_column],
                        color=color, linestyle='-', linewidth=2, marker='o', markersize=3,
                        label='Observed Surface')

            # Plot modeled data if available (dashed line)
            if pavement in model_results_dict:
                model_df = model_results_dict[pavement]
                model_df['date'] = pd.to_datetime(model_df['date'])

                # Filter model results for the period
                model_period = model_df[(model_df['date'] >= start_date) &
                                        (model_df['date'] < end_date)]

                if not model_period.empty:
                    ax.plot(model_period['date'], model_period['surface_temp'],
                            color=color, linestyle='--', linewidth=2,
                            label='Modeled Surface')

                    # Calculate and display RMSE if we have both observed and modeled data
                    if not temp_data.empty:
                        # Merge on date to align timestamps
                        merged = pd.merge(temp_data, model_period[['date', 'surface_temp']],
                                          on='date', how='inner')

                        if not merged.empty:
                            rmse = np.sqrt(((merged[temp_column] - merged['surface_temp']) ** 2).mean())

                            # Add RMSE annotation to the subplot
                            ax.annotate(f'RMSE: {rmse:.2f}°C',
                                        xy=(0.95, 0.05), xycoords='axes fraction',
                                        ha='right', va='bottom', fontsize=12, fontweight='bold',
                                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.7))

                            print(f"{pavement} RMSE: {rmse:.2f}°C")

        # Format subplot
        ax.set_ylabel('Temperature (°C)', fontsize=LABEL_SIZE)
        ax.set_title(f'{pavement} Surface Temperature', fontsize=TITLE_SIZE)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='upper right', fontsize=LABEL_SIZE)
        ax.tick_params(axis='both', which='major', labelsize=TICK_SIZE)

        # Set the same y-axis limits for all subplots for better comparison
        # Calculate min and max across all observed and modeled data
        all_temps = []
        for pav in pavements:
            col = f'{pav}_surface_temperature'
            if col in observed_period.columns:
                all_temps.extend(observed_period[col].dropna().tolist())

            if pav in model_results_dict:
                model_df = model_results_dict[pav]
                model_df['date'] = pd.to_datetime(model_df['date'])
                model_period = model_df[(model_df['date'] >= start_date) & (model_df['date'] < end_date)]
                if not model_period.empty:
                    all_temps.extend(model_period['surface_temp'].dropna().tolist())

        if all_temps:
            y_min = min(all_temps) - 2
            y_max = max(all_temps) + 2

            for ax_i in axs:
                ax_i.set_ylim(y_min, y_max)

    # Format x-axis (only for the bottom subplot)
    axs[-1].set_xlabel('Date', fontsize=LABEL_SIZE)
    axs[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))

    # Adjust x-axis tick locator based on period duration
    days_in_period = (end_date - start_date).days
    if days_in_period <= 2:
        axs[-1].xaxis.set_major_locator(mdates.HourLocator(interval=6))  # Every 6 hours for short periods
    elif days_in_period <= 7:
        axs[-1].xaxis.set_major_locator(mdates.DayLocator())  # Daily for weekly periods
        axs[-1].xaxis.set_minor_locator(mdates.HourLocator(interval=6))  # Minor ticks every 6 hours
    else:
        axs[-1].xaxis.set_major_locator(mdates.DayLocator(interval=2))  # Every 2 days for longer periods

    plt.xticks(rotation=30)

    # Add overall title
    plt.suptitle(
        f'Surface Temperature Comparison: {start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")}',
        fontsize=16, y=0.995)

    plt.tight_layout()

    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    return fig


def plot_all_pavement_surface_temps(observed_data, model_results_dict, end_date, save_path=None):
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import numpy as np

    color_mapping = {
        'CP': (0.121, 0.467, 0.706),
        'PICP': (1.0, 0.498, 0.055),
        'PGr': (0.173, 0.627, 0.173),
        'PA': (0.580, 0.404, 0.741),
        'PC': (0.839, 0.153, 0.157),
    }

    observed_data['date'] = pd.to_datetime(observed_data['date'])
    end_date = pd.to_datetime(end_date)
    start_date = observed_data['date'].min()

    observed_period = observed_data[(observed_data['date'] >= start_date) & (observed_data['date'] < end_date)].copy()

    if len(observed_period) == 0:
        print(f"No observed data available before {end_date}")
        return None

    pavements = ['CP', 'PICP', 'PGr', 'PA', 'PC']
    TITLE_SIZE, LABEL_SIZE, TICK_SIZE = 14, 12, 10
    fig, axs = plt.subplots(len(pavements), 1, figsize=(16, 16), sharex=True)

    all_temps = []

    for i, pavement in enumerate(pavements):
        ax = axs[i]
        color = color_mapping.get(pavement, (0, 0, 0))
        temp_column = f'{pavement}_surface_temperature'

        temp_data = observed_period[['date', temp_column]].dropna()
        if not temp_data.empty:
            ax.plot(temp_data['date'], temp_data[temp_column],
                    color=color, linestyle='-', linewidth=2, marker='o', markersize=3,
                    label='Observed Surface')

        # Model results masking
        if pavement in model_results_dict:
            model_df = model_results_dict[pavement].copy()
            model_df['date'] = pd.to_datetime(model_df['date'])

            model_period = model_df[(model_df['date'] >= start_date) & (model_df['date'] < end_date)]

            # Merge only on dates where observed data exists
            merged = pd.merge(temp_data[['date']], model_period, on='date', how='inner')

            if not merged.empty:
                ax.plot(merged['date'], merged['surface_temp'],
                        color=color, linestyle='--', linewidth=2,
                        label='Modeled Surface')

                rmse = np.sqrt(((temp_data.set_index('date').loc[merged['date']][temp_column].values -
                                 merged['surface_temp'].values) ** 2).mean())

                ax.annotate(f'RMSE: {rmse:.2f}°C',
                            xy=(0.95, 0.05), xycoords='axes fraction',
                            ha='right', va='bottom', fontsize=12, fontweight='bold',
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.7))

                print(f"{pavement} RMSE: {rmse:.2f}°C")

        ax.set_ylabel('Temperature (°C)', fontsize=LABEL_SIZE)
        ax.set_title(f'{pavement} Surface Temperature', fontsize=TITLE_SIZE)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='upper right', fontsize=LABEL_SIZE)
        ax.tick_params(axis='both', which='major', labelsize=TICK_SIZE)

        if temp_column in observed_period.columns:
            all_temps.extend(observed_period[temp_column].dropna().tolist())

        if pavement in model_results_dict:
            all_temps.extend(merged['surface_temp'].dropna().tolist())

    if all_temps:
        y_min, y_max = min(all_temps) - 2, max(all_temps) + 2
        for ax_i in axs:
            ax_i.set_ylim(y_min, y_max)

    axs[-1].set_xlabel('Date', fontsize=LABEL_SIZE)
    axs[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    days_in_period = (end_date - start_date).days

    if days_in_period <= 2:
        axs[-1].xaxis.set_major_locator(mdates.HourLocator(interval=6))
    elif days_in_period <= 7:
        axs[-1].xaxis.set_major_locator(mdates.DayLocator())
        axs[-1].xaxis.set_minor_locator(mdates.HourLocator(interval=6))
    else:
        axs[-1].xaxis.set_major_locator(mdates.DayLocator(interval=2))

    plt.xticks(rotation=30)
    plt.suptitle(f'Surface Temperature Comparison: {start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")}',
                 fontsize=16, y=0.995)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    return fig


def get_average_temperatures_per_pavement(pavements, meteo_data, surface_temperature, rainfall,
                                          water_temperature, bottom_temperature, water_events):
    """
    Generate a dataframe with average temperatures for multiple pavement types.

    Parameters:
    -----------
    pavements : list
        List of pavement types to process (e.g., ['CP', 'PICP', 'PGr', 'PA', 'PC'])
    meteo_data : pandas.DataFrame
        Meteorological data
    surface_temperature : pandas.DataFrame
        Surface temperature data
    rainfall : pandas.DataFrame
        Rainfall data
    water_temperature : pandas.DataFrame
        Water temperature data
    bottom_temperature : pandas.DataFrame
        Bottom temperature data
    water_events : dict
        Dictionary of event names and date ranges

    Returns:
    --------
    pandas.DataFrame
        Combined dataframe with average temperatures for each pavement type and event
    """
    # Dictionary to store results for each pavement
    all_results = {}

    # Process each pavement type
    for pavement in pavements:
        print(f"Processing {pavement}...")

        # Generate input dataframe for this pavement
        input_df = generate_input_file(pavement, meteo_data, surface_temperature,
                                       rainfall, water_temperature, bottom_temperature)

        # Get parameters file path
        parameters_file = f"input_data/parameters_{pavement}.ini"

        # Run temperature model
        results = temperature_model.model_pavement_temperature(input_df, parameters_file)

        # For impermeable concrete (CP), handle differently
        if pavement == 'CP':
            # For CP, we only have modeled surface water (no infiltration)
            # Create a custom dataframe for CP with only modeled data
            avg_temps = {}

            for event, (start, end) in water_events.items():
                start_dt = pd.to_datetime(start)
                end_dt = pd.to_datetime(end)

                # Filter for this event
                event_results = results[(results['date'] >= start_dt) & (results['date'] < end_dt)]

                if not event_results.empty:
                    # Calculate average surface water temperature
                    avg_temps[event] = {
                        'model_avg_temp': event_results['water_temp_surface'].mean().round(2),
                        'observed_avg_temp': None  # No observed data for CP
                    }
                else:
                    avg_temps[event] = {
                        'model_avg_temp': None,
                        'observed_avg_temp': None
                    }

            # Convert to dataframe
            avg_temps_df = pd.DataFrame.from_dict(avg_temps, orient='index')
            avg_temps_df.index.name = 'event'
            avg_temps_df.reset_index(inplace=True)
        else:
            # For permeable pavements, use the existing function
            avg_temps_df = generate_averages_water_temperature_df(water_events, results, input_df)

        # Add pavement type to the dataframe
        avg_temps_df['pavement'] = pavement

        # Store in the results dictionary
        all_results[pavement] = avg_temps_df

    # Combine all results into a single dataframe
    combined_df = pd.concat(all_results.values(), ignore_index=True)

    # Reorder columns for better readability
    combined_df = combined_df[['pavement', 'event', 'model_avg_temp', 'observed_avg_temp']]

    return combined_df


def get_average_temperatures_with_wide_format(pavements, meteo_data, surface_temperature, rainfall,
                                              water_temperature, bottom_temperature, water_events):
    """
    Generate a dataframe with average temperatures for multiple pavement types in wide format.

    Parameters:
    -----------
    pavements : list
        List of pavement types to process (e.g., ['CP', 'PICP', 'PGr', 'PA', 'PC'])
    meteo_data : pandas.DataFrame
        Meteorological data
    surface_temperature : pandas.DataFrame
        Surface temperature data
    rainfall : pandas.DataFrame
        Rainfall data
    water_temperature : pandas.DataFrame
        Water temperature data
    bottom_temperature : pandas.DataFrame
        Bottom temperature data
    water_events : dict
        Dictionary of event names and date ranges

    Returns:
    --------
    pandas.DataFrame
        Wide-format dataframe with columns: event, CP_observed, CP_modeled, PICP_observed, PICP_modeled, etc.
    """
    # First, get the dataframe with all the results
    results_df = get_average_temperatures_per_pavement(
        pavements,
        meteo_data,
        surface_temperature,
        rainfall,
        water_temperature,
        bottom_temperature,
        water_events
    )

    # Create a new dataframe with 'event' as index
    events = sorted(list(water_events.keys()))
    wide_df = pd.DataFrame(index=events)
    wide_df.index.name = 'event'

    # For each pavement, add two columns (observed and modeled)
    for pavement in pavements:
        # Filter data for this pavement
        pavement_data = results_df[results_df['pavement'] == pavement]

        # Create dictionaries to map event to values
        observed_dict = dict(zip(pavement_data['event'], pavement_data['observed_avg_temp']))
        modeled_dict = dict(zip(pavement_data['event'], pavement_data['model_avg_temp']))

        # Add columns to the wide dataframe
        wide_df[f'{pavement}_observed'] = wide_df.index.map(observed_dict)
        wide_df[f'{pavement}_modeled'] = wide_df.index.map(modeled_dict)

    # Reset index to make 'event' a column
    wide_df = wide_df.reset_index()

    return wide_df


def get_rmse_temperatures_wide_format(pavements, meteo_data, surface_temperature, rainfall,
                                      water_temperature, bottom_temperature, water_events):
    """
    Generate a dataframe with RMSE values for water temperature per pavement type in wide format.

    Parameters:
    -----------
    pavements : list
        List of pavement types to process (e.g., ['CP', 'PICP', 'PGr', 'PA', 'PC'])
    meteo_data : pandas.DataFrame
        Meteorological data
    surface_temperature : pandas.DataFrame
        Surface temperature data
    rainfall : pandas.DataFrame
        Rainfall data
    water_temperature : pandas.DataFrame
        Water temperature data
    bottom_temperature : pandas.DataFrame
        Bottom temperature data
    water_events : dict
        Dictionary of event names and date ranges

    Returns:
    --------
    pandas.DataFrame
        Wide-format dataframe with columns: event, CP_rmse, PICP_rmse, etc.
    """
    # Initialize dictionary to store RMSE values for each pavement and event
    rmse_values = {event: {} for event in water_events.keys()}

    # Process each pavement type
    for pavement in pavements:
        print(f"Processing {pavement}...")

        # Generate input dataframe for this pavement
        input_df = generate_input_file(pavement, meteo_data, surface_temperature,
                                       rainfall, water_temperature, bottom_temperature)

        # Get parameters file path (adjust path as needed)
        parameters_file = f"input_data/parameters_{pavement}.ini"

        # Run temperature model
        results = temperature_model.model_pavement_temperature(input_df, parameters_file)

        # Process each event
        for event_name, event_dates in water_events.items():
            # Parse event dates
            event_start = pd.to_datetime(event_dates[0])
            event_end = pd.to_datetime(event_dates[1])

            # Filter for the event period
            event_temp = input_df[(input_df['date'] >= event_start) & (input_df['date'] < event_end)]
            event_results = results[(results['date'] >= event_start) & (results['date'] < event_end)]

            # For impermeable concrete (CP), no observed water temperature
            if pavement == 'CP':
                rmse_values[event_name][pavement] = None
                continue

            # Get observed water temperature data, dropping NaN values
            observed_water_data = event_temp[['date', 'WaterTemperature']].dropna()

            # Skip if no observed data
            if observed_water_data.empty:
                rmse_values[event_name][pavement] = None
                continue

            # Only use model results for timestamps where we have observed data
            common_dates = observed_water_data['date']
            filtered_results = event_results[event_results['date'].isin(common_dates)]

            # Skip if no matching data
            if filtered_results.empty:
                rmse_values[event_name][pavement] = None
                continue

            # Get observed and modeled values
            observed_values = observed_water_data['WaterTemperature'].values
            modeled_values = filtered_results['water_temp_infil'].values

            # Calculate RMSE
            squared_diff = (observed_values - modeled_values) ** 2
            rmse = np.sqrt(np.mean(squared_diff))

            # Store RMSE value
            rmse_values[event_name][pavement] = round(rmse, 2)

    # Convert to dataframe
    rmse_df = pd.DataFrame.from_dict(rmse_values, orient='index')
    rmse_df.index.name = 'event'

    # Rename columns to add '_rmse' suffix
    rmse_df.columns = [f'{col}_rmse' for col in rmse_df.columns]

    # Reset index to make 'event' a column
    rmse_df = rmse_df.reset_index()

    return rmse_df

'''
7-days Surface Temperature Plots for the whole period with the 5 pavements.
'''
# surface_temperature = pd.read_csv(r'input_data\surface_temperature.csv', parse_dates=[0])
# rainfall = pd.read_csv(r'input_data\rainfall.csv', parse_dates=[0])
# water_temperature = pd.read_csv(r'input_data\water_temperature.csv', parse_dates=[0])
# meteo_data = pd.read_csv(r'input_data\meteo_data.csv', parse_dates=[0])
# bottom_temperature = pd.read_csv(r'input_data\bottom_temperature.csv', parse_dates=[0])
#
# # Define pavement types and date range
# pavements = ['CP', 'PICP', 'PGr', 'PA', 'PC']
#
# # Dictionary to store model results for each pavement
# model_results_dict = {}
#
# # Run the model for each pavement type
# for pavement in pavements:
#     print(f"Processing {pavement}...")
#
#     # Generate input file for this pavement
#     input_df = generate_input_file(
#         pavement,
#         meteo_data,
#         surface_temperature,
#         rainfall,
#         water_temperature,
#         bottom_temperature
#     )
#
#     # Path to parameters file for this pavement
#     parameters_file = rf"input_data\parameters_{pavement}.ini"
#
#     # Run the temperature model
#     results = temperature_model.model_pavement_temperature(input_df, parameters_file)
#
#     # Store the results
#     model_results_dict[pavement] = results
#
# # Create output directory for plots
# output_dir = "pavement_comparison_plots"
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)
#
# # Find the overall date range from the data
# start_overall = surface_temperature['date'].min()
# end_overall = surface_temperature['date'].max()
# print(f"Data range: {start_overall} to {end_overall}")
#
# # Loop through the entire date range with 7-day intervals
# current_start = start_overall
# while current_start < end_overall:
#     current_end = current_start + pd.Timedelta(days=7)
#
#     # Make sure we don't go past the end of the data
#     if current_end > end_overall:
#         current_end = end_overall
#
#     print(f"Processing period: {current_start.strftime('%Y-%m-%d')} to {current_end.strftime('%Y-%m-%d')}")
#
#     # Create filename for this period
#     period_str = f"{current_start.strftime('%Y%m%d')}_to_{current_end.strftime('%Y%m%d')}"
#     save_path = os.path.join(output_dir, f"pavement_comparison_{period_str}.png")
#
#     # Generate plot for this period
#     fig = plot_all_pavement_surface_temps(
#         surface_temperature,  # Contains observed data for all pavements
#         model_results_dict,  # Dictionary with model results for each pavement
#         current_start,  # Start of current period
#         current_end,  # End of current period
#         save_path=save_path
#     )
#
#     # Don't show all plots (would be too many), just save them
#     plt.close(fig)
#
#     # Move to next 7-day period
#     current_start = current_end
#
# print(f"All plots saved to {output_dir} directory.")


# '''
# Plot the surface temperature for the whole period.
# '''
# surface_temperature = pd.read_csv(r'input_data\surface_temperature.csv', parse_dates=[0])
# rainfall = pd.read_csv(r'input_data\rainfall.csv', parse_dates=[0])
# water_temperature = pd.read_csv(r'input_data\water_temperature.csv', parse_dates=[0])
# meteo_data = pd.read_csv(r'input_data\meteo_data.csv', parse_dates=[0])
# bottom_temperature = pd.read_csv(r'input_data\bottom_temperature.csv', parse_dates=[0])
#
# # Define pavement types and date range
# pavements = ['CP', 'PICP', 'PGr', 'PA', 'PC']
#
# # Dictionary to store model results for each pavement
# model_results_dict = {}
#
# # Run the model for each pavement type
# for pavement in pavements:
#     print(f"Processing {pavement}...")
#
#     # Generate input file for this pavement
#     input_df = generate_input_file(
#         pavement,
#         meteo_data,
#         surface_temperature,
#         rainfall,
#         water_temperature,
#         bottom_temperature
#     )
#
#     # Path to parameters file for this pavement
#     parameters_file = rf"input_data\parameters_{pavement}.ini"
#
#     # Run the temperature model
#     results = temperature_model.model_pavement_temperature(input_df, parameters_file)
#
#     # Store the results
#     model_results_dict[pavement] = results
#
# # Create output directory for plots
# output_dir = "pavement_comparison_plots"
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)
#
# # Generate plot for this period
# fig = plot_all_pavement_surface_temps(
#     surface_temperature,
#     model_results_dict,
#     end_date="2024-07-01 00:00:00-06:00",
#     save_path="output_plot.png"
# )


'''
Plot the Water Temperature
'''
#modeled should be continous and observed should be only dots (not connected). Add that the water is effluent. Surface pavement temperature instead of surface temperature
pavement = 'CP'

surface_temperature = pd.read_csv(r'input_data\surface_temperature.csv', parse_dates=[0])
rainfall = pd.read_csv(r'input_data\rainfall.csv', parse_dates=[0])
water_temperature = pd.read_csv(r'input_data\water_temperature.csv', parse_dates=[0])
meteo_data = pd.read_csv(r'input_data\meteo_data.csv', parse_dates=[0])
bottom_temperature = pd.read_csv(r'input_data\bottom_temperature.csv', parse_dates=[0])

input_df = generate_input_file(pavement, meteo_data, surface_temperature, rainfall, water_temperature, bottom_temperature)
parameters_file = rf"C:\Users\Artur\PycharmProjects\pavement_temperature_model\input_data\parameters_{pavement}.ini"

results = temperature_model.model_pavement_temperature(input_df, parameters_file)

water_events = {
    'event_1': ['2023-10-05 00:00:00-06:00', '2023-10-05 21:00:00-06:00'],
    'event_2': ['2023-10-26 00:00:00-06:00', '2023-10-27 00:00:00-06:00'],
    'event_3': ['2023-11-09 15:00:00-06:00', '2023-11-10 06:00:00-06:00'],
    'event_4': ['2024-01-21 00:00:00-06:00', '2024-01-23 02:00:00-06:00'],
    'event_5': ['2024-01-31 18:00:00-06:00', '2024-02-03 10:20:00-06:00'],
    'event_6': ['2024-04-28 00:00:00-06:00', '2024-04-28 22:00:00-06:00'],
}

if pavement != 'CP':
    avg_temps_df = generate_averages_water_temperature_df(water_events, results, input_df)
    figures = plot_event_data_with_model(
        dataframe=input_df,
        results_df=results,
        pavement_type=pavement,
        water_events=water_events,
        avg_temps_df=avg_temps_df
    )
elif pavement == 'CP':
    figures = plot_event_data_impermeable(
        dataframe=input_df,
        results_df=results,
        pavement_type=pavement,
        water_events=water_events,
    )


'''
Average Water Temperature For All Pavements
'''
# pavements = ['CP', 'PICP', 'PGr', 'PA', 'PC']
#
# # Load your data
# surface_temperature = pd.read_csv(r'input_data\surface_temperature.csv', parse_dates=[0])
# rainfall = pd.read_csv(r'input_data\rainfall.csv', parse_dates=[0])
# water_temperature = pd.read_csv(r'input_data\water_temperature.csv', parse_dates=[0])
# meteo_data = pd.read_csv(r'input_data\meteo_data.csv', parse_dates=[0])
# bottom_temperature = pd.read_csv(r'input_data\bottom_temperature.csv', parse_dates=[0])
#
# # Define water events
# water_events = {
#     'event_1': ['2023-10-05 00:00:00-06:00', '2023-10-05 21:00:00-06:00'],
#     'event_2': ['2023-10-26 00:00:00-06:00', '2023-10-27 00:00:00-06:00'],
#     'event_3': ['2023-11-09 15:00:00-06:00', '2023-11-10 06:00:00-06:00'],
#     'event_4': ['2024-01-21 00:00:00-06:00', '2024-01-23 02:00:00-06:00'],
#     'event_5': ['2024-01-31 18:00:00-06:00', '2024-02-03 10:20:00-06:00'],
#     'event_6': ['2024-04-21 00:00:00-06:00', '2024-04-21 12:00:00-06:00'],
# }
#
# # Get average temperatures in wide format
# wide_avg_temps = get_average_temperatures_with_wide_format(
#     pavements,
#     meteo_data,
#     surface_temperature,
#     rainfall,
#     water_temperature,
#     bottom_temperature,
#     water_events
# )
#
# # Display the results
# print(wide_avg_temps)
#
# # Save to CSV if needed
# wide_avg_temps.to_csv('average_temperatures_wide_format.csv', index=False)

'''
RMSE Water Temperature for All Pavement
'''
# pavements = ['CP', 'PICP', 'PGr', 'PA', 'PC']
#
# # Load your data
# surface_temperature = pd.read_csv(r'input_data\surface_temperature.csv', parse_dates=[0])
# rainfall = pd.read_csv(r'input_data\rainfall.csv', parse_dates=[0])
# water_temperature = pd.read_csv(r'input_data\water_temperature.csv', parse_dates=[0])
# meteo_data = pd.read_csv(r'input_data\meteo_data.csv', parse_dates=[0])
# bottom_temperature = pd.read_csv(r'input_data\bottom_temperature.csv', parse_dates=[0])
#
# # Define water events
# water_events = {
#     'event_1': ['2023-10-05 00:00:00-06:00', '2023-10-05 21:00:00-06:00'],
#     'event_2': ['2023-10-26 00:00:00-06:00', '2023-10-27 00:00:00-06:00'],
#     'event_3': ['2023-11-09 15:00:00-06:00', '2023-11-10 06:00:00-06:00'],
#     'event_4': ['2024-01-21 00:00:00-06:00', '2024-01-23 02:00:00-06:00'],
#     'event_5': ['2024-01-31 18:00:00-06:00', '2024-02-03 10:20:00-06:00'],
#     'event_6': ['2024-04-28 00:00:00-06:00', '2024-04-28 22:00:00-06:00'],
# }
#
# # Get RMSE values for all pavement types
# rmse_df = get_rmse_temperatures_wide_format(
#     pavements,
#     meteo_data,
#     surface_temperature,
#     rainfall,
#     water_temperature,
#     bottom_temperature,
#     water_events
# )
#
# # Display the results
# print(rmse_df)
#
# # Save to CSV if needed
# rmse_df.to_csv('water_temperature_rmse_by_pavement.csv', index=False)