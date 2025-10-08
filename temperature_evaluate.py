import temperature_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.dates as mdates


def plot_energy_balance(df):
    """
    Create a two-panel plot:
      - Top panel: Flux terms (h_s, h_li, h_l0, h_rad, h_evap, h_conv, h_r0, h_net)
      - Bottom panel: Surface temperature over time.

    Parameters:
      df (pandas.DataFrame): DataFrame returned from model_pavement_temperature.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Plot flux terms
    ax1.plot(df['h_s'], label='h_s')
    ax1.plot(df['h_li'], label='h_li')
    ax1.plot(df['h_l0'], label='h_l0')
    ax1.plot(df['h_rad'], label='h_rad')
    ax1.plot(df['h_evap'], label='h_evap')
    ax1.plot(df['h_conv'], label='h_conv')
    ax1.plot(df['h_r0'], label='h_r0')
    ax1.plot(df['h_net'], label='h_net', linewidth=2, color='black')
    ax1.set_ylabel('Heat Flux (W/m²)')
    ax1.set_title('Energy Balance Flux Terms')
    ax1.legend(loc='upper right')
    ax1.grid(True)

    # Plot surface temperature
    ax2.plot(df['surface_temp'], label='Surface Temperature', color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Temperature (°C)')
    ax2.set_title('Surface Temperature over Time')
    ax2.legend(loc='upper right')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


def plot_simulated_observed(model_results, obs_df):
    plt.plot(model_results['surface_temp'], label='Modeled')
    plt.plot(obs_df['PavementTemperature'], label='Observed')
    plt.legend(loc='best')
    plt.show()


def plot_pavement_temperature(combined_df, start_date=None, end_date=None, pavement_type="CP",
                              y_min=None, y_max=None, save_fig=False, output_path=None):
    """
    Plot pavement temperature comparison (modeled vs observed) for a selected date range,
    optimized for PowerPoint presentation (readable when resized to 2.84 x 5.71").

    Parameters:
        combined_df (DataFrame): DataFrame containing date, surface_temp, PavementTemperature
        start_date (str): Start date in format 'YYYY-MM-DD HH:MM'
        end_date (str): End date in format 'YYYY-MM-DD HH:MM'
        pavement_type (str): Type of pavement (CP, PICP, PA, PC, PGr)
        y_min (float): Minimum value for y-axis
        y_max (float): Maximum value for y-axis
        save_fig (bool): Whether to save the figure
        output_path (str): Path to save the figure
    """
    # Set figure style for better PowerPoint visibility
    plt.rcParams.update({
        'font.size': 16,  # Increase base font size
        'axes.titlesize': 18,  # Larger title
        'axes.labelsize': 16,  # Larger axis labels
        'xtick.labelsize': 14,  # Larger tick labels
        'ytick.labelsize': 14,  # Larger tick labels
        'legend.fontsize': 14,  # Larger legend text
        'lines.linewidth': 3,  # Thicker lines
        'axes.linewidth': 2,  # Thicker axis lines
        'axes.grid': True,  # Enable grid by default
        'grid.alpha': 0.7  # Semi-transparent grid
    })

    # Convert pavement abbreviation to full name
    pavement_names = {
        "CP": "Conventional Pavement",
        "PC": "Permeable Concrete",
        "PICP": "Permeable Interlocking Concrete Pavers",
        "PA": "Porous Asphalt",
        "PGr": "Permeable Gravel"
    }

    pavement_full_name = pavement_names.get(pavement_type, pavement_type)

    # Ensure date column is datetime
    if combined_df['date'].dtype != 'datetime64[ns]':
        combined_df['date'] = pd.to_datetime(combined_df['date'])

    # Filter data for selected date range
    if start_date and end_date:
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        mask = (combined_df['date'] >= start_date) & (combined_df['date'] <= end_date)
        event_data = combined_df[mask].copy()
    else:
        event_data = combined_df.copy()

    # Calculate statistics for the selected period
    rmse = np.sqrt(np.mean((event_data['surface_temp'] - event_data['PavementTemperature']) ** 2))

    # Create figure with dimensions optimized for PowerPoint
    # Using a wider aspect ratio to match PowerPoint slide dimensions (16:9)
    fig, ax1 = plt.subplots(figsize=(16, 8))

    # Format dates for x-axis (MM/DD/YYYY)
    # Simplified date format - removing time for cleaner x-axis
    event_data['date_formatted'] = event_data['date'].dt.strftime('%m/%d/%Y')
    date_range = range(len(event_data))

    # Plot temperatures with thicker lines for better visibility
    ax1.plot(date_range, event_data['PavementTemperature'],
             label='Observed', linewidth=3, color='#ff7f0e')
    ax1.plot(date_range, event_data['surface_temp'],
             label='Modeled', linewidth=3, color='#1f77b4', linestyle='--')

    # Set x-ticks to show fewer dates to avoid cluttering
    # Reduce number of ticks for better readability
    n_ticks = min(7, len(event_data))  # Reduced from 10 to 7
    tick_indices = np.linspace(0, len(event_data) - 1, n_ticks, dtype=int)
    ax1.set_xticks(tick_indices)

    # Get unique dates only to reduce x-axis clutter
    unique_dates = []
    seen_dates = set()
    for i in tick_indices:
        date_str = event_data['date_formatted'].iloc[i]
        if date_str not in seen_dates:
            unique_dates.append((i, date_str))
            seen_dates.add(date_str)

    # Set only unique dates on x-axis
    ax1.set_xticks([i for i, _ in unique_dates])
    ax1.set_xticklabels([date for _, date in unique_dates], rotation=45, ha='right', fontsize=18)

    # Set y-axis limits if provided
    if y_min is not None:
        ax1.set_ylim(bottom=y_min)
    if y_max is not None:
        ax1.set_ylim(top=y_max)

    # Add legend with larger markers
    ax1.legend(loc='upper right', fontsize=20, markerscale=2)

    # Customize plot appearance
    ax1.set_ylabel('Pavement Temperature (°C)', fontsize=20, labelpad=10)
    title = f"{pavement_full_name} Temperature: Modeled vs Observed\nRMSE: {rmse:.2f}°C"
    plt.title(title, fontsize=20, fontweight='bold', pad=15)

    # Add grid for better readability with bolder lines
    ax1.grid(True, linestyle='--', alpha=0.7, linewidth=1.2)

    # Add more space at the bottom to prevent date labels from being cut off
    plt.subplots_adjust(bottom=0.15)

    # Adjust layout, but with more space for labels
    plt.tight_layout()

    # Save figure if requested with higher DPI for better quality in PowerPoint
    if save_fig and output_path:
        plt.savefig(output_path, dpi=400, bbox_inches='tight')

    plt.show()

    # Reset rcParams to default values to avoid affecting other plots
    plt.rcdefaults()

    return fig, ax1


def rmse(sim, obs):
    return np.sqrt(np.mean((obs - sim) ** 2))


def nse(sim, obs):
    return 1 - np.sum((obs - sim) ** 2) / np.sum((obs - np.mean(obs)) ** 2)


def evaluate_model(model_result, input_df, pavement='CP', end_date='2024-07-01 00:00:00-06:00'):
    # Ensure datetime and sort
    model_result['date'] = pd.to_datetime(model_result['date'])
    input_df['date'] = pd.to_datetime(input_df['date'])
    end_date = pd.to_datetime(end_date)

    # Merge modeled and observed data
    merged = pd.merge(model_result[['date', 'surface_temp']],
                      input_df[['date', f'{pavement}_surface_temperature']],
                      on='date',
                      how='inner')

    # Rename for clarity
    merged = merged.rename(columns={
        'surface_temp': 'modeled',
        f'{pavement}_surface_temperature': 'observed'
    }).dropna()

    # Filter up to specified end_date
    merged = merged[merged['date'] <= end_date]

    # Check if there's enough data
    if len(merged) < 2:
        print("Not enough data before specified end_date.")
        return None

    # Split data
    split_index = int(len(merged) * 0.4)
    calibration = merged.iloc[:split_index]
    validation = merged.iloc[split_index:]

    # Compute metrics
    calibration_rmse = rmse(calibration['modeled'].values, calibration['observed'].values)
    calibration_nse = nse(calibration['modeled'].values, calibration['observed'].values)

    validation_rmse = rmse(validation['modeled'].values, validation['observed'].values)
    validation_nse = nse(validation['modeled'].values, validation['observed'].values)

    # Print results
    print(f"Evaluation up to {end_date.strftime('%Y-%m-%d %H:%M:%S')}:")
    print("Calibration Period:")
    print(f"  RMSE: {calibration_rmse:.2f}")
    print(f"  NSE : {calibration_nse:.2f}\n")

    print("Validation Period:")
    print(f"  RMSE: {validation_rmse:.2f}")
    print(f"  NSE : {validation_nse:.2f}\n")

    # Return metrics
    return {
        'calibration': {
            'RMSE': calibration_rmse,
            'NSE': calibration_nse
        },
        'validation': {
            'RMSE': validation_rmse,
            'NSE': validation_nse
        }
    }


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


def plot_pavement_surface_temp(observed_data, model_results_dict, pavement, start_date, end_date, save_path=None):
    """
    Plot surface temperature for a given pavement type, comparing observed and modeled data.

    Parameters:
    -----------
    observed_data : pandas.DataFrame
        Must contain 'date' and '{pavement}_surface_temperature' column
    model_results_dict : dict
        Dictionary with pavement types as keys and DataFrames with 'date' and 'surface_temp' as values
    pavement : str
        Pavement type to plot (e.g., 'PA', 'CP', 'PICP', etc.)
    start_date : str or datetime
        Start of the plotting period
    end_date : str or datetime
        End of the plotting period
    save_path : str, optional
        If provided, saves the plot to this path

    Returns:
    --------
    matplotlib.figure.Figure or None
    """
    # Custom colors
    color_mapping = {
        'CP': (0.121, 0.467, 0.706),
        'PICP': (1.0, 0.498, 0.055),
        'PGr': (0.173, 0.627, 0.173),
        'PA': (0.580, 0.404, 0.741),
        'PC': (0.839, 0.153, 0.157),
    }
    color = color_mapping.get(pavement, (0, 0, 0))

    # Prepare datetime
    observed_data['date'] = pd.to_datetime(observed_data['date'])
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Filter observed data
    temp_col = f'{pavement}_surface_temperature'
    observed_period = observed_data[(observed_data['date'] >= start_date) &
                                    (observed_data['date'] < end_date)].copy()

    if observed_period.empty or temp_col not in observed_period.columns:
        print(f"No observed data available for {pavement} in the selected period.")
        return None

    # Setup plot
    fig, ax = plt.subplots(figsize=(12, 5))

    # Plot observed
    obs = observed_period[['date', temp_col]].dropna()
    ax.plot(obs['date'], obs[temp_col], label='Observed Surface',
            color=color, linestyle='-', linewidth=2, marker='o', markersize=3)

    # Plot modeled
    if pavement in model_results_dict:
        model_df = model_results_dict[pavement].copy()
        model_df['date'] = pd.to_datetime(model_df['date'])

        model_period = model_df[(model_df['date'] >= start_date) & (model_df['date'] < end_date)]
        ax.plot(model_period['date'], model_period['surface_temp'], label='Modeled Surface',
                color=color, linestyle='--', linewidth=2)

        # RMSE
        merged = pd.merge(obs, model_period[['date', 'surface_temp']], on='date', how='inner')
        if not merged.empty:
            rmse = np.sqrt(((merged[temp_col] - merged['surface_temp']) ** 2).mean())
            ax.annotate(f'RMSE: {rmse:.2f}°C',
                        xy=(0.95, 0.05), xycoords='axes fraction',
                        ha='right', va='bottom', fontsize=11,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.6))
            print(f"{pavement} RMSE: {rmse:.2f}°C")

    # Formatting
    ax.set_title(f'{pavement} Surface Temperature\n{start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")}')
    ax.set_ylabel('Temperature (°C)')
    ax.set_xlabel('Date')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='upper right')

    # X-axis ticks
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    days = (end_date - start_date).days
    if days <= 2:
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
    elif days <= 7:
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.xaxis.set_minor_locator(mdates.HourLocator(interval=6))
    else:
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))

    plt.xticks(rotation=30)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig

surface_temperature = pd.read_csv(r'input_data\surface_temperature.csv', parse_dates=[0])
rainfall = pd.read_csv(r'input_data\rainfall.csv', parse_dates=[0])
water_temperature = pd.read_csv(r'input_data\water_temperature.csv', parse_dates=[0])
meteo_data = pd.read_csv(r'input_data\meteo_data.csv', parse_dates=[0])
bottom_temperature = pd.read_csv(r'input_data\bottom_temperature.csv', parse_dates=[0])

# Store all results
metrics_list = []

# List of pavements to evaluate
pavements = ['CP', 'PICP', 'PGr', 'PA', 'PC']
end_date = '2024-07-01 00:00:00-06:00'
for pavement in pavements:
    print(f"Processing {pavement}...")

    input_df = generate_input_file(
        pavement,
        meteo_data,
        surface_temperature,
        rainfall,
        water_temperature,
        bottom_temperature
    )

    parameters_file = rf"input_data/parameters_{pavement}.ini"
    model_results = temperature_model.model_pavement_temperature(input_df, parameters_file)

    metrics = evaluate_model(model_results, input_df, pavement, end_date=end_date)

    if metrics is not None:
        metrics_list.append({
            'Pavement': pavement,
            'Calibration_RMSE': metrics['calibration']['RMSE'],
            'Calibration_NSE': metrics['calibration']['NSE'],
            'Validation_RMSE': metrics['validation']['RMSE'],
            'Validation_NSE': metrics['validation']['NSE']
        })

# Convert to DataFrame
metrics_df = pd.DataFrame(metrics_list)
metrics_df.to_csv('results/model_performance_metrics.csv', index=False, float_format='%.2f')

# Optional: display it
print(metrics_df)





# # Dictionary to store model results for each pavement
# model_results_dict = {}
#
# # Generate input file for this pavement
# input_df = generate_input_file(
#     pavement,
#     meteo_data,
#     surface_temperature,
#     rainfall,
#     water_temperature,
#     bottom_temperature
# )
#
# # Path to parameters file for this pavement
# parameters_file = rf"input_data\parameters_{pavement}.ini"
#
# # Run the temperature model
# results = temperature_model.model_pavement_temperature(input_df, parameters_file)
#
# # Store the results
# model_results_dict[pavement] = results
#
# # Create output directory for plots
# output_dir = "single_pavement_comparison_plots"
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
#     fig = plot_pavement_surface_temp(
#         observed_data=surface_temperature,
#         model_results_dict=model_results_dict,
#         pavement=pavement,
#         start_date=current_start,
#         end_date=current_end,
#         save_path=os.path.join(output_dir, f"{pavement}_surface_temp_{period_str}.png")
#     )
#
#     # Don't show all plots (would be too many), just save them
#     plt.close(fig)
#
#     # Move to next 7-day period
#     current_start = current_end








# model_results_composite, temperature = temperature_model.model_pavement_temperature_simplified(sim_df, parameters_file)
# bottom_temperature = pd.DataFrame({'date':sim_df['date'], 'bottom_temperature_modeled':model_results_composite['subsurface_temp'], 'bottom_temperature_observed': sim_df['BottomTemperature']})
# bottom_temperature['date'] = pd.to_datetime(bottom_temperature['date'])
#
# plt.rcParams.update({'font.size': 16})
# index = 1000
#
# # Let's assume uniform layer thicknesses and assign depth values
# layer_thickness = 0.05  # meters per layer (example)
# depth = np.arange(0, len(temperature[index])) * layer_thickness  # in meters
#
# # Plot
# plt.figure(figsize=(6, 8))
# plt.plot(temperature[index], depth, marker='o', linewidth=2)
# plt.gca().invert_yaxis()  # So the top surface is at the top
# plt.xlabel('Temperature (°C)', fontsize=16)
# plt.ylabel('Depth (m)', fontsize=16)
# plt.title('Permeable Concrete Temperature Profile', fontsize=16, fontweight='bold')
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.tight_layout()
# plt.savefig(f'temperature_profile_{pavement}.png', dpi=300, bbox_inches='tight')
# plt.show()
#
#
# plt.figure(figsize=(10, 6))
# plt.plot(bottom_temperature['bottom_temperature_modeled'],
#          label="Modeled",
#          linewidth=2,
#          color='#1f77b4', linestyle='dashed', markersize=2)
# plt.plot(bottom_temperature['bottom_temperature_observed'],
#          label='Observed',
#          linewidth=2,
#          color='#ff7f0e')
#
# plt.xlabel('Time', fontsize=14)
# plt.ylabel('Bottom Temperature (°C)', fontsize=14)
# plt.title(f'{pavement} Pavement Bottom Temperature: Modeled vs Observed', fontsize=16, fontweight='bold')
# plt.legend(fontsize=12, loc='best', frameon=True, facecolor='white', edgecolor='gray')
# plt.grid(True, linestyle='--', alpha=0.7)
# plt.tick_params(axis='both', which='major', labelsize=12)
# plt.gcf().autofmt_xdate()
# plt.tight_layout()
# plt.savefig(f'pavement_temperature_comparison_{pavement}.png', dpi=300, bbox_inches='tight')
# plt.show()
# #
# #
# #
# # NSE_bottom = temperature_model.NSE_bottom(bottom_temperature)
# # RMSE_bottom = temperature_model.RMSE_bottom(bottom_temperature)
# # print(f"NSE_Bottom: {NSE_bottom:.2f}")
# # print(f"RMSE Bottom: {RMSE_bottom:.2f}")
#
# # plot_energy_balance(model_results)
# #
# calib_size = int(0.4 * len(sim_df))
# calib_df = sim_df.iloc[:calib_size].reset_index(drop=True)
# calib_results = model_results.iloc[:calib_size].reset_index(drop=True)
#
# val_df = sim_df.iloc[calib_size:].reset_index(drop=True)
# val_results = model_results.iloc[calib_size:].reset_index(drop=True)
#
# NSE_calib = temperature_model.NSE(calib_df, calib_results)
# NSE_valid = temperature_model.NSE(val_df, val_results)
#
# RMSE_calib = temperature_model.RMSE(calib_df, calib_results)
# RMSE_valid = temperature_model.RMSE(val_df, val_results)
#
# print(f"NSE Calibration: {NSE_calib:.2f}")
# print(f"RMSE Calibration: {RMSE_calib:.2f}")
#
# print(f"NSE Validation: {NSE_valid:.2f}")
# print(f"RMSE Validation: {RMSE_valid:.2f}")
#
#
# # Calibration period plot
# plt.figure(figsize=(10, 4))
# time_calib = calib_df.index
# plt.plot(time_calib, calib_df['PavementTemperature'], label="Observed (Calibration)")
# plt.plot(time_calib, calib_results['surface_temp'], label="Modeled (Calibration)")
# plt.xlabel("Time Step")
# plt.ylabel("Pavement Temperature (°C)")
# plt.title("Calibration Period")
# plt.legend()
# plt.tight_layout()
# plt.show()
#
# # Validation period plot
# time_val = val_df.index
# plt.figure(figsize=(10, 4))
# plt.plot(time_val, val_df['PavementTemperature'], label="Observed (Validation)")
# plt.plot(time_val, val_results['surface_temp'], label="Modeled (Validation)")
# plt.xlabel("Time Step")
# plt.ylabel("Pavement Temperature (°C)")
# plt.title("Validation Period")
# plt.legend()
# plt.tight_layout()
# plt.show()
#
