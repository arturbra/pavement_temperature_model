import temperature_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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


# Example usage:
# Load your data as before
pavement = 'CP'  # You can change this to PICP, PA, PC, or PGr as needed
input_file = rf"C:\Users\Artur\PycharmProjects\pavement_temperature_model\input_data\1h\input_data_{pavement}.csv"
parameters_file = rf"input_data/1h/parameters_{pavement}.ini"
sim_df = pd.read_csv(input_file)

# Run your model as before
model_results = temperature_model.model_pavement_temperature(sim_df, parameters_file)

# Combine data into one DataFrame for plotting
combined_df = pd.concat([sim_df, model_results], axis=1)
combined_df['date'] = pd.to_datetime(combined_df['date'])
combined_df['PavementTemperature'] = combined_df['PavementTemperature'].shift(0)
combined_df = combined_df.dropna(subset=['surface_temp'])

# Example: Plot with custom y-axis limits
plot_pavement_temperature(
    combined_df,
    start_date="2023-08-04 04:00-06:00",
    end_date="2023-08-12 00:00-06:00",
    pavement_type=pavement,
    y_min=27,  # Set minimum temperature on y-axis
    y_max=65,  # Set maximum temperature on y-axis
    save_fig=True,
    output_path=f"pavement_temp_{pavement}_for_powerpoint.png"
)

















#
#
pavement = 'CP'
input_file = rf"C:\Users\Artur\PycharmProjects\pavement_temperature_model\input_data\input_data_{pavement}.csv"
# parameters_file = rf"input_data/1h/parameters_{pavement}.ini"
parameters_file = rf"input_data/parameters_{pavement}.ini"
sim_df = pd.read_csv(input_file)

model_results = temperature_model.model_pavement_temperature(sim_df, parameters_file)
model_results_composite, temperature = temperature_model.model_pavement_temperature_simplified(sim_df, parameters_file)
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
#

# start_dates = ["2023-08-21 15:00", "2023-10-04 04:00", "2023-10-25 08:00", "2023-11-08 20:00", "2023-12-22 22:00",
#                "2024-01-20 23:00", "2024-02-01 19:00", "2024-04-08 22:00", "2024-04-19 22:00", "2024-04-27 06:00",
#                "2024-05-12 08:00"]
#
# end_dates = ["2023-08-23 22:00", "2023-10-06 22:00", "2023-10-28 03:10", "2023-11-11 06:00", "2023-12-25 18:00",
#              "2024-01-24 10:00", "2024-02-04 10:00", "2024-04-11 06:00", "2024-04-22 15:00", "2024-04-29 20:00",
#              "2024-05-15 00:00"]

start_dates = ["2023-11-09 10:00"]

end_dates = ["2023-11-10 05:00"]

combined_df = pd.concat([sim_df, model_results], axis=1)
combined_df['date'] = pd.to_datetime(combined_df['date'])
combined_df['surface_temp'] = combined_df['surface_temp'].shift(-1)


df_tz = combined_df['date'].dt.tz
start_dates = pd.to_datetime(start_dates).tz_localize(df_tz)
end_dates = pd.to_datetime(end_dates).tz_localize(df_tz)

plt.rcParams.update({'font.size': 14})

for start_date, end_date in zip(start_dates, end_dates):
    event_data = combined_df[(combined_df['date'] >= start_date) & (combined_df['date'] <= end_date)]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Surface temperatures
    ax1.plot(event_data['date'], event_data['PavementTemperature'], label='Observed',
             linewidth=2, color='#ff7f0e')
    ax1.plot(event_data['date'], event_data['surface_temp'], label='Modeled',
             linewidth=2, color='#1f77b4', linestyle='dashed')
    ax1.set_ylabel('Surface Temperature (°C)', fontsize=14)
    ax1.tick_params(axis='both', which='major', labelsize=14)
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Rainfall bars
    ax2 = ax1.twinx()
    ax2.bar(event_data['date'], event_data['Rainfall'],
            color='black', alpha=1, width=0.003, label="Rainfall")  # wider bar width
    ax2.set_ylim(4, 0)  # Inverted
    ax2.set_ylabel('Rainfall (in.)', fontsize=14)
    # Format x-axis as MM/DD/YYYY HH:mm
    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=4))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y %H:%M'))
    fig.autofmt_xdate()

    # Combine legends
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2,
               fontsize=12, loc='upper right', bbox_to_anchor=(1, 0.85),
               frameon=True, facecolor='white', edgecolor='gray')
    # Title
    ax1.set_title('Conventional Pavement Surface Temperature and Rainfall', fontsize=16, fontweight='bold')

    plt.tight_layout()
    plt.savefig("rainfall_temperature_cp.png", dpi=300, bbox_inches='tight')

    plt.show()