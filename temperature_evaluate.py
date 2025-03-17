import temperature_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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


input_file = r"C:\Users\artur\OneDrive\Doutorado\UTSA\PP\Model\Process-based-Nitrogen-Model-main\pavement_temperature_model\input_data\input_data_CP.csv"
parameters_file = r"input_data/parameters_CP.ini"
sim_df = pd.read_csv(input_file)
model_results = temperature_model.model_pavement_temperature(sim_df, parameters_file)
# plot_energy_balance(model_results)

calib_size = int(0.4 * len(sim_df))
calib_df = sim_df.iloc[:calib_size].reset_index(drop=True)
calib_results = model_results.iloc[:calib_size].reset_index(drop=True)

val_df = sim_df.iloc[calib_size:].reset_index(drop=True)
val_results = model_results.iloc[calib_size:].reset_index(drop=True)

NSE_calib = temperature_model.NSE(calib_df, calib_results)
NSE_valid = temperature_model.NSE(val_df, val_results)

RMSE_calib = temperature_model.RMSE(calib_df, calib_results)
RMSE_valid = temperature_model.RMSE(val_df, val_results)

print(f"NSE Calibration: {NSE_calib:.2f}")
print(f"RMSE Calibration: {RMSE_calib:.2f}")

print(f"NSE Validation: {NSE_valid:.2f}")
print(f"RMSE Validation: {RMSE_valid:.2f}")


# Calibration period plot
plt.figure(figsize=(10, 4))
time_calib = calib_df.index
plt.plot(time_calib, calib_df['PavementTemperature'], label="Observed (Calibration)")
plt.plot(time_calib, calib_results['surface_temp'], label="Modeled (Calibration)")
plt.xlabel("Time Step")
plt.ylabel("Pavement Temperature (°C)")
plt.title("Calibration Period")
plt.legend()
plt.tight_layout()
plt.show()

# Validation period plot
time_val = val_df.index
plt.figure(figsize=(10, 4))
plt.plot(time_val, val_df['PavementTemperature'], label="Observed (Validation)")
plt.plot(time_val, val_results['surface_temp'], label="Modeled (Validation)")
plt.xlabel("Time Step")
plt.ylabel("Pavement Temperature (°C)")
plt.title("Validation Period")
plt.legend()
plt.tight_layout()
plt.show()


start_dates = ["2023-08-21 15:00", "2023-10-04 04:00", "2023-10-25 08:00", "2023-11-08 18:00", "2023-12-22 22:00",
               "2024-01-20 23:00", "2024-02-01 19:00", "2024-04-08 22:00", "2024-04-19 22:00", "2024-04-27 06:00",
               "2024-05-12 08:00"]

end_dates = ["2023-08-23 22:00", "2023-10-06 22:00", "2023-10-28 03:10", "2023-11-11 06:00", "2023-12-25 18:00",
             "2024-01-24 10:00", "2024-02-04 10:00", "2024-04-11 06:00", "2024-04-22 15:00", "2024-04-29 20:00",
             "2024-05-15 00:00"]

combined_df = pd.concat([sim_df, model_results], axis=1)
combined_df['date'] = pd.to_datetime(combined_df['date'])
df_tz = combined_df['date'].dt.tz
start_dates = pd.to_datetime(start_dates).tz_localize(df_tz)
end_dates = pd.to_datetime(end_dates).tz_localize(df_tz)

for start_date, end_date in zip(start_dates, end_dates):
    event_data = combined_df[(combined_df['date'] >= start_date) & (combined_df['date'] <= end_date)]

    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.plot(range(len(event_data)), event_data['PavementTemperature'], label='Observed')
    ax1.plot(range(len(event_data)), event_data['surface_temp'], label='Modeled')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Temperature (degC)')
    ax1.legend(loc='upper right')

    ax2 = ax1.twinx()
    ax2.bar(range(len(event_data)), event_data['Rainfall'], color='black', alpha=0.5, width=0.4, label="Rainfall")
    ax2.invert_yaxis()

    plt.show()