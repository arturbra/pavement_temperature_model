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


input_file = r"C:\Users\artur\OneDrive\Doutorado\UTSA\PP\Model\Process-based-Nitrogen-Model-main\pavement_temperature_model\input_data\input_data_PA.csv"
parameters_file = r"input_data/parameters.ini"
sim_df = pd.read_csv(input_file)
model_results = temperature_model.model_pavement_temperature(sim_df, parameters_file)
# plot_energy_balance(model_results)

NSE = temperature_model.NSE(sim_df, model_results)
print(NSE)

index_start = 0
index_end = index_start + 169
plot_simulated_observed(model_results.iloc[index_start:index_end, :], sim_df.iloc[index_start:index_end, :])


