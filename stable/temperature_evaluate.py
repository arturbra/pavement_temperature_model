import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
#
#
def simulate_pavement_temperature(sim_df, params):
    """
    Run the pavement temperature simulation with calibrated dynamic parameters and
    a multiplier for the Layer 1 conductivity.

    Calibrated parameters (in order):
      0: reflectivity         (e.g., 0.1 to 1.0)
      1: emissivity           (e.g., 0.1 to 1.0)
      2: f_lam_layer1         (multiplier for Layer 1 conductivity, e.g., 0.1 to 4.0)

    Fixed values:
      - φ (porosity for Layer 1) is fixed at 0.4.
      - hc₀ is fixed at 5.6 and hc_slope at 4, so that h_c = 5.6 + 4*v.
      - shift is fixed to 0.
      - Layers 2–4 properties are set to fixed default values.
    """
    # Unpack calibrated parameters
    reflectivity = params[0]
    emissivity = params[1]
    f_lam_layer1 = params[3]

    # Fixed parameters
    phi = 0.4  # Fixed porosity for Layer 1
    hc0 = 5.6  # Fixed base convective coefficient
    hc_slope = 4  # Fixed multiplier for wind speed in h_c
    ER = 1e-6 # Conversion factor

    # Spatial discretization and depth nodes (adjusted for a thinner pavement)
    total_depth = 0.3  # m
    dx = 0.05  # m (spatial resolution)
    N = int(np.ceil(total_depth / dx)) + 1
    x = np.linspace(0, total_depth, N)

    # Define layer boundaries (in m)
    layer1_end = 0.05  # Layer 1: Permeable surface
    layer2_end = layer1_end + 0.02  # Layer 2: Leveling layer
    layer3_end = layer2_end + 0.12  # Layer 3: Base layer
    layer4_end = total_depth  # Layer 4: Soil bedding

    # Initialize arrays for density (rho), specific heat (c), and conductivity (lam)
    rho = np.zeros(N)
    c = np.zeros(N)
    lam = np.zeros(N)

    for i, depth in enumerate(x):
        if depth <= layer1_end:
            # --- Layer 1: Permeable surface (porous) ---
            # Solid phase (pervious concrete) properties:
            rho_s = 2000.0  # kg/m³
            c_s = 880.0  # J/(kg·K)
            lam_s = 0.68  # W/(m·K)
            # Fluid phase (assumed water in the voids) properties:
            rho_f = 1000.0  # kg/m³
            c_f = 4186.0  # J/(kg·K)
            lam_f = 0.6  # W/(m·K)
            # Compute effective (apparent) properties
            volumetric_heat_capacity = (1 - phi) * (rho_s * c_s) + phi * (rho_f * c_f)
            rho_effective = (1 - phi) * rho_s + phi * rho_f
            c_effective = volumetric_heat_capacity / rho_effective
            lam_effective = (1 - phi) * lam_s + phi * lam_f
            # Apply calibrated multiplier to effective conductivity
            lam_effective *= f_lam_layer1
            rho[i] = rho_effective
            c[i] = c_effective
            lam[i] = lam_effective
        elif depth <= layer2_end:
            # --- Layer 2: Leveling layer (cement mortar) ---
            rho[i] = 2100.0  # kg/m³
            c[i] = 800.0  # J/(kg·K)
            lam[i] = 0.9  # W/(m·K)
        elif depth <= layer3_end:
            # --- Layer 3: Base layer (gravel) ---
            rho[i] = 1400.0  # kg/m³
            c[i] = 900.0  # J/(kg·K)
            lam[i] = 0.55  # W/(m·K)
        else:
            # --- Layer 4: Soil bedding (compacted soil) ---
            rho[i] = 1700.0  # kg/m³
            c[i] = 840.0  # J/(kg·K)
            lam[i] = 1.78  # W/(m·K)

    # Time integration parameters and constants
    dt = 600  # s (5-minute time step)
    n_steps = len(sim_df)
    sigma_const = 5.67e-8  # Stefan-Boltzmann constant (W/(m²·K⁴))
    L_latent = 2.45e6  # Latent heat of vaporization (J/kg)

    # Initialize temperature field (°C)
    T = np.ones(N) * 20.0
    surface_temp = np.zeros(n_steps)

    # Time integration loop
    for n in range(n_steps):
        I = sim_df['Solar_Radiation'].iloc[n]  # W/m²
        T_air = sim_df['Air_Temperature'].iloc[n]  # °C
        v = sim_df['Wind_Speed'].iloc[n]  # m/s

        # Calculate convective heat transfer coefficient using fixed values
        h_c = hc0 + hc_slope * v  # W/(m²·K)

        T_surface = T[0]
        T_surface_K = T_surface + 273.15
        T_air_K = T_air + 273.15
        T_sky_K = T_air_K - 20.0  # Simple estimate for sky temperature

        if T_surface_K <= 0:
            T_surface_K = 1.0

        try:
            L_rad = emissivity * sigma_const * (T_surface_K ** 4 - T_sky_K ** 4)
        except OverflowError:
            L_rad = 0.0

        H_conv = h_c * (T_surface - T_air)
        E_evap = ER * L_latent / 3.6
        Q_solar = I * (1 - reflectivity)
        Q_net = Q_solar - L_rad - H_conv - E_evap

        T_new = T.copy()
        # Top Boundary (surface node)
        T_new[0] = T[0] + (dt / (rho[0] * c[0])) * (
                lam[0] * (2 * (T[1] - T[0]) + 2 * dx * (Q_net / lam[0])) / (dx ** 2)
        )
        # Interior nodes
        for i in range(1, N - 1):
            lam_ip = 0.5 * (lam[i] + lam[i + 1])
            lam_im = 0.5 * (lam[i] + lam[i - 1])
            T_new[i] = T[i] + (dt / (rho[i] * c[i])) * (
                    (lam_ip * (T[i + 1] - T[i]) - lam_im * (T[i] - T[i - 1])) / (dx ** 2)
            )
        # Bottom Boundary (Neumann condition)
        T_new[N - 1] = T[N - 1] + (dt / (rho[N - 1] * c[N - 1])) * (
                2 * lam[N - 1] * (T[N - 2] - T[N - 1]) / (dx ** 2)
        )
        T = T_new.copy()
        surface_temp[n] = T[0]

    # No time shift is applied since shift = 0
    return surface_temp


def evaluate(params):
    sim_temp = simulate_pavement_temperature(calib_df, params)
    obs_temp = calib_df['Pavement_Temperature'].values

    if np.any(np.isnan(sim_temp)) or np.any(np.isinf(sim_temp)):
        return (-1e6,)

    # Compute Nash–Sutcliffe Efficiency (NSE)
    denom = np.sum((obs_temp - np.mean(obs_temp)) ** 2)
    if denom == 0:
        nse = -1e6
    else:
        nse = 1 - np.sum((obs_temp - sim_temp) ** 2) / denom
    if np.isnan(nse):
        return (-1e6,)
    return (nse,)

save_path = r"C:\Users\Artur\PycharmProjects\PPPaper\outputs"
parameters = ["reflectivity, emissivity, ER, f_lam_layer1"]
individual = [0.23476433456957307, 0.31168493946667913, 1.0000000000000002e-5, 0.1000000000000005]
filename = r'temperature_data_PA.csv'
pavement = filename.split(".")[0].split("_")[-1]
df = pd.read_csv(filename)

calib_size = int(0.4 * len(df))
calib_df = df.iloc[:calib_size].reset_index(drop=True)
val_df = df.iloc[calib_size:].reset_index(drop=True)

sim_cal = simulate_pavement_temperature(calib_df, individual)[25:]
sim_val = simulate_pavement_temperature(val_df, individual)[25:]
obs_cal = calib_df['Pavement_Temperature'].values[25:]
obs_val = val_df['Pavement_Temperature'].values[25:]

denom_cal = np.sum((obs_cal - np.mean(obs_cal)) ** 2)
denom_val = np.sum((obs_val - np.mean(obs_val)) ** 2)

if (denom_cal or denom_val) == 0:
    nse_cal = -1e6
    nse_val = -1e6

else:
    nse_cal = 1 - np.sum((obs_cal - sim_cal) ** 2) / denom_cal
    nse_val = 1 - np.sum((obs_val - sim_val) ** 2) / denom_val


print("Calibration NSE:", nse_cal)
print("\nValidation NSE:", nse_val)


# Calibration period plot
plt.figure(figsize=(10, 4))
time_calib = calib_df.index
plt.plot(time_calib[25:], calib_df['Pavement_Temperature'][25:], label="Observed (Calibration)")
sim_calib = simulate_pavement_temperature(calib_df, individual)
plt.plot(time_calib[25:], sim_calib[25:], label="Modeled (Calibration)")
plt.xlabel("Time Step")
plt.ylabel("Pavement Temperature (°C)")
plt.title(f"Calibration Period: {pavement}")
plt.legend()
plt.tight_layout()
plt.savefig(rf"{save_path}\{pavement}_calibration.png", dpi=300)
plt.show()

# Validation period plot
plt.figure(figsize=(10, 4))
time_val = val_df.index
plt.plot(val_df['Date'][25:], val_df['Pavement_Temperature'][25:], label="Observed (Validation)")
plt.plot(val_df['Date'][25:], sim_val, label="Modeled (Validation)")
plt.xlabel("Time Step")
plt.ylabel("Pavement Temperature (°C)")
plt.title(f"Validation Period: {pavement}")
plt.legend()
plt.tight_layout()
plt.savefig(rf"{save_path}\{pavement}_validation.png", dpi=300)
plt.show()



##########################################################################################################
##########################################################################################################
                                    # SENSITIVITY ANALYSIS
##########################################################################################################
##########################################################################################################

# Baseline calibrated individual:
baseline = [0.2, 0.7, 1e-20, 1.2]  # [reflectivity, emissivity, ER, f_lam_layer1]
param_names = ["reflectivity", "emissivity", "ER", "f_lam_layer1"]

# Define ranges for each parameter (you can adjust the range and resolution)
param_ranges = [
    np.linspace(0.1, 1.0, 20),   # reflectivity
    np.linspace(0.1, 1.0, 20),   # emissivity
    np.linspace(1e-10, 1e-2, 20),  # ER
    np.linspace(0.1, 4.0, 20)    # f_lam_layer1
]

# Dictionary to store sensitivity results
sensitivity_results = {}

for i in range(len(baseline)):
    values = param_ranges[i]
    nse_list = []
    for val in values:
        # Create a new parameter set with the i-th parameter changed
        new_params = baseline.copy()
        new_params[i] = val

        # Run simulation on the calibration set and compute NSE
        sim_cal = simulate_pavement_temperature(calib_df, new_params)[25:]
        obs_cal = calib_df['Pavement_Temperature'].values[25:]
        denom_cal = np.sum((obs_cal - np.mean(obs_cal)) ** 2)
        if denom_cal == 0:
            nse = -1e6
        else:
            nse = 1 - np.sum((obs_cal - sim_cal) ** 2) / denom_cal
        nse_list.append(nse)
    sensitivity_results[param_names[i]] = (values, nse_list)

# Plot sensitivity curves for each parameter
for param in param_names:
    x_vals, y_vals = sensitivity_results[param]
    plt.figure()
    plt.plot(x_vals, y_vals, marker='o')
    plt.xlabel(param)
    plt.ylabel("Calibration NSE")
    plt.title(f"Sensitivity of NSE to {param}")
    plt.grid(True)
    plt.savefig(f'{param}_sensitivity.png', dpi=300)
    plt.show()
#
