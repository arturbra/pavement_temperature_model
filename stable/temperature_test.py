import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# def compute_valid_temperature(row):
#     temp1 = row['box_dc_temperature_1']
#     temp2 = row['box_dc_temperature_2']
#
#     if temp1 < 0 and temp2 < -0:
#         return np.nan  # Both are invalid
#     elif temp1 < -0:
#         return temp2  # Only temp1 is invalid
#     elif temp2 < -0:
#         return temp1  # Only temp2 is invalid
#     else:
#         return (temp1 + temp2) / 2  # Average if both are valid
#
#
# sdf_path = r"C:\Users\artur\OneDrive\Doutorado\UTSA\PP\Final Report\data\level\pp_data_level_2.csv"
# sdf = pd.read_csv(sdf_path, parse_dates=[0], low_memory=False)
#
# df = sdf[['date', 'box_dc_temperature_1', 'box_dc_temperature_2']]
# df['date'] = pd.to_datetime(df['date'])
#
# pavement_temperature = df.resample('1h', on='date').mean()
# pavement_temperature['box_dc_temperature'] = pavement_temperature.apply(compute_valid_temperature, axis=1)
# pavement_temperature.drop(columns=['box_dc_temperature_1', 'box_dc_temperature_2'], inplace=True)
# pavement_temperature = pavement_temperature.dropna().reset_index()
#
# input_data = pd.read_csv(r"C:\Users\Artur\OneDrive\Doutorado\UTSA\PP\PPPaper_3\data\2644067_29.64_-98.49_2023.csv", skiprows=2)
# input_data['date'] = pd.to_datetime(input_data[['Year', 'Month', 'Day', 'Hour']])
# input_data.drop(columns=['Year', 'Month', 'Day', 'Hour', 'Minute'], inplace=True)
# merged_df = input_data.merge(pavement_temperature, on='date', how='inner')
# merged_df.to_csv('input_data.csv', index=False)


df = pd.read_csv(r'input_data.csv')
df.columns = ['Solar_Radiation', 'Humidity', 'Wind_Speed', 'Air_Temperature', 'Date', 'Pavement_Temperature']
df['Pavement_Temperature'] = (df['Pavement_Temperature'] - 32) / 1.8

plt.scatter(df['Air_Temperature'], df['Pavement_Temperature'])

total_depth = 1.41  # m
dx = 0.03           # m, spatial resolution
N = int(np.ceil(total_depth / dx)) + 1  # number of nodes
x = np.linspace(0, total_depth, N)

# Define layer boundaries (in m)
layer1_end = 0.06             # Layer 1: 6 cm
layer2_end = layer1_end + 0.15  # Layer 2: additional 15 cm
layer3_end = layer2_end + 0.40  # Layer 3: additional 40 cm
layer4_end = total_depth       # Layer 4: remaining 80 cm

# Initialize arrays for effective density (rho), specific heat (c), and thermal conductivity (lam)
rho = np.zeros(N)
c   = np.zeros(N)
lam = np.zeros(N)

for i, depth in enumerate(x):
    if depth <= layer1_end:
        # --- Layer 1: Permeable surface (porous) ---
        # Incorporate porosity in the effective (apparent) thermal properties.
        # Assume:
        phi = 0.4  # Porosity (fraction)
        # Solid phase (pervious concrete) properties:
        rho_s = 2000.0    # kg/m³
        c_s = 880.0       # J/(kg·K)
        lam_s = 0.68       # W/(m·K)
        # Fluid phase (assumed water in the voids) properties:
        rho_f = 1000.0    # kg/m³
        c_f = 4186.0      # J/(kg·K)
        lam_f = 0.6       # W/(m·K)
        # Compute effective (apparent) volumetric heat capacity:
        volumetric_heat_capacity = (1 - phi) * (rho_s * c_s) + phi * (rho_f * c_f)
        # Compute effective density:
        rho_effective = (1 - phi) * rho_s + phi * rho_f
        # Then effective specific heat:
        c_effective = volumetric_heat_capacity / rho_effective
        # Compute effective thermal conductivity:
        lam_effective = (1 - phi) * lam_s + phi * lam_f
        # Assign effective properties:
        rho[i] = rho_effective
        c[i]   = c_effective
        lam[i] = lam_effective
    elif depth <= layer2_end:
        # --- Layer 2: Leveling layer (cement mortar) ---
        rho[i] = 2100.0      # kg/m³
        c[i]   = 800.0       # J/(kg·K)
        lam[i] = 0.9         # W/(m·K)
    elif depth <= layer3_end:
        # --- Layer 3: Base layer (gravel) ---
        rho[i] = 1400.0      # kg/m³
        c[i]   = 900.0       # J/(kg·K)
        lam[i] = 0.55         # W/(m·K)
    else:
        # --- Layer 4: Soil bedding (compacted soil) ---
        rho[i] = 1700.0      # kg/m³
        c[i]   = 840.0       # J/(kg·K)
        lam[i] = 1.78         # W/(m·K)

# Initial temperature everywhere is 20°C
T = np.ones(N) * 20.0

reflectivity = 0.3      # Reflection coefficient (fraction of solar radiation reflected)
emissivity   = 0.95     # Emissivity of the surface
sigma_const  = 5.67e-8  # Stefan-Boltzmann constant (W/(m²·K⁴))
L_latent     = 2.45e6   # Latent heat of vaporization (J/kg)

# Evaporation rate (surface evaporation rate in kg/(m²·h))
# (A reduced value is chosen to keep fluxes within a reasonable range.)
ER = 0.00005

dt = 300       # s (5-minute time step)
n_steps = len(df)  # 2016 time steps (one week)

# Array to store the pavement surface temperature over time
surface_temp = np.zeros(n_steps)

"""
TIME INTEGRATION LOOP
"""

for n in range(n_steps):
    # Retrieve the forcing for the current time step:
    I = df['Solar_Radiation'].iloc[n]       # W/m²
    T_air = df['Air_Temperature'].iloc[n]     # °C
    v = df['Wind_Speed'].iloc[n]              # m/s

    # Estimate convective heat transfer coefficient via Newton’s Law of Cooling:
    h_c = 5.6 + 4.0 * v  # W/(m²·K)

    # Get current surface temperature and convert to Kelvin
    T_surface = T[0]
    T_surface_K = T_surface + 273.15
    T_air_K = T_air + 273.15
    T_sky_K = T_air_K - 20.0  # Simple estimate for sky temperature

    # Safeguard against nonphysical Kelvin values:
    if T_surface_K <= 0:
        T_surface_K = 1.0

    # Calculate long-wave radiation term (W/m²)
    try:
        L_rad = emissivity * sigma_const * (T_surface_K**4 - T_sky_K**4)
    except OverflowError:
        L_rad = 0.0

    # Convective heat exchange (W/m²)
    H_conv = h_c * (T_surface - T_air)

    # Evaporative cooling term (W/m²)
    E_evap = ER * L_latent / 3.6

    # Absorbed solar radiation (W/m²)
    Q_solar = I * (1 - reflectivity)

    # Net surface heat flux (W/m²): positive flux heats the pavement
    Q_net = Q_solar - L_rad - H_conv - E_evap

    T_new = T.copy()

    # --- Top Boundary (Surface Node, i = 0) ---
    T_new[0] = T[0] + (dt / (rho[0] * c[0])) * (
                    lam[0] * (2 * (T[1] - T[0]) + 2 * dx * (Q_net / lam[0])) / (dx**2)
                )

    # --- Interior Nodes (i = 1 to N-2) ---
    for i in range(1, N - 1):
        # Use arithmetic averages for interface conductivity:
        lam_ip = 0.5 * (lam[i] + lam[i + 1])
        lam_im = 0.5 * (lam[i] + lam[i - 1])
        T_new[i] = T[i] + (dt / (rho[i] * c[i])) * (
            (lam_ip * (T[i + 1] - T[i]) - lam_im * (T[i] - T[i - 1])) / (dx**2)
        )

    T_new[N - 1] = T[N - 1] + (dt / (rho[N - 1] * c[N - 1])) * (
                        2 * lam[N - 1] * (T[N - 2] - T[N - 1]) / (dx**2)
                    )

    T = T_new.copy()

    surface_temp[n] = T[0]


"""
PLOTTING THE RESULTS
"""

# Create a time index corresponding to the synthetic forcing data
time_index = df.index

subset_pavement = df.iloc[72:144, 5]
subset_time = time_index[72:144]
surface_temp2 = np.array(surface_temp)
surface_shifted = np.roll(surface_temp2, 6)
subset_modeled = np.array(surface_shifted[72: 144])

# Plot modeled surface temperature vs. observed surface temperature over time
plt.plot(time_index, df['Pavement_Temperature'], label="Pavement Temperature (Observed) (°C)")
plt.plot(time_index, surface_temp, label="Pavement Temperature (Modeled) (°C)")
plt.xlabel('Time')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.tight_layout()
plt.show()

# Plot a subset of the modeled surface temperature vs. observed surface temperature over time
plt.plot(subset_time, subset_pavement, label="Pavement Temperature (Observed) (°C)")
plt.plot(subset_time, subset_modeled, label="Pavement Temperature (Modeled) (°C)")
plt.xlabel('Time')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.tight_layout()
plt.show()

# Plot pavement surface temperature vs. ambient air temperature over time
plt.figure(figsize=(12, 6))
plt.plot(time_index, surface_temp, label='Pavement Surface Temperature (°C)')
plt.plot(time_index, df['Air_Temperature'], label='Ambient Air Temperature (°C)', linestyle='--')
plt.xlabel('Time')
plt.ylabel('Temperature (°C)')
plt.title('Pavement Surface Temperature vs Ambient Air Temperature')
plt.legend()
plt.tight_layout()
plt.show()

# Plot the final temperature profile in the pavement
plt.figure(figsize=(6, 6))
plt.plot(T, x, marker='o')
plt.gca().invert_yaxis()
plt.xlabel('Temperature (°C)')
plt.ylabel('Depth (m)')
plt.title('Final Temperature Profile in the Pavement')
plt.tight_layout()
plt.show()


