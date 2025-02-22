import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# PART 1: GENERATE SYNTHETIC METEOROLOGICAL DATA
# =============================================================================

# For reproducibility
np.random.seed(0)

# Create a datetime index for one week at 5-minute intervals (using '5min' instead of '5T')
start_date = '2025-01-01'
periods = 7 * 24 * 12  # 7 days * 24 hours * 12 intervals/hour = 2016 points
date_range = pd.date_range(start=start_date, periods=periods, freq='5min')

# Initialize DataFrame for forcing data
df = pd.DataFrame(index=date_range)

# Compute hour-of-day in decimal hours
df['hour_of_day'] = df.index.hour + df.index.minute / 60.0

# 1. Solar Radiation: Zero at night; Gaussian profile (6:00–18:00)
max_radiation = 800.0  # W/m²
sigma_val = 3.0      # hours (width of bell curve)
df['Solar_Radiation'] = 0.0
day_mask = (df['hour_of_day'] >= 6) & (df['hour_of_day'] < 18)
df.loc[day_mask, 'Solar_Radiation'] = max_radiation * np.exp(
    -((df.loc[day_mask, 'hour_of_day'] - 12) ** 2) / (2 * sigma_val ** 2)
)

# 2. Air Temperature: Sinusoidal (roughly 15–30°C) with noise
mean_temp = 22.5        # °C
amplitude_temp = 7.5    # °C
temp_noise = np.random.normal(0, 1, size=len(df))
df['Air_Temperature'] = mean_temp + amplitude_temp * np.sin(2 * np.pi * (df['hour_of_day'] - 9) / 24) + temp_noise

# 3. Humidity: Sinusoidal (20–80%) with added noise
mean_humidity = 50      # %
amplitude_humidity = 20 # %
humidity_noise = np.random.normal(0, 7, size=len(df))
humidity_raw = mean_humidity + amplitude_humidity * np.sin(2 * np.pi * (df['hour_of_day'] - 9) / 24) + humidity_noise
df['Humidity'] = np.clip(humidity_raw, 20, 80)

# 4. Wind Speed: Sinusoidal (0–4 m/s) with significant noise
mean_wind = 2         # m/s
amplitude_wind = 1    # m/s
wind_noise = np.random.normal(0, 1.5, size=len(df))
wind_raw = mean_wind + amplitude_wind * np.sin(2 * np.pi * (df['hour_of_day'] - 9) / 24) + wind_noise
df['Wind_Speed'] = np.clip(wind_raw, 0, 4)

# Remove the helper column
df.drop(columns='hour_of_day', inplace=True)

df[['Solar_Radiation', 'Air_Temperature', 'Humidity', 'Wind_Speed']].plot(
    subplots=True, figsize=(10, 8), title="Synthetic Weather Data"
)
plt.tight_layout()
plt.show()


# =============================================================================
# PART 2: SET UP THE 1D TRANSIENT HEAT CONDUCTION MODEL FOR THE PAVEMENT
# =============================================================================
# The pavement is modeled as a 3D section (5m x 5m x 2m) with a 1D vertical discretization.
# It is divided into four layers:
#   1. Permeable surface layer (6 cm, pervious concrete; porous)
#   2. Leveling layer (15 cm, cement mortar)
#   3. Base layer (40 cm, gravel)
#   4. Soil bedding (80 cm, compacted soil)
# Total depth = 0.06 + 0.15 + 0.40 + 0.80 = 1.41 m.

total_depth = 1.41  # m
dx = 0.05           # m, spatial resolution
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

# =============================================================================
# MODEL CONSTANTS FOR THE SURFACE ENERGY BALANCE
# =============================================================================

reflectivity = 0.3      # Reflection coefficient (fraction of solar radiation reflected)
emissivity   = 0.95     # Emissivity of the surface
sigma_const  = 5.67e-8  # Stefan-Boltzmann constant (W/(m²·K⁴))
L_latent     = 2.45e6   # Latent heat of vaporization (J/kg)

# Evaporation rate (surface evaporation rate in kg/(m²·h))
# (A reduced value is chosen to keep fluxes within a reasonable range.)
ER = 0.0005

# =============================================================================
# TIME INTEGRATION PARAMETERS
# =============================================================================

dt = 300       # s (5-minute time step)
n_steps = len(df)  # 2016 time steps (one week)

# Array to store the pavement surface temperature over time
surface_temp = np.zeros(n_steps)

# =============================================================================
# TIME INTEGRATION LOOP
# =============================================================================

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

    # Create a new temperature array for the updated values
    T_new = T.copy()

    # --- Top Boundary (Surface Node, i = 0) ---
    # Apply a Neumann boundary condition with a ghost node:
    # Define T_ghost such that:
    #   -lam[0]*(T[1] - T_ghost)/(2*dx) = Q_net
    # => T_ghost = T[1] + 2*dx*(Q_net/lam[0])
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

    # --- Bottom Boundary (Insulated, i = N-1) ---
    # For an insulated boundary, assume zero flux by mirroring the second-last node:
    T_new[N - 1] = T[N - 1] + (dt / (rho[N - 1] * c[N - 1])) * (
                        2 * lam[N - 1] * (T[N - 2] - T[N - 1]) / (dx**2)
                    )

    # Update the temperature profile for the next time step:
    T = T_new.copy()

    # Save the surface temperature for later plotting:
    surface_temp[n] = T[0]

# =============================================================================
# PLOTTING THE RESULTS
# =============================================================================

# Create a time index corresponding to the synthetic forcing data
time_index = df.index

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
