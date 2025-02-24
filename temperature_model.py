import pandas as pd
import numpy as np


def compute_convection_coefficients(v, T_air, T_surface):
    """
    Compute the forced (C_fc) and natural (C_nc) convection coefficients for a single timestep
    based on input meteorological conditions.

    Args:
        v (float): Wind speed at 10 m (m/s)
        T_air (float): Ambient air temperature (°C)
        T_surface (float): Surface temperature (°C)

    Returns:
        tuple: (C_fc, C_nc) - Forced and natural convection coefficients.
    """

    # Constants
    rho_air = 1.225  # kg/m³ (air density)
    cp_air = 1005  # J/(kg·K) (specific heat of air at constant pressure)
    Lv = 2.5e6  # J/kg (latent heat of vaporization of water)
    g = 9.81  # m/s² (gravitational acceleration)
    beta = 1 / 300  # K⁻¹ (thermal expansion coefficient for air)
    nu_air = 1.5e-5  # m²/s (kinematic viscosity of air)
    Pr = 0.71  # Prandtl number for air
    k_air = 0.025  # W/(m·K) (thermal conductivity of air)
    L = 1.0  # m (characteristic length for pavement surface)

    # Ensure wind speed is not zero to avoid division errors
    v_eff = max(v, 0.1)  # Default to 0.1 m/s if v is zero or negative

    # Convert temperatures to Kelvin
    T_air_K = T_air + 273.15
    T_surface_K = T_surface + 273.15

    # Compute forced convection coefficient (C_fc)
    h_forced = 3
    C_fc = h_forced / (rho_air * cp_air * v_eff)

    # Compute virtual temperature difference
    delta_theta_v = T_surface_K - T_air_K

    # Compute Grashof number (Gr) for natural convection
    Gr = (g * beta * delta_theta_v * L ** 3) / (nu_air ** 2)
    Ra = Gr * Pr  # Rayleigh number

    # Compute Nusselt number for natural convection using empirical correlations
    Nu_nat = 0
    if 1e4 <= Ra < 1e7:
        Nu_nat = 0.54 * Ra ** (1 / 4)
    elif 1e7 <= Ra < 1e11:
        Nu_nat = 0.15 * Ra ** (1 / 3)

    # Compute heat transfer coefficient for natural convection (W/m²K)
    h_natural = Nu_nat * k_air / L

    # Convert to empirical coefficient C_nc
    C_nc = h_natural / (rho_air * Lv * (abs(delta_theta_v) ** (1 / 3)))

    return C_fc, C_nc


def calculate_h_s(reflectivity, I):
    """
    Calculate net solar radiation.
    h_s = (1 - α_s) * R_s
    NEW: α_s is the calibrated reflectivity.
    """
    return (1 - reflectivity) * I


def calculate_h_li(emissivity, sigma_const, CR, e_a, T_air_K):
    """
    Calculate incoming longwave radiation.
    h_li = ε * σ * (CR + 0.67*(1-CR) * e_a^(0.08)) * T_air_K^4
    NEW: CR is the cloud cover ratio, e_a is ambient vapor pressure.
    """
    return emissivity * sigma_const * (CR + 0.67 * (1 - CR) * (e_a ** 0.08)) * (T_air_K ** 4)


def calculate_h_l0(emissivity, sigma_const, T_surface_K):
    """
    Calculate outgoing longwave radiation.
    h_l0 = ε * σ * T_surface_K^4
    NEW: T_surface_K is the surface temperature in Kelvin.
    """
    return emissivity * sigma_const * (T_surface_K ** 4)


def calculate_h_rad(h_s, h_li, h_l0):
    """
    Combine radiation terms to obtain total net radiation.
    h_rad = h_s + h_li - h_l0
    """
    return h_s + h_li - h_l0


# ============================
# Helper Functions for Convection and Evaporation
# ============================

def calculate_u_s(CSh, v):
    """
    Calculate effective surface wind speed.
    u_s = CSh * v
    NEW: CSh is the sheltering coefficient.
    """
    return CSh * v


def calculate_h_evap(rho_a, L_latent, C_fc, C_nc, u_s, delta_theta_v, q_sat, q_a):
    """
    Calculate the evaporative heat flux.
    h_evap = ρ_a * L_v * (C_fc * u_s + C_nc * |Δθ_v|^0.33) * (q_sat - q_a)

    When the surface temperature is lower than the air temperature, q_sat can be less than q_a,
    which would yield a negative h_evap. Since evaporation represents energy loss by vaporization,
    a negative value (indicating net condensation) is not feasible in this formulation.
    Therefore, the result is capped to a minimum of zero.
    """
    h_evap = rho_a * L_latent * (C_fc * u_s + C_nc * (abs(delta_theta_v) ** 0.33)) * (q_sat - q_a)
    # Cap evaporation heat flux to zero if negative.
    return max(h_evap, 0.0)


def calculate_h_conv(rho_a, c_p, C_fc, C_nc, u_s, delta_theta_v, T_surface, T_air):
    """
    Calculate the convective (sensible) heat flux.
    h_conv = ρ_a * c_p * (C_fc * u_s + C_nc * |Δθ_v|^0.33) * (T_surface - T_air)
    """
    return rho_a * c_p * (C_fc * u_s + C_nc * (abs(delta_theta_v) ** 0.33)) * (T_surface - T_air)


# ============================
# Additional Helper Functions for Moisture Calculations
# ============================

def calculate_saturation_vapor_pressure(T):
    """
    Calculate saturation vapor pressure using the Tetens formula.
    T is in Celsius.
    e_sat = 0.611 * exp((17.27 * T) / (T + 237.3)) * 1000  [Pa]
    NEW: Converts the result to Pascals.
    """
    return 0.611 * np.exp((17.27 * T) / (T + 237.3)) * 1000


def calculate_specific_humidity(e, P=101325):
    """
    Calculate specific humidity.
    q = 0.622 * e / (P - e)
    NEW: P is the atmospheric pressure (default 101325 Pa).
    """
    return 0.622 * e / (P - e + 1e-6)


# ============================
# Main Simulation Function
# ============================

def model_pavement_temperature(sim_df, params):
    """
    Run the pavement temperature simulation using the new energy balance:
       h_net = h_rad - h_evap - h_conv - h_r0
    with:
       h_rad = h_s + h_li - h_l0
    Each term is computed via dedicated functions.

    Calibrated parameters (in order):
      0: reflectivity         (α_s, e.g., 0.1 to 1.0) [calibrated]
      1: emissivity           (ε, e.g., 0.1 to 1.0)   [calibrated]
      2: f_lam_layer1         (multiplier for Layer 1 conductivity, e.g., 0.1 to 4.0) [calibrated]

    The function now returns a dataframe containing:
      - h_s, h_li, h_l0: The radiation sub-terms
      - h_rad: Net radiation (h_s + h_li - h_l0)
      - h_evap: Evaporative heat flux
      - h_conv: Convective heat flux
      - h_r0: Runoff heat flux (here set to zero)
      - h_net: Total net heat flux (h_rad - h_evap - h_conv - h_r0)
      - surface_temp: The simulated surface temperature at each time step

    Other fixed parameters and spatial discretization remain as in the original function.
    """
    reflectivity = params[0]
    emissivity = params[1]
    f_lam_layer1 = params[2]

    # Fixed parameters for conduction and layer properties
    phi = 0.4  # Porosity for Layer 1
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
            # Solid phase properties
            rho_s = 2000.0  # kg/m³
            c_s = 880.0  # J/(kg·K)
            lam_s = 0.68  # W/(m·K)
            # Fluid phase properties (water)
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
            rho[i] = 2100.0
            c[i] = 800.0
            lam[i] = 0.9
        elif depth <= layer3_end:
            # --- Layer 3: Base layer (gravel) ---
            rho[i] = 1400.0
            c[i] = 900.0
            lam[i] = 0.55
        else:
            # --- Layer 4: Soil bedding (compacted soil) ---
            rho[i] = 1700.0
            c[i] = 840.0
            lam[i] = 1.78

    # Time integration parameters and constants
    dt = 600  # s (5-minute time step)
    n_steps = len(sim_df)
    sigma_const = 5.67e-8  # Stefan-Boltzmann constant (W/(m²·K⁴))
    L_latent = 2.45e6  # Latent heat of vaporization (J/kg)

    # Initialize temperature field (°C)
    T = np.ones(N) * 40.0

    # Initialize a dictionary to store all terms per time step
    results = {
        'h_s': [],
        'h_li': [],
        'h_l0': [],
        'h_rad': [],
        'h_evap': [],
        'h_conv': [],
        'h_r0': [],
        'h_net': [],
        'surface_temp': []
    }

    # ============================
    # Time Integration Loop
    # ============================
    for n in range(n_steps):
        # Retrieve meteorological data from sim_df
        I = sim_df['SolarRadiation'].iloc[n]  # W/m², incident solar radiation (GHI)
        T_air = sim_df['AirTemperature'].iloc[n]  # °C, ambient air temperature
        v = sim_df['WindSpeed'].iloc[n]  # m/s, wind speed at 10 m (u_10)
        RH = sim_df['RelativeHumidity'].iloc[n]  # relative humidity at 2m
        CR = sim_df['CloudCoverage'].iloc[n]  # total cloud coverage ratio

        # Convert temperatures to Kelvin
        T_surface = T[0]
        T_surface_K = T_surface + 273.15
        T_air_K = T_air + 273.15

        # --- Calculate Radiation Terms ---
        h_s = calculate_h_s(reflectivity, I)

        e_sat = calculate_saturation_vapor_pressure(T_air)  # Pa, saturation vapor pressure at T_air
        e_a = RH * e_sat  # Pa, ambient vapor pressure
        h_li = calculate_h_li(emissivity, sigma_const, CR, e_a, T_air_K)
        h_l0 = calculate_h_l0(emissivity, sigma_const, T_surface_K)
        h_rad = calculate_h_rad(h_s, h_li, h_l0)

        # --- Calculate Convection and Evaporation Terms ---
        CSh = 0.7  # Default sheltering coefficient
        u_s = calculate_u_s(CSh, v)
        # Convection coefficients (forced and natural)
        C_fc, C_nc = compute_convection_coefficients(v, T_air, T[0])
        # Virtual temperature difference (in Kelvin)
        delta_theta_v = T_surface_K - T_air_K

        # Compute specific humidity values using Tetens formula and specific humidity relation
        P = 101325  # Pa, atmospheric pressure
        e_sat_surface = calculate_saturation_vapor_pressure(T_surface)  # Pa at surface temperature
        q_sat = calculate_specific_humidity(e_sat_surface, P)  # Saturated specific humidity
        q_a = calculate_specific_humidity(e_a, P)  # Ambient specific humidity

        # Air properties for convection/evaporation calculations
        rho_a = 1.2  # kg/m³, air density
        c_p = 1005  # J/(kg·K), specific heat of air

        h_evap = calculate_h_evap(rho_a, L_latent, C_fc, C_nc, u_s, delta_theta_v, q_sat, q_a)
        h_conv = calculate_h_conv(rho_a, c_p, C_fc, C_nc, u_s, delta_theta_v, T_surface, T_air)
        h_r0 = 0  # Heat flux due to surface runoff (set to zero by default)
        # Total net heat flux at the surface
        h_net = h_rad - h_evap - h_conv - h_r0

        # ============================
        # Temperature Conduction Update
        # ============================
        T_new = T.copy()
        # Top Boundary (surface node) using h_net as the boundary flux term
        T_new[0] = T[0] + (dt / (rho[0] * c[0])) * (
            lam[0] * (2 * (T[1] - T[0]) + 2 * dx * (h_net / lam[0])) / (dx ** 2)
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
        surface_temp = T[0]

        # Record each term and the surface temperature for this timestep
        results['h_s'].append(h_s)
        results['h_li'].append(h_li)
        results['h_l0'].append(h_l0)
        results['h_rad'].append(h_rad)
        results['h_evap'].append(h_evap)
        results['h_conv'].append(h_conv)
        results['h_r0'].append(h_r0)
        results['h_net'].append(h_net)
        results['surface_temp'].append(surface_temp)

    return pd.DataFrame(results)


def old_model_pavement_temperature(sim_df, params):
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
    f_lam_layer1 = params[2]

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
        I = sim_df['SolarRadiation'].iloc[n]  # W/m²
        T_air = sim_df['AirTemperature'].iloc[n]  # °C
        v = sim_df['WindSpeed'].iloc[n]  # m/s

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


def NSE(obs_df, modeled_df):
    obs = obs_df['PavementTemperature']
    modeled = modeled_df['surface_temp']
    denom = np.sum((obs - np.mean(obs)) ** 2)
    return 1 - np.sum((obs - modeled) ** 2) / denom
