import configparser
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import spsolve

def load_parameters(filename="input_data/parameters.ini"):
    """ Load all parameters from the ini file and override values with calibrated parameters if they exist. """
    config = configparser.ConfigParser()
    config.read(filename)

    def parse_section(section):
        return {k: float(v) for k, v in config[section].items()}

    # Load all parameters normally
    params = {section: parse_section(section) for section in config.sections()}

    # Override with calibrated parameters
    if "calibration" in params:
        for key, value in params["calibration"].items():
            # Find if the key originally exists in another section and replace it
            for section in params:
                if key in params[section] and section != "calibration":
                    params[section][key] = value  # Override original value

    return params


def compute_convection_coefficients(h_forced, rho_air, cp_air, g, beta, L, nu_air, Pr, k_air, Lv, v, T_air, T_surface):
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

    # Ensure wind speed is not zero to avoid division errors
    v_eff = max(v, 0.1)  # Default to 0.1 m/s if v is zero or negative

    # Convert temperatures to Kelvin
    T_air_K = T_air + 273.15
    T_surface_K = T_surface + 273.15

    # Compute forced convection coefficient (C_fc)
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


def calculate_h_evap(rainfall, rho_a, L_latent, C_fc, C_nc, u_s, delta_theta_v, q_sat, q_a):
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
    e_sat = 6.11 * 10 ** ((7.5 * T_air) / (T_air + 237.3)) * 100 [Pa]
    e     = 6.11 * 10 ** ((7.5 * T_dew) / (T_dew + 237.3)) * 100 [Pa]
    NEW: Converts the result to Pascals.
    """
    return 6.112 * np.exp((17.67 * T) / (T + 243.5)) * 100  # converting from mba to Pa


def calculate_specific_humidity(e, P=101325):
    """
    Calculate specific humidity.
    q = 0.622 * e / (P - e)
    NEW: P is the atmospheric pressure (default 101325 Pa).
    """
    return 0.622 * e / (P - (0.378 * e))


# ============================
# Calculation of the Heat Exchange Due to the Rainfall Water
# ============================

def calculate_h_r0(dt, P, T_s0, T_dp, rho_cp_p, water_rho, water_cp, lam_surface):
    """
    Calculate the conductive heat flux from the pavement to the runoff water (h_ro)
    and the resulting water temperature.

    Parameters:
        dt         : Time step duration (s)
        P          : Rainfall depth during the time step (m)
        T_s0       : Pavement surface temperature at the start of the rain event (K)
        T_dp       : Dew point temperature (K)
        rho_cp_p   : Product of density and specific heat for pavement (kg/m³ * J/(kg·K))
        water_rho  : Density of water (kg/m³)
        water_cp   : Specific heat capacity of water (J/(kg·K))
        lam_surface: Surface conductivity of pavement (W/m·K)

    Returns:
        h_ro   : Conductive heat flux from pavement to runoff water (W/m²)
        T_water: Predicted water temperature after heat exchange (K)
    """
    # Compute thermal diffusivity D of the pavement (m²/s)
    D = lam_surface / rho_cp_p
    delta = np.sqrt(4 * D * dt)

    # Precipitation rate (m/s)
    i = P / dt

    # Compute (ρ c_p)_w for water
    water_rho_cp = water_rho * water_cp

    # Compute beta factor
    beta = (delta * rho_cp_p) / (2 * P * water_rho_cp)

    # Compute h_ro using the provided formulation
    h_ro = i * water_rho_cp * (T_s0 - T_dp) * (beta / (1 + beta))

    # Energy balance for the water:
    # (P * water_rho_cp) * (T_water - T_dp) = h_ro * dt  => T_water = T_dp + (h_ro*dt)/(P*water_rho_cp)
    T_water = T_dp + (T_s0 - T_dp) * (beta / (1 + beta))

    return h_ro, T_water

# ============================
# Main Simulation Function
# ============================

def model_pavement_temperature(sim_df, parameters_file):
    """
    Run the pavement temperature simulation using the new energy balance:
       h_net = h_rad - h_evap - h_conv - h_r0
    with:
       h_rad = h_s + h_li - h_l0

    Uses the Crank-Nicolson scheme for unconditional stability with 1-hour time steps.

    Calibrated parameters (in order):
      0: reflectivity         (α_s, e.g., 0.1 to 1.0) [calibrated]
      1: emissivity           (ε, e.g., 0.1 to 1.0)   [calibrated]
      2: f_lam_layer1         (multiplier for Layer 1 conductivity, e.g., 0.1 to 4.0) [calibrated]

    The function returns a dataframe containing:
      - h_s, h_li, h_l0: The radiation sub-terms
      - h_rad: Net radiation (h_s + h_li - h_l0)
      - h_evap: Evaporative heat flux
      - h_conv: Convective heat flux
      - h_r0: Runoff heat flux
      - h_net: Total net heat flux (h_rad - h_evap - h_conv - h_r0)
      - surface_temp: The simulated surface temperature at each time step
    """

    parameters = load_parameters(parameters_file)

    # Convection parameters:
    rho_air = parameters['convection']['rho_air']
    cp_air = parameters['convection']['cp_air']
    Lv = parameters['convection']["lv"]
    g = parameters['convection']["g"]
    beta = parameters['convection']["beta"]
    nu_air = parameters['convection']["nu_air"]
    Pr = parameters['convection']["pr"]
    k_air = parameters['convection']["k_air"]
    L = parameters['convection']["l"]
    h_forced = parameters['convection']["h_forced"]

    # Pavement parameters:
    phi = parameters['pavement']['phi']
    total_depth = parameters['pavement']['total_depth']
    dx = parameters['pavement']['dx']
    layer1_end = parameters['pavement']['layer1_end']
    layer2_thickness = parameters['pavement']['layer2_thickness']
    layer3_thickness = parameters['pavement']['layer3_thickness']
    reflectivity = parameters['pavement']['reflectivity']
    emissivity = parameters['pavement']['emissivity']

    # Surface layer:
    l1_rho_s = parameters['layer1']['rho_s']
    l1_c_s = parameters['layer1']['c_s']
    l1_lam_s = parameters['layer1']['lam_s']
    l1_rho_f = parameters['layer1']['rho_f']
    l1_c_f = parameters['layer1']['c_f']
    l1_lam_f = parameters['layer1']['lam_f']
    f_lam_layer1 = parameters['layer1']['f_lam_layer1']

    # Leveling layer:
    l2_rho = parameters['layer2']['rho']
    l2_c = parameters['layer2']['c']
    l2_lam = parameters['layer2']['lam']

    # Base layer:
    l3_rho = parameters['layer3']['rho']
    l3_c = parameters['layer3']['c']
    l3_lam = parameters['layer3']['lam']

    # Sub-base layer
    l4_rho = parameters['layer4']['rho']
    l4_c = parameters['layer4']['c']
    l4_lam = parameters['layer4']['lam']

    # General
    dt = parameters['general']['dt']
    P = parameters['general']['p']
    CSh = parameters['general']['csh']
    sigma_const = parameters['general']['sigma_const']
    initial_temperature = parameters['general']['initial_temperature']
    L_latent = parameters['general']['l_latent']
    rho_water = parameters['general']['rho_water']
    cp_water = parameters['general']['cp_water']

    # Fixed parameters for conduction and layer properties
    N = int(np.ceil(total_depth / dx)) + 1
    x = np.linspace(0, total_depth, N)

    # Define layer boundaries (in m)
    layer2_end = layer1_end + layer2_thickness
    layer3_end = layer2_end + layer3_thickness  # Layer 3: Base layer
    layer4_end = total_depth  # Layer 4: Soil bedding

    # Initialize arrays for density (rho), specific heat (c), and conductivity (lam)
    rho = np.zeros(N)
    c = np.zeros(N)
    lam = np.zeros(N)

    for i, depth in enumerate(x):
        if depth <= layer1_end:
            # --- Layer 1: Permeable surface (porous) ---
            # Compute effective (apparent) properties
            volumetric_heat_capacity = (1 - phi) * (l1_rho_s * l1_c_s) + phi * (l1_rho_f * l1_c_f)
            rho_effective = (1 - phi) * l1_rho_s + phi * l1_rho_f
            c_effective = volumetric_heat_capacity / rho_effective
            lam_effective = (1 - phi) * l1_lam_s + phi * l1_lam_f
            # Apply calibrated multiplier to effective conductivity
            lam_effective *= f_lam_layer1
            rho[i] = rho_effective
            c[i] = c_effective
            lam[i] = lam_effective
        elif depth <= layer2_end:
            # --- Layer 2: Leveling layer (cement mortar) ---
            rho[i] = l2_rho
            c[i] = l2_c
            lam[i] = l2_lam
        elif depth <= layer3_end:
            # --- Layer 3: Base layer (gravel) ---
            rho[i] = l3_rho
            c[i] = l3_c
            lam[i] = l3_lam
        else:
            # --- Layer 4: Soil bedding (compacted soil) ---
            rho[i] = l4_rho
            c[i] = l4_c
            lam[i] = l4_lam

    # Compute thermal diffusivity for each node
    alpha = np.zeros(N)
    for i in range(N):
        alpha[i] = lam[i] / (rho[i] * c[i])

    n_steps = len(sim_df)

    # Initialize temperature field (°C)
    T = np.ones(N) * initial_temperature

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
        'surface_temp': [],
        'water_temp': []
    }

    rain_event_active = False
    rain_event_start_temp = None

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
        T_dew = sim_df['DewPoint'].iloc[n]  # °C dew point temperature
        rainfall = sim_df['Rainfall'].iloc[n]  # Hourly rainfall in mm
        rainfall = rainfall / 1000
        # rainfall = 0
        # Convert temperatures to Kelvin
        T_surface = T[0]
        T_surface_K = T_surface + 273.15
        T_air_K = T_air + 273.15

        # --- Calculate Radiation Terms ---
        h_s = calculate_h_s(reflectivity, I)
        e_s = calculate_saturation_vapor_pressure(T_air)
        e_a = e_s * RH
        h_li = calculate_h_li(emissivity, sigma_const, CR, e_a / 100, T_air_K)  # e_a is converted from Pa to mbar
        h_l0 = calculate_h_l0(emissivity, sigma_const, T_surface_K)
        h_rad = calculate_h_rad(h_s, h_li, h_l0)

        # --- Calculate Convection and Evaporation Terms ---
        u_s = calculate_u_s(CSh, v)
        # Convection coefficients (forced and natural)
        C_fc, C_nc = compute_convection_coefficients(h_forced, rho_air, cp_air, g, beta, L, nu_air, Pr, k_air, Lv, v,
                                                     T_air, T[0])
        # Virtual temperature difference (in Kelvin)
        delta_theta_v = T_surface_K - T_air_K

        # Compute specific humidity values and convection
        h_conv = calculate_h_conv(rho_air, cp_air, C_fc, C_nc, u_s, delta_theta_v, T_surface, T_air)

        # Initialize rainfall heat exchange variables
        h_r0 = 0
        T_water = np.nan
        h_evap = 0

        if rainfall > 0:
            # Start a new rain event if previous timestep was dry
            if not rain_event_active:
                rain_event_start_temp = T_surface_K
                rain_event_active = True

            q_sat = calculate_specific_humidity(e_s, P)  # Saturated specific humidity
            q_a = calculate_specific_humidity(e_a, P)  # Ambient specific humidity
            h_evap = calculate_h_evap(rainfall, rho_air, L_latent, C_fc, C_nc, u_s, delta_theta_v, q_sat, q_a)

            h_r0, T_water = calculate_h_r0(dt, rainfall, rain_event_start_temp,
                                           T_dew + 273.15, rho[0] * c[0],
                                           rho_water, cp_water, lam[0])
        else:
            rain_event_active = False

        # Total net heat flux at the surface
        h_net = h_rad - h_evap - h_conv - h_r0

        # ============================
        # Crank-Nicolson Temperature Update
        # ============================

        # Calculate time step ratio for each node
        r = np.zeros(N)
        for i in range(N):
            r[i] = alpha[i] * dt / (2 * dx * dx)

        # Create sparse matrices for Crank-Nicolson system
        # A*T_new = b
        # We need to set up the coefficients for the tridiagonal system

        # Setup diagonal vectors for tridiagonal matrix
        main_diag = np.ones(N)  # Main diagonal
        lower_diag = np.zeros(N - 1)  # Lower diagonal
        upper_diag = np.zeros(N - 1)  # Upper diagonal

        # Setup RHS vector
        b = np.zeros(N)

        # Interior nodes (1 to N-2)
        for i in range(1, N - 1):
            # For interface between different materials, compute average properties
            r_im = 0.5 * (r[i] + r[i - 1])  # Average r between i and i-1
            r_ip = 0.5 * (r[i] + r[i + 1])  # Average r between i and i+1

            # Coefficients for T_new (LHS)
            main_diag[i] = 1 + r_im + r_ip
            lower_diag[i - 1] = -r_im
            upper_diag[i] = -r_ip

            # Coefficients for T (RHS)
            b[i] = T[i] + r_im * (T[i - 1] - T[i]) + r_ip * (T[i + 1] - T[i])

        # Surface boundary (node 0) with heat flux
        # Incorporate the heat flux boundary condition into the Crank-Nicolson scheme
        flux_term = h_net * dx / lam[0]
        main_diag[0] = 1 + 2 * r[0]
        upper_diag[0] = -2 * r[0]
        b[0] = T[0] + 2 * r[0] * (T[1] - T[0] + flux_term)

        # Bottom boundary (node N-1) - Neumann boundary condition (zero flux)
        main_diag[N - 1] = 1 + 2 * r[N - 1]
        lower_diag[N - 2] = -2 * r[N - 1]
        b[N - 1] = T[N - 1] + 2 * r[N - 1] * (T[N - 2] - T[N - 1])

        # Create sparse matrix A
        A = sparse.diags(
            [lower_diag, main_diag, upper_diag],
            [-1, 0, 1],
            shape=(N, N)
        )
        A = A.tocsr()
        # Solve the system
        T = spsolve(A, b)
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
        results['water_temp'].append(T_water)

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


def RMSE(obs_df, modeled_df):
    return np.sqrt(np.mean((obs_df['PavementTemperature'] - modeled_df['surface_temp']) ** 2))

