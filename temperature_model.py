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


def calculate_composite_properties(parameters):
    """
    Calculate effective thermal properties for a composite layer representing all subsurface layers.

    Args:
        parameters: Dictionary containing parameters for all layers

    Returns:
        tuple: (rho_composite, c_composite, lam_composite) - Effective properties for the composite layer
    """
    # Extract layer parameters
    layer1_end = parameters['pavement']['layer1_end']
    layer2_thickness = parameters['pavement']['layer2_thickness']
    layer3_thickness = parameters['pavement']['layer3_thickness']
    total_depth = parameters['pavement']['total_depth']

    # Calculate the thickness of each layer
    layer2_end = layer1_end + layer2_thickness
    layer3_end = layer2_end + layer3_thickness
    layer4_thickness = total_depth - layer3_end

    # Calculate the thickness of the composite layer (all subsurface layers)
    composite_thickness = total_depth - layer1_end

    # Calculate the fractional thickness of each layer in the composite
    f2 = layer2_thickness / composite_thickness
    f3 = layer3_thickness / composite_thickness
    f4 = layer4_thickness / composite_thickness

    # Extract thermal properties for each layer
    l2_rho = parameters['layer2']['rho_l2']
    l2_c = parameters['layer2']['c_l2']
    l2_lam = parameters['layer2']['lam_l2']

    l3_rho = parameters['layer3']['rho_l3']
    l3_c = parameters['layer3']['c_l3']
    l3_lam = parameters['layer3']['lam_l3']

    l4_rho = parameters['layer4']['rho_l4']
    l4_c = parameters['layer4']['c_l4']
    l4_lam = parameters['layer4']['lam_l4']

    # Calculate weighted averages for density and specific heat
    rho_composite = f2 * l2_rho + f3 * l3_rho + f4 * l4_rho

    # For specific heat, calculate the volumetric heat capacity first
    vhc_composite = f2 * l2_rho * l2_c + f3 * l3_rho * l3_c + f4 * l4_rho * l4_c
    c_composite = vhc_composite / rho_composite

    # For thermal conductivity, use the weighted harmonic mean (series resistance model)
    # This is more appropriate for heat conduction through layers
    lam_composite = 1.0 / (f2 / l2_lam + f3 / l3_lam + f4 / l4_lam)

    return rho_composite, c_composite, lam_composite


def calculate_h_infiltration(dt, P_infil, T_sub_K, T_water_in_K, rho_cp_sub, water_rho, water_cp, lam_sub):
    """
    Calculate the conductive heat flux from the subsurface layer to the infiltrating water
    and the resulting infiltrating water temperature.

    This function follows the same 'semi-infinite' conduction approach as 'calculate_h_r0'.

    Args:
        dt            : Time step duration (s)
        P_infil       : Infiltration depth (m) during this time step (assumed = rainfall if fully permeable)
        T_sub_K       : Temperature of the subsurface layer in Kelvin
        T_water_in_K  : Temperature of the water entering infiltration in Kelvin
        rho_cp_sub    : Bulk volumetric heat capacity of the subsurface layer (rho * c)
        water_rho     : Density of water (kg/m³)
        water_cp      : Specific heat of water (J/(kg·K))
        lam_sub       : Thermal conductivity of the subsurface layer (W/m·K)

    Returns:
        h_infil       : Conductive heat flux from subsurface to infiltrating water (W/m²)
        T_water_out_K : Updated water temperature (K) after infiltration heat exchange
    """

    # If there is no infiltration or dt=0, return zeros
    if P_infil <= 0 or dt <= 0:
        return 0.0, T_water_in_K

    # Thermal diffusivity of the subsurface
    D_sub = lam_sub / rho_cp_sub
    # Approx. conduction penetration depth
    delta_sub = np.sqrt(4.0 * D_sub * dt)

    # Effective infiltration "rate" in m/s
    i = P_infil / dt

    # Volumetric heat capacity of water
    rho_cp_w = water_rho * water_cp

    # The same dimensionless 'beta' parameter as in calculate_h_r0
    beta_sub = (delta_sub * rho_cp_sub) / (2.0 * P_infil * rho_cp_w)

    # Heat flux from the subsurface to the infiltrating water
    h_infil = i * rho_cp_w * (T_sub_K - T_water_in_K) * (beta_sub / (1.0 + beta_sub))

    # Updated water temperature after infiltration passes the subsurface
    T_water_out_K = T_water_in_K + (T_sub_K - T_water_in_K) * (beta_sub / (1.0 + beta_sub))

    return h_infil, T_water_out_K

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
        T_water_runoff: Predicted water temperature after heat exchange (K)
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
    # (P * water_rho_cp) * (T_water_runoff - T_dp) = h_ro * dt  => T_water_runoff = T_dp + (h_ro*dt)/(P*water_rho_cp)
    T_water_runoff = T_dp + (T_s0 - T_dp) * (beta / (1 + beta))
    T_water_runoff = T_water_runoff - 273.15 # Convert from K to C
    return h_ro, T_water_runoff


def calculate_h_r0_permeable(parameters, P, T, x, T_water_outflow, lam_composite, subsurface_temp):
    """
    Calculate heat exchange between infiltrating water and permeable pavement layers
    using an improved formulation that explicitly accounts for key parameters and
    incorporates empirical adjustments, including a travel time based on Darcy's law.

    Parameters:
        dt: Time step duration (s)
        P: Rainfall depth during the time step (m)
        phi: Porosity of the surface layer (fraction, 0 < phi <= 1)
        infiltration_rate: Surface infiltration rate (m/s) [volumetric flux]
        T: Temperature profile (°C) across all nodes
        x: Depth array for nodes (m)
        rho: Density profile (kg/m³) across all nodes
        c: Specific heat capacity profile (J/(kg·K)) across all nodes
        lam: Thermal conductivity profile (W/(m·K)) across all nodes
        T_dp: Dew point (rainwater) temperature (°C)
        layer1_end: Depth to the end of the surface layer (m)
        layer2_thickness: Thickness of the second layer (m)
        pore_size: Average pore size (m); default is 0.01 m (1 cm)

    Returns:
        h_r0: Heat flux from pavement to infiltrating water (W/m²)
        T_outflow: Water temperature at outflow (°C)

    Notes:
        - Water properties are assumed to be: density = 1000 kg/m³ and specific heat = 4186 J/(kg·K).
        - The function uses a lumped capacitance (exponential) approach to model the water’s temperature change,
          which is reasonable when the Biot number is small.
        - Empirical factors (contact_factor and specific_surface) are based on spherical pore geometry
          and may require calibration.
        - Darcy's law is used to compute the water travel time: the actual water velocity is infiltration_rate/phi.
    """
    # Load parameters
    dt = parameters['general']['dt']
    layer1_end = parameters['pavement']['layer1_end']
    layer2_thickness = parameters['pavement']['layer2_thickness']
    infiltration_rate = parameters['pavement']['infiltration_rate']
    phi = parameters['pavement']['phi']
    pore_size = parameters['pavement']['pore_size']
    cp_water = parameters['general']['cp_water']
    rho_water = parameters['general']['rho_water']

    # Determine the intended travel depth (e.g., reaching the bottom of the base layer)
    intended_travel_depth = layer1_end + layer2_thickness  # m

    # Using Darcy's law: actual water velocity (seepage velocity) = infiltration_rate / phi
    water_velocity = infiltration_rate / phi  # m/s
    travel_time = intended_travel_depth / water_velocity  # s

    # If the computed travel time is longer than the simulation time step,
    # adjust the travel depth and time to represent partial penetration.
    if travel_time > dt:
        intended_travel_depth = water_velocity * dt  # new travel depth based on dt
        travel_time = dt
    travel_depth = intended_travel_depth

    # Empirical heat transfer factors:
    # Increase in contact area with higher porosity
    contact_factor = 1 + 2 * phi
    specific_surface = 6 * (1 - phi) / (phi * pore_size)
    h_transfer = lam_composite * specific_surface * contact_factor

    temperature_difference = subsurface_temp - T_water_outflow

    # Apply the lumped capacitance model
    exponent = -h_transfer * travel_time / (rho_water * cp_water * travel_depth)
    approach_factor = 1 - np.exp(exponent)

    # Water temperature change (°C)
    T_water_change = temperature_difference * approach_factor

    # Final outflow water temperature (°C)
    T_water_outflow = T_water_outflow + T_water_change

    # Compute the total heat exchanged per unit area (J/m²)
    heat_exchange = P * rho_water * cp_water * T_water_change

    # Convert the heat exchange to a heat flux (W/m²)
    h_r0 = heat_exchange / dt

    return h_r0, T_water_outflow


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
    l2_rho = parameters['layer2']['rho_l2']
    l2_c = parameters['layer2']['c_l2']
    l2_lam = parameters['layer2']['lam_l2']

    # Base layer:
    l3_rho = parameters['layer3']['rho_l3']
    l3_c = parameters['layer3']['c_l3']
    l3_lam = parameters['layer3']['lam_l3']

    # Sub-base layer
    l4_rho = parameters['layer4']['rho_l4']
    l4_c = parameters['layer4']['c_l4']
    l4_lam = parameters['layer4']['lam_l4']

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
        'water_temp_surface': [],
        'water_temp_infil': []
    }

    rain_event_active = False
    rain_event_start_temp = None
    T_water_infil = np.nan
    h_infil = 0.0

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
        T_water_runoff = np.nan
        h_evap = 0

        if rainfall > 0:
            if not rain_event_active:
                rain_event_start_temp = T_surface_K
                rain_event_active = True

            q_sat = calculate_specific_humidity(e_s, P)  # Saturated specific humidity
            q_a = calculate_specific_humidity(e_a, P)  # Ambient specific humidity
            h_evap = calculate_h_evap(rainfall, rho_air, L_latent, C_fc, C_nc, u_s, delta_theta_v, q_sat, q_a)

            current_surface_temp_K = T_surface_K
            h_r0, T_water_runoff = calculate_h_r0(dt, rainfall, current_surface_temp_K,
                                           T_dew + 273.15, rho[0] * c[0],
                                           rho_water, cp_water, lam[0])

            if phi != 0:
                infiltration_depth = rainfall
                T_dew_K = T_dew + 273.15
                T_sub_K = T[1:].mean() + 273.15
                T_water_in_K = (T_water_runoff + 273.15
                                if not np.isnan(T_water_runoff)
                                else T_dew_K)

                h_infil, T_water_infil_K = calculate_h_infiltration(
                    dt,
                    infiltration_depth,
                    T_sub_K,
                    T_water_in_K,
                    rho[1] * c[1],  # sub-surface volumetric heat capacity
                    rho_water,
                    cp_water,
                    lam[1]  # sub-surface thermal conductivity
                )
                T_water_infil = T_water_infil_K - 273.15

        else:
            rain_event_active = False

        # Total net heat flux at the surface
        h_net = h_rad - h_evap - h_conv - h_r0 - h_infil

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
        results['water_temp_surface'].append(T_water_runoff)
        results['water_temp_infil'].append(T_water_infil)
    results['date'] = sim_df['date']
    return pd.DataFrame(results)


def model_pavement_temperature_simplified(sim_df, parameters_file):
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
      - subsurface_temp: The simulated composite temperature (from the well measurement depth)
      - water_temp: Temperature used in rainfall calculations (if any)
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

    # Surface layer properties (Layer 1):
    l1_rho_s = parameters['layer1']['rho_s']
    l1_c_s = parameters['layer1']['c_s']
    l1_lam_s = parameters['layer1']['lam_s']
    l1_rho_f = parameters['layer1']['rho_f']
    l1_c_f = parameters['layer1']['c_f']
    l1_lam_f = parameters['layer1']['lam_f']
    f_lam_layer1 = parameters['layer1']['f_lam_layer1']

    # --- Instead of using individual properties for layers 2, 3, and 4,
    # we compute composite effective properties for the subsurface ---
    rho_composite, c_composite, lam_composite = calculate_composite_properties(parameters)

    # General parameters
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
    layer3_end = layer2_end + layer3_thickness  # Previously base layer boundary
    layer4_end = total_depth  # Sub-base layer (soil bedding)

    # Initialize arrays for density (rho), specific heat (c), and conductivity (lam)
    rho = np.zeros(N)
    c = np.zeros(N)
    lam = np.zeros(N)

    for i, depth in enumerate(x):
        if depth <= layer1_end:
            # --- Layer 1: Permeable surface (porous) ---
            # Compute effective (apparent) properties for the surface layer
            volumetric_heat_capacity = (1 - phi) * (l1_rho_s * l1_c_s) + phi * (l1_rho_f * l1_c_f)
            rho_effective = (1 - phi) * l1_rho_s + phi * l1_rho_f
            c_effective = volumetric_heat_capacity / rho_effective
            lam_effective = (1 - phi) * l1_lam_s + phi * l1_lam_f
            # Apply calibrated multiplier to effective conductivity
            lam_effective *= f_lam_layer1
            rho[i] = rho_effective
            c[i] = c_effective
            lam[i] = lam_effective
        else:
            # --- Subsurface composite: Layers 2, 3, and 4 combined ---
            rho[i] = rho_composite
            c[i] = c_composite
            lam[i] = lam_composite

    # Compute thermal diffusivity for each node
    alpha = lam / (rho * c)

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
        'subsurface_temp': [],
        'water_temp': []
    }

    temperature = []

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
        T_dew = sim_df['DewPoint'].iloc[n]  # °C, dew point temperature
        rainfall = sim_df['Rainfall'].iloc[n]  # Hourly rainfall in mm
        rainfall = rainfall / 1000  # Convert mm to m

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

        h_conv = calculate_h_conv(rho_air, cp_air, C_fc, C_nc, u_s, delta_theta_v, T_surface, T_air)

        # Initialize rainfall heat exchange variables
        h_r0 = 0
        T_water_runoff = np.nan
        h_evap = 0

        if rainfall > 0:
            if phi == 0:  # Conventional impermeable pavement - Water temperature is calculated by the Herb (2009) Equations
                # Start a new rain event if previous timestep was dry
                if not rain_event_active:
                    rain_event_start_temp = T_surface_K
                    rain_event_active = True

                q_sat = calculate_specific_humidity(e_s, P)  # Saturated specific humidity
                q_a = calculate_specific_humidity(e_a, P)  # Ambient specific humidity
                h_evap = calculate_h_evap(rainfall, rho_air, L_latent, C_fc, C_nc, u_s, delta_theta_v, q_sat, q_a)

                h_r0, T_water_runoff = calculate_h_r0(dt, rainfall, rain_event_start_temp,
                                               T_dew + 273.15, rho[0] * c[0],
                                               rho_water, cp_water, lam[0])
            else:
                if not rain_event_active:
                    T_water_outflow = T_dew + 273.15
                    rain_event_active = True



        else:
            rain_event_active = False

        # Total net heat flux at the surface
        h_net = h_rad - h_evap - h_conv - h_r0

        # ============================
        # Crank-Nicolson Temperature Update
        # ============================

        # Calculate time step ratio for each node
        r = alpha * dt / (2 * dx * dx)

        # Setup tridiagonal matrix coefficients and right-hand side vector b
        main_diag = np.ones(N)
        lower_diag = np.zeros(N - 1)
        upper_diag = np.zeros(N - 1)
        b = np.zeros(N)

        # Interior nodes (1 to N-2)
        for i in range(1, N - 1):
            # Compute average r between adjacent nodes
            r_im = 0.5 * (r[i] + r[i - 1])
            r_ip = 0.5 * (r[i] + r[i + 1])

            # Coefficients for T_new (LHS)
            main_diag[i] = 1 + r_im + r_ip
            lower_diag[i - 1] = -r_im
            upper_diag[i] = -r_ip

            # Coefficients for T (RHS)
            b[i] = T[i] + r_im * (T[i - 1] - T[i]) + r_ip * (T[i + 1] - T[i])

        # Surface boundary (node 0) with heat flux boundary condition
        flux_term = h_net * dx / lam[0]
        main_diag[0] = 1 + 2 * r[0]
        upper_diag[0] = -2 * r[0]
        b[0] = T[0] + 2 * r[0] * (T[1] - T[0] + flux_term)

        # Bottom boundary (node N-1): Neumann boundary condition (zero flux)
        main_diag[N - 1] = 1 + 2 * r[N - 1]
        lower_diag[N - 2] = -2 * r[N - 1]
        b[N - 1] = T[N - 1] + 2 * r[N - 1] * (T[N - 2] - T[N - 1])

        # Create sparse matrix A
        A = sparse.diags(
            [lower_diag, main_diag, upper_diag],
            [-1, 0, 1],
            shape=(N, N)
        ).tocsr()

        # Solve for new temperature distribution
        T = spsolve(A, b)
        surface_temp = T[0]

        # --- Retrieve the composite (sub-surface) temperature ---
        # We assume that all nodes with depth > layer1_end represent the well temperature.
        composite_indices = np.where(x >= layer3_end)[0]
        subsurface_temp = np.mean(T[composite_indices])

        # Record each term and temperatures for this timestep
        results['h_s'].append(h_s)
        results['h_li'].append(h_li)
        results['h_l0'].append(h_l0)
        results['h_rad'].append(h_rad)
        results['h_evap'].append(h_evap)
        results['h_conv'].append(h_conv)
        results['h_r0'].append(h_r0)
        results['h_net'].append(h_net)
        results['surface_temp'].append(surface_temp)
        results['subsurface_temp'].append(subsurface_temp)
        results['water_temp'].append(T_water_runoff)
        temperature.append(T)

    return pd.DataFrame(results), temperature


def NSE(obs_df, modeled_df):
    obs = obs_df['PavementTemperature']
    modeled = modeled_df['surface_temp']
    denom = np.sum((obs - np.mean(obs)) ** 2)
    return 1 - np.sum((obs - modeled) ** 2) / denom


def NSE_bottom(bottom_df):
    obs = bottom_df['bottom_temperature_observed']
    modeled = bottom_df['bottom_temperature_modeled']
    denom = np.sum((obs - np.mean(obs)) ** 2)
    return 1 - np.sum((obs - modeled) ** 2) / denom


def RMSE(obs_df, modeled_df):
    return np.sqrt(np.mean((obs_df['PavementTemperature'] - modeled_df['surface_temp']) ** 2))


def RMSE_bottom(bottom_df):
    obs = bottom_df['bottom_temperature_observed']
    modeled = bottom_df['bottom_temperature_modeled']
    return np.sqrt(np.mean((obs - modeled) ** 2))