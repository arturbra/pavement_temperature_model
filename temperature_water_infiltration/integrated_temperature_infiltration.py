"""
Integrated Temperature and Water Infiltration Model for Permeable Pavements

This module combines the pavement temperature model with the water flow/infiltration model
to provide a comprehensive simulation that accounts for:
1. Heat transfer through pavement layers
2. Water infiltration and storage
3. Heat exchange between water and pavement materials
4. Dynamic thermal properties based on water content

"""

import numpy as np
import pandas as pd
import configparser
from scipy import sparse
from scipy.sparse.linalg import spsolve
from datetime import datetime
import water_flow_module as wfm


def load_parameters(temperature_params_file, water_flow_params_file):
    """
    Load parameters from both temperature and water flow parameter files.

    Args:
        temperature_params_file: Path to temperature model parameters (.ini)
        water_flow_params_file: Path to water flow model parameters (.ini)

    Returns:
        tuple: (temp_params, wf_params) - dictionaries containing parameters
    """
    # Load temperature parameters
    config_temp = configparser.ConfigParser()
    config_temp.read(temperature_params_file)

    def parse_section(section):
        return {k: float(v) for k, v in section.items()}

    temp_params = {section: parse_section(config_temp[section])
                   for section in config_temp.sections()}

    # Override with calibrated parameters if they exist
    if "calibration" in temp_params:
        for key, value in temp_params["calibration"].items():
            for section in temp_params:
                if key in temp_params[section] and section != "calibration":
                    temp_params[section][key] = value

    # Load water flow parameters
    setup_wf = wfm.read_setup_file(water_flow_params_file)
    wf_params = wfm.initialize_parameters(setup_wf)

    return temp_params, wf_params


def update_thermal_properties_with_moisture(rho_dry, c_dry, lam_dry,
                                            theta, rho_water, c_water, lam_water):
    """
    Update thermal properties based on volumetric water content.

    The effective properties are computed as a weighted average between
    dry material properties and water properties based on the volumetric
    water content (theta).

    Args:
        rho_dry: Dry density of material (kg/m³)
        c_dry: Dry specific heat of material (J/(kg·K))
        lam_dry: Dry thermal conductivity (W/(m·K))
        theta: Volumetric water content (-)
        rho_water: Water density (kg/m³)
        c_water: Water specific heat (J/(kg·K))
        lam_water: Water thermal conductivity (W/(m·K))

    Returns:
        tuple: (rho_eff, c_eff, lam_eff) - effective thermal properties
    """
    # Effective density (mass-weighted)
    rho_eff = (1 - theta) * rho_dry + theta * rho_water

    # Effective volumetric heat capacity
    vol_heat_cap = (1 - theta) * rho_dry * c_dry + theta * rho_water * c_water

    # Effective specific heat
    c_eff = vol_heat_cap / rho_eff

    # Effective thermal conductivity (geometric mean model - better for porous media)
    lam_eff = lam_dry ** (1 - theta) * lam_water ** theta

    return rho_eff, c_eff, lam_eff


def run_integrated_model(sim_df, temp_params_file, wf_params_file,
                        rainfall_column='Rainfall', rainfall_unit='mm',
                        climate_change=False):
    """
    Run the integrated temperature and water infiltration model.

    This function couples the temperature model with the water flow model,
    allowing for:
    - Dynamic thermal properties based on water content
    - Heat exchange between infiltrating water and pavement layers
    - Water storage and movement through pavement layers

    Args:
        sim_df: DataFrame with meteorological forcing data
        temp_params_file: Path to temperature parameters file
        wf_params_file: Path to water flow parameters file
        rainfall_column: Name of rainfall column in sim_df
        rainfall_unit: Unit of rainfall ('mm' or 'in')
        climate_change: Whether using climate change forcing data

    Returns:
        tuple: (results_df, water_flow_df) - DataFrames with results
    """

    # Load parameters
    temp_params, wf_params = load_parameters(temp_params_file, wf_params_file)

    # Import temperature model functions
    from temperature_model import (
        calculate_h_s, calculate_h_li, calculate_h_li_climate_change,
        calculate_h_l0, calculate_h_rad, calculate_h_conv, calculate_h_evap,
        calculate_u_s, compute_convection_coefficients,
        calculate_saturation_vapor_pressure, calculate_specific_humidity,
        calculate_tdew, calculate_h_r0, calculate_h_infiltration
    )

    # Extract temperature model parameters
    rho_air = temp_params['convection']['rho_air']
    cp_air = temp_params['convection']['cp_air']
    Lv = temp_params['convection']["lv"]
    g = temp_params['convection']["g"]
    beta = temp_params['convection']["beta"]
    nu_air = temp_params['convection']["nu_air"]
    Pr = temp_params['convection']["pr"]
    k_air = temp_params['convection']["k_air"]
    L = temp_params['convection']["l"]
    h_forced = temp_params['convection']["h_forced"]

    # Pavement parameters
    phi = temp_params['pavement']['phi']
    total_depth = temp_params['pavement']['total_depth']
    dx = temp_params['pavement']['dx']
    layer1_end = temp_params['pavement']['layer1_end']
    layer2_thickness = temp_params['pavement']['layer2_thickness']
    layer3_thickness = temp_params['pavement']['layer3_thickness']
    reflectivity = temp_params['pavement']['reflectivity']
    emissivity = temp_params['pavement']['emissivity']

    # Layer properties (dry)
    l1_rho_s = temp_params['layer1']['rho_s']
    l1_c_s = temp_params['layer1']['c_s']
    l1_lam_s = temp_params['layer1']['lam_s']
    l1_rho_f = temp_params['layer1']['rho_f']
    l1_c_f = temp_params['layer1']['c_f']
    l1_lam_f = temp_params['layer1']['lam_f']
    f_lam_layer1 = temp_params['layer1']['f_lam_layer1']

    l2_rho = temp_params['layer2']['rho_l2']
    l2_c = temp_params['layer2']['c_l2']
    l2_lam = temp_params['layer2']['lam_l2']

    l3_rho = temp_params['layer3']['rho_l3']
    l3_c = temp_params['layer3']['c_l3']
    l3_lam = temp_params['layer3']['lam_l3']

    l4_rho = temp_params['layer4']['rho_l4']
    l4_c = temp_params['layer4']['c_l4']
    l4_lam = temp_params['layer4']['lam_l4']

    # General parameters
    dt = temp_params['general']['dt']
    P = temp_params['general']['p']
    CSh = temp_params['general']['csh']
    sigma_const = temp_params['general']['sigma_const']
    initial_temperature = temp_params['general']['initial_temperature']
    L_latent = temp_params['general']['l_latent']
    rho_water = temp_params['general']['rho_water']
    cp_water = temp_params['general']['cp_water']

    # Setup spatial discretization
    N = int(np.ceil(total_depth / dx)) + 1
    x = np.linspace(0, total_depth, N)

    # Define layer boundaries
    layer2_end = layer1_end + layer2_thickness
    layer3_end = layer2_end + layer3_thickness
    layer4_end = total_depth

    # Initialize arrays for DRY properties
    rho_dry = np.zeros(N)
    c_dry = np.zeros(N)
    lam_dry = np.zeros(N)

    # Assign dry properties to each layer
    for i, depth in enumerate(x):
        if depth <= layer1_end:
            # Layer 1: Permeable surface
            volumetric_heat_capacity = (1 - phi) * (l1_rho_s * l1_c_s) + phi * (l1_rho_f * l1_c_f)
            rho_effective = (1 - phi) * l1_rho_s + phi * l1_rho_f
            c_effective = volumetric_heat_capacity / rho_effective
            lam_effective = (1 - phi) * l1_lam_s + phi * l1_lam_f
            lam_effective *= f_lam_layer1
            rho_dry[i] = rho_effective
            c_dry[i] = c_effective
            lam_dry[i] = lam_effective
        elif depth <= layer2_end:
            # Layer 2: Leveling layer
            rho_dry[i] = l2_rho
            c_dry[i] = l2_c
            lam_dry[i] = l2_lam
        elif depth <= layer3_end:
            # Layer 3: Base layer
            rho_dry[i] = l3_rho
            c_dry[i] = l3_c
            lam_dry[i] = l3_lam
        else:
            # Layer 4: Soil bedding
            rho_dry[i] = l4_rho
            c_dry[i] = l4_c
            lam_dry[i] = l4_lam

    # Initialize temperature field
    T = np.ones(N) * initial_temperature

    # Initialize water flow state variables
    hp_prev = 0  # ponding depth
    s_prev = 0  # saturation of unsaturated zone
    hsz_prev = max(wf_params['hpipe'], 1e-6) if wf_params['hpipe'] > 0 else wf_params['dpipe']
    husz_prev = max(wf_params['husz_ini'], 1e-6)
    nusz_prev = max(wf_params['nusz_ini'], 1e-6)
    nsz_prev = max(wf_params['ng'], 1e-6)


    n_steps = len(sim_df)

    # Initialize results storage
    results = {
        'date': [],
        'h_s': [], 'h_li': [], 'h_l0': [], 'h_rad': [],
        'h_evap': [], 'h_conv': [], 'h_r0': [], 'h_infil': [], 'h_net': [],
        'surface_temp': [], 'subsurface_temp': [],
        'water_temp_surface': [], 'water_temp_infil': [],
        'theta_surface': [], 'theta_subsurface': [],
        'ponding_depth': [], 'saturation': []
    }

    water_flow_results = {
        'date': [], 'Qpipe': [], 'Qover': [], 'Qpf': [], 'Qfs': [],
        's': [], 'hsz': [], 'husz': [], 'hp': []
    }

    # Initialize rain event tracking variables
    rain_event_active = False
    rain_event_start_temp = None

    # Initialize water temperature (carry forward between timesteps)
    T_water_infil_prev = np.nan

    # Time integration loop
    for n in range(n_steps):
        # Extract meteorological data
        if climate_change:
            I = sim_df['rsds'].iloc[n]
            T_air = sim_df['tas'].iloc[n]
            v = sim_df['sfcwind'].iloc[n]
            RH = sim_df['hurs'].iloc[n] / 100
            RLDS = sim_df['rlds'].iloc[n]
            rainfall = sim_df['pr'].iloc[n] / 1000  # mm to m
            T_dew = calculate_tdew(T_air, RH)
        else:
            I = sim_df['SolarRadiation'].iloc[n]
            T_air = sim_df['AirTemperature'].iloc[n]
            v = sim_df['WindSpeed'].iloc[n]
            RH = sim_df['RelativeHumidity'].iloc[n]
            CR = sim_df['CloudCoverage'].iloc[n]
            T_dew = sim_df['DewPoint'].iloc[n]
            rainfall = sim_df[rainfall_column].iloc[n] / 1000  # mm to m

        # Convert temperatures to Kelvin
        T_surface = T[0]
        T_surface_K = T_surface + 273.15
        T_air_K = T_air + 273.15

        # === WATER FLOW MODEL STEP ===
        # Convert rainfall to flow rate (m³/s)
        # Handle NaN rainfall values by treating as zero
        rainfall_clean = rainfall if not np.isnan(rainfall) else 0.0
        qrain = rainfall_clean * wf_params['area'] / dt
        qin = 0  # No additional inflow
        emax = 0  # No evapotranspiration for now (could be coupled later)

        # Run single water flow timestep
        wf_state = wfm.run_single_timestep(
            wf_params, qin, qrain, emax,
            hp_prev, s_prev, hsz_prev, husz_prev,
            nusz_prev, nsz_prev
        )

        # Update water flow state
        hp_prev = wf_state['hp']
        s_prev = wf_state['s']
        hsz_prev = wf_state['hsz']
        husz_prev = wf_state['husz']
        nusz_prev = wf_state['nusz']
        nsz_prev = wf_state['nsz']
        qpipe_prev = wf_state['Qpipe']

        # Calculate volumetric water content (theta) for each layer
        theta = np.zeros(N)
        for i, depth in enumerate(x):
            if depth <= layer1_end:
                # Surface layer - use unsaturated zone saturation
                theta[i] = s_prev * nusz_prev
            elif depth <= wf_params['l']:
                # Within pavement depth - check if saturated or unsaturated
                if depth <= (wf_params['l'] - husz_prev):
                    # In saturated zone
                    theta[i] = nsz_prev
                else:
                    # In unsaturated zone
                    theta[i] = s_prev * nusz_prev
            else:
                # Below pavement - assume some residual moisture
                theta[i] = 0.05  # 5% moisture content

        # Update thermal properties based on moisture content
        rho = np.zeros(N)
        c = np.zeros(N)
        lam = np.zeros(N)

        lam_water = 0.6  # W/(m·K)

        for i in range(N):
            rho[i], c[i], lam[i] = update_thermal_properties_with_moisture(
                rho_dry[i], c_dry[i], lam_dry[i],
                theta[i], rho_water, cp_water, lam_water
            )

        # === TEMPERATURE MODEL STEP ===
        # Calculate radiation terms
        h_s = calculate_h_s(reflectivity, I)
        e_s = calculate_saturation_vapor_pressure(T_air)
        e_a = e_s * RH

        if climate_change:
            h_li = calculate_h_li_climate_change(emissivity, RLDS)
        else:
            h_li = calculate_h_li(emissivity, sigma_const, CR, e_a / 100, T_air_K)

        h_l0 = calculate_h_l0(emissivity, sigma_const, T_surface_K)
        h_rad = calculate_h_rad(h_s, h_li, h_l0)

        # Calculate convection
        u_s = calculate_u_s(CSh, v)
        C_fc, C_nc = compute_convection_coefficients(
            h_forced, rho_air, cp_air, g, beta, L, nu_air, Pr, k_air, Lv, v,
            T_air, T[0]
        )
        delta_theta_v = T_surface_K - T_air_K
        h_conv = calculate_h_conv(rho_air, cp_air, C_fc, C_nc, u_s, delta_theta_v,
                                  T_surface, T_air)

        # Calculate evaporation and rainfall heat exchange
        h_evap = 0
        h_r0 = 0
        h_infil = 0
        T_water_runoff = np.nan
        T_water_infil = np.nan

        # Determine if we should calculate water temperature
        # Calculate when: (1) there's rainfall, OR (2) there's water in saturated zone, OR (3) there's pipe outflow
        has_water = rainfall_clean > 0 or hsz_prev > 0 or qpipe_prev > 0

        if has_water:
            # Start rain event tracking if rainfall is active
            if rainfall_clean > 0:
                if not rain_event_active:
                    rain_event_start_temp = T_surface_K
                    rain_event_active = True

                q_sat = calculate_specific_humidity(e_s, P)
                q_a = calculate_specific_humidity(e_a, P)
                h_evap = calculate_h_evap(rho_air, L_latent, C_fc, C_nc, u_s,
                                         delta_theta_v, q_sat, q_a)

                # STAGE 1: Surface water heat exchange (calculate_h_r0)
                # Water contacts the hot pavement surface
                current_surface_temp_K = T_surface_K
                h_r0, T_water_runoff = calculate_h_r0(
                    dt, rainfall_clean, current_surface_temp_K,
                    T_dew + 273.15, rho[0] * c[0],
                    rho_water, cp_water, lam[0]
                )

            # STAGE 2: Infiltration/stored water heat exchange (if permeable and has water)
            if phi != 0 and (rainfall_clean > 0 or hsz_prev > 0):
                if rainfall_clean > 0:
                    infiltration_depth = rainfall_clean  # Assume all rainfall infiltrates
                else:
                    # Use saturated zone depth as proxy for water volume when no rainfall
                    infiltration_depth = hsz_prev / 1000  # Convert from mm to m

                T_dew_K = T_dew + 273.15

                # Average subsurface temperature
                T_sub_K = T[1:].mean() + 273.15

                # Input water temperature for infiltration
                if rainfall_clean > 0:
                    # Use surface-heated water temp if available, otherwise dew point
                    T_water_in_K = (T_water_runoff + 273.15
                                    if not np.isnan(T_water_runoff)
                                    else T_dew_K)
                else:
                    # For stored water, use previous water temperature or subsurface temp
                    if not np.isnan(T_water_infil_prev):
                        T_water_in_K = T_water_infil_prev + 273.15
                    else:
                        T_water_in_K = T_sub_K

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
            T_water_infil = np.nan

        # Total net heat flux
        h_net = h_rad - h_evap - h_conv - h_r0 - h_infil

        # === CRANK-NICOLSON TEMPERATURE UPDATE ===
        alpha = lam / (rho * c)
        r = alpha * dt / (2 * dx * dx)

        main_diag = np.ones(N)
        lower_diag = np.zeros(N - 1)
        upper_diag = np.zeros(N - 1)
        b = np.zeros(N)

        # Interior nodes
        for i in range(1, N - 1):
            r_im = 0.5 * (r[i] + r[i - 1])
            r_ip = 0.5 * (r[i] + r[i + 1])

            main_diag[i] = 1 + r_im + r_ip
            lower_diag[i - 1] = -r_im
            upper_diag[i] = -r_ip

            b[i] = T[i] + r_im * (T[i - 1] - T[i]) + r_ip * (T[i + 1] - T[i])

        # Surface boundary with heat flux
        flux_term = h_net * dx / lam[0]
        main_diag[0] = 1 + 2 * r[0]
        upper_diag[0] = -2 * r[0]
        b[0] = T[0] + 2 * r[0] * (T[1] - T[0] + flux_term)

        # Bottom boundary (zero flux)
        main_diag[N - 1] = 1 + 2 * r[N - 1]
        lower_diag[N - 2] = -2 * r[N - 1]
        b[N - 1] = T[N - 1] + 2 * r[N - 1] * (T[N - 2] - T[N - 1])

        # Solve system
        A = sparse.diags([lower_diag, main_diag, upper_diag], [-1, 0, 1],
                        shape=(N, N)).tocsr()
        T = spsolve(A, b)

        # Calculate subsurface temperature
        subsurface_indices = np.where(x >= layer3_end)[0]
        subsurface_temp = np.mean(T[subsurface_indices])

        # Store results
        results['date'].append(sim_df['date'].iloc[n] if 'date' in sim_df.columns else n)
        results['h_s'].append(h_s)
        results['h_li'].append(h_li)
        results['h_l0'].append(h_l0)
        results['h_rad'].append(h_rad)
        results['h_evap'].append(h_evap)
        results['h_conv'].append(h_conv)
        results['h_r0'].append(h_r0)
        results['h_infil'].append(h_infil)
        results['h_net'].append(h_net)
        results['surface_temp'].append(T[0])
        results['subsurface_temp'].append(subsurface_temp)
        results['water_temp_surface'].append(T_water_runoff)
        results['water_temp_infil'].append(T_water_infil)
        results['theta_surface'].append(theta[0])
        results['theta_subsurface'].append(np.mean(theta[1:]))
        results['ponding_depth'].append(hp_prev)
        results['saturation'].append(s_prev)

        # Update previous water temperature for next timestep
        T_water_infil_prev = T_water_infil

        # Store water flow results
        water_flow_results['date'].append(sim_df['date'].iloc[n] if 'date' in sim_df.columns else n)
        water_flow_results['Qpipe'].append(wf_state['Qpipe'])
        water_flow_results['Qover'].append(0)  # Would need to store from timestep
        water_flow_results['Qpf'].append(0)
        water_flow_results['Qfs'].append(0)
        water_flow_results['s'].append(s_prev)
        water_flow_results['hsz'].append(hsz_prev)
        water_flow_results['husz'].append(husz_prev)
        water_flow_results['hp'].append(hp_prev)

    return pd.DataFrame(results), pd.DataFrame(water_flow_results)


if __name__ == "__main__":
    print("Integrated Temperature and Water Infiltration Model")
    print("=" * 60)
    print("\nThis module combines:")
    print("  1. Pavement temperature model with heat transfer")
    print("  2. Water infiltration model with storage dynamics")
    print("  3. Coupled heat-moisture interactions")
    print("\nUse run_integrated_model() to execute simulations.")
