import numpy as np

def model_pavement_temperature(sim_df, params):
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
