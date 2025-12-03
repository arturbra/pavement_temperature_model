# Water Temperature Calculation - Critical Fix

## Issue Identified

You correctly identified that the integrated model had **oversimplified the water temperature calculation**, which is one of the main contributions of your model. The original implementation was incomplete and did not properly represent the physics of heat exchange between water and pavement.

## What Was Wrong

### Original (Incorrect) Implementation:
```python
# WRONG - Oversimplified
if rainfall > 0 and hp_prev > 0:
    T_water_surface = (T_surface + T_dew) / 2  # Simple average - NO PHYSICS!
    h_r0 = hp_prev * rho_water * cp_water * (T_surface - T_dew) / dt  # Incorrect

    if s_prev > 0:
        T_sub_avg = T[1:].mean()
        T_water_infil = (T_water_surface + T_sub_avg) / 2  # Another simple average - NO PHYSICS!
```

**Problems:**
1. Used simple arithmetic averages instead of proper heat transfer equations
2. Ignored the semi-infinite conduction approach (beta factor method)
3. Did not use the calibrated `calculate_h_r0()` function
4. Did not use the validated `calculate_h_infiltration()` function
5. Lost all the physics that makes your model unique

## What Was Fixed

### Corrected Implementation (Following temperature_model.py):

```python
if rainfall > 0:
    # Start rain event tracking
    if not rain_event_active:
        rain_event_start_temp = T_surface_K
        rain_event_active = True

    # Calculate evaporation
    q_sat = calculate_specific_humidity(e_s, P)
    q_a = calculate_specific_humidity(e_a, P)
    h_evap = calculate_h_evap(rho_air, L_latent, C_fc, C_nc, u_s,
                             delta_theta_v, q_sat, q_a)

    # STAGE 1: Surface water heat exchange (calculate_h_r0)
    # Water contacts the hot pavement surface
    current_surface_temp_K = T_surface_K
    h_r0, T_water_runoff = calculate_h_r0(
        dt, rainfall, current_surface_temp_K,
        T_dew + 273.15, rho[0] * c[0],
        rho_water, cp_water, lam[0]
    )

    # STAGE 2: Infiltration heat exchange (if permeable)
    if phi != 0:
        infiltration_depth = rainfall  # Assume all rainfall infiltrates
        T_dew_K = T_dew + 273.15
        T_sub_K = T[1:].mean() + 273.15

        # Input water temperature for infiltration
        # Use surface-heated water temp if available, otherwise dew point
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
    T_water_infil = np.nan

# Total net heat flux (INCLUDING h_infil)
h_net = h_rad - h_evap - h_conv - h_r0 - h_infil
```

## Physics of Water Temperature Calculation

### Stage 1: Surface Contact (`calculate_h_r0`)

**Physical Process:**
- Rainwater (at dew point temperature) contacts hot pavement surface
- Heat transfer occurs via conduction through a thin water film
- Semi-infinite conduction approximation is used

**Key Equations:**
```
D = λ_surface / (ρ_surface * c_surface)  # Thermal diffusivity
δ = √(4 * D * dt)                        # Penetration depth
β = (δ * ρ_surface * c_surface) / (2 * P * ρ_water * c_water)  # Beta factor

h_r0 = (P/dt) * ρ_water * c_water * (T_surface - T_dew) * (β / (1 + β))

T_water_runoff = T_dew + (T_surface - T_dew) * (β / (1 + β))
```

**Physical Meaning:**
- β represents the ratio of pavement heat capacity to water heat capacity
- When β >> 1: Water temperature approaches pavement temperature (pavement dominates)
- When β << 1: Water temperature stays near dew point (water dominates)

### Stage 2: Infiltration (`calculate_h_infiltration`)

**Physical Process:**
- Surface-heated water infiltrates through subsurface layers
- Additional heat exchange with subsurface materials
- Similar semi-infinite conduction approach

**Key Equations:**
```
D_sub = λ_sub / (ρ_sub * c_sub)
δ_sub = √(4 * D_sub * dt)
β_sub = (δ_sub * ρ_sub * c_sub) / (2 * P_infil * ρ_water * c_water)

h_infil = (P/dt) * ρ_water * c_water * (T_sub - T_water_in) * (β_sub / (1 + β_sub))

T_water_infil = T_water_in + (T_sub - T_water_in) * (β_sub / (1 + β_sub))
```

**Physical Meaning:**
- Water enters subsurface at T_water_in (from surface contact)
- Exchanges heat with subsurface materials
- Final temperature T_water_infil is what reaches the drainage system

## Heat Balance

The complete heat balance now properly accounts for:

```
h_net = h_rad - h_evap - h_conv - h_r0 - h_infil

Where:
  h_rad    = Net radiation (solar + longwave)
  h_evap   = Evaporative cooling (latent heat)
  h_conv   = Convective heat exchange
  h_r0     = Heat flux to surface water (Stage 1)
  h_infil  = Heat flux to infiltrating water (Stage 2)
  h_net    = Net heat flux to pavement surface
```

**Critical:** Both h_r0 AND h_infil must be subtracted from the surface energy balance because:
1. h_r0: Heat lost to warm surface runoff water
2. h_infil: Heat lost to warm infiltrating water in subsurface

## Validation Updates

The validation script now:

1. **Compares the correct variable:**
   - Uses `water_temp_infil` (infiltrated water temperature) as the main output
   - Also tracks `water_temp_surface` (after surface contact) for completeness

2. **Creates enhanced plots:**
   - Time series of observed vs. modeled infiltrated water temperature
   - Comparison plot showing both surface and infiltrated water temperatures
   - Scatter plot for infiltrated water temperature validation

3. **Reports both metrics:**
   - Water Temperature (Infiltrated) - Main comparison
   - Water Temperature (Surface) - For understanding the two-stage process

4. **Shows all heat fluxes:**
   - Heat flux plot now includes both h_r0 and h_infil
   - Clearly labeled as "Rainfall (Surface)" and "Infiltration (Subsurface)"

## Why This Matters

### Scientific Contribution
Your model's main contribution is modeling infiltrated water temperature, which most models ignore. The proper physics includes:

1. **Two-stage heating process:**
   - Surface contact heating
   - Subsurface infiltration heating

2. **Energy conservation:**
   - Proper accounting of heat fluxes
   - Correct thermal properties in heat exchange

3. **Validated approach:**
   - Beta factor method from heat transfer literature
   - Calibrated against observations
   - Physically meaningful parameters

### Practical Impact
For permeable pavements:
- Water temperature affects thermal stress
- Affects drainage system design
- Important for heat island mitigation
- Critical for thermal runoff impacts

## Files Modified

### 1. `integrated_temperature_infiltration.py`
**Changes:**
- Added imports: `calculate_h_r0`, `calculate_h_infiltration`
- Replaced simplified water temp calculation with proper physics
- Added rain event tracking (`rain_event_active`, `rain_event_start_temp`)
- Added h_infil to results storage
- Fixed h_net calculation: `h_net = h_rad - h_evap - h_conv - h_r0 - h_infil`

### 2. `validate_PA_pavement.py`
**Changes:**
- Compare observed vs. `water_temp_infil` (main output)
- Also track `water_temp_surface` for completeness
- Enhanced water temperature plots (3 subplots):
  - Observed vs. modeled infiltrated water
  - Surface vs. infiltrated water comparison
  - Scatter plot for validation
- Updated heat flux plot to show h_infil
- Reports metrics for both water temperatures

## Testing

To verify the fix works correctly:

1. **Run the validation:**
   ```bash
   python validate_PA_pavement.py
   ```

2. **Check outputs:**
   - Water temperature metrics should now be calculated (not "No data")
   - `water_temperature_comparison.png` should show 3 subplots
   - Heat flux plot should show 6 components (including h_infil)

3. **Verify physics:**
   - T_water_surface should be between T_dew and T_surface
   - T_water_infil should be between T_water_surface and T_subsurface
   - h_r0 should be non-zero during rainfall
   - h_infil should be non-zero during rainfall (if phi > 0)

## Summary

**What was wrong:** Oversimplified water temperature using simple averages

**What is fixed:** Proper two-stage heat exchange physics using validated functions

**Result:** Model now properly represents your key contribution - infiltrated water temperature modeling with correct physics

This fix ensures the integrated model maintains the scientific rigor and physical accuracy of your original temperature model.

---

*Thank you for catching this critical issue!*
