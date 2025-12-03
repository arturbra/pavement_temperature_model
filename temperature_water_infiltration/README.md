# Integrated Temperature and Water Infiltration Model for Permeable Pavements

## Overview

This folder contains an integrated model that combines:
1. **Pavement Temperature Model** - Heat transfer through pavement layers with surface energy balance
2. **Water Infiltration Model** - Water flow through permeable pavement layers with storage dynamics

The integration addresses a key limitation of the standalone temperature model: it now accounts for actual water volume and storage in the pavement, rather than assuming all water leaves immediately during each timestep.

## Key Features

### Temperature Model Components
- Surface energy balance (radiation, convection, evaporation)
- Multi-layer heat conduction (Crank-Nicolson scheme)
- Heat exchange between rainfall water and pavement
- Dynamic thermal properties based on water content

### Water Infiltration Model Components
- Ponding zone dynamics with overflow
- Unsaturated zone water movement
- Saturated zone storage
- Pipe/underdrain outflow
- Layer-specific porosity and hydraulic properties

### Integration Features
- **Dynamic thermal properties**: Material properties (density, specific heat, thermal conductivity) are updated based on volumetric water content from the water flow model
- **Coupled heat-moisture transfer**: Water saturation affects heat storage and conduction
- **Water storage tracking**: Realistic water retention in pavement layers instead of immediate drainage
- **Heat exchange with infiltrating water**: Energy transfer between water and pavement materials during infiltration

## Folder Structure

```
temperature_water_infiltration/
├── README.md                                    # This file
├── integrated_temperature_infiltration.py       # Main integrated model
├── water_flow_module.py                        # Water infiltration model (adapted from GreenDT)
├── example_run_integrated_model.py             # Example usage script
├── input_data/
│   ├── temperature_parameters.ini              # Temperature model parameters
│   └── water_flow_parameters.ini               # Water flow model parameters
└── output/
    ├── integrated_results.csv                  # Temperature and moisture results
    └── water_flow_results.csv                  # Detailed water flow results
```

## Files Description

### Core Model Files

#### `integrated_temperature_infiltration.py`
The main integration module that couples temperature and water flow models. Key functions:
- `load_parameters()`: Loads parameters from both model configuration files
- `update_thermal_properties_with_moisture()`: Updates material properties based on water content
- `run_integrated_model()`: Main simulation function that runs the coupled model

#### `water_flow_module.py`
Adapted from the GreenDT project's calibrated and validated water flow model. Includes:
- Multi-zone water flow (ponding, unsaturated, saturated)
- Infiltration and drainage calculations
- Time-stepping functions for integration with temperature model

### Configuration Files

#### `input_data/temperature_parameters.ini`
Temperature model parameters organized in sections:
- `[convection]`: Air properties and convection parameters
- `[pavement]`: Pavement geometry and optical properties
- `[layer1-4]`: Thermal properties for each pavement layer
- `[general]`: Simulation settings and physical constants
- `[calibration]`: Calibrated parameter values

#### `input_data/water_flow_parameters.ini`
Water flow model parameters organized in sections:
- `[GENERAL]`: Layer depths, porosities
- `[PONDING_ZONE]`: Surface ponding parameters
- `[UNSATURATED_ZONE]`: Hydraulic properties for unsaturated flow
- `[SATURATED_ZONE]`: Storage and drainage parameters
- `[TIMESTEP]`: Time discretization
- `[CALIBRATION]`: Calibration ranges for parameters

### Example and Output Files

#### `example_run_integrated_model.py`
Comprehensive example script showing:
- How to load meteorological forcing data
- How to set up and run the integrated model
- How to save and analyze results
- Synthetic data generation for testing

#### Output Files (Generated)
- `integrated_results.csv`: Temperature, heat fluxes, moisture content, saturation
- `water_flow_results.csv`: Water flow rates, depths, storage states

## Dependencies

### Required Python Packages
```python
numpy>=1.20.0          # Numerical arrays and operations
pandas>=1.3.0          # Data handling and analysis
scipy>=1.7.0           # Sparse matrices for heat equation solver
configparser           # Parameter file reading (standard library)
datetime               # Time handling (standard library)
```

### Model Dependencies
- **temperature_model.py**: Located in parent directory, provides temperature calculation functions
  - Path: `C:\Users\Artur\PycharmProjects\pavement_temperature_model\temperature_model.py`

### Original Source Models
- **Temperature Model**: From pavement_temperature_model project (current directory parent)
- **Water Flow Model**: Adapted from GreenDT project
  - Original path: `C:\Users\Artur\PycharmProjects\GreenDT\greendt\water_flow_module.py`
  - Modified to remove GreenDT package dependencies

## Installation and Setup

### 1. Ensure Dependencies are Installed
```bash
pip install numpy pandas scipy
```

### 2. Verify File Structure
Ensure the following files exist:
- Parent directory: `temperature_model.py`
- This directory: `integrated_temperature_infiltration.py`, `water_flow_module.py`
- Input data: Both `.ini` parameter files in `input_data/`

### 3. Prepare Input Data
Your meteorological forcing data should include:
- `date`: Datetime stamps
- `AirTemperature`: Air temperature (°C)
- `SolarRadiation`: Incoming solar radiation (W/m²)
- `WindSpeed`: Wind speed at 10m (m/s)
- `RelativeHumidity`: Relative humidity (0-1)
- `CloudCoverage`: Cloud coverage fraction (0-1)
- `DewPoint`: Dew point temperature (°C)
- `Rainfall`: Rainfall intensity (mm/hour)

Alternative climate change forcing columns:
- `tas`, `rsds`, `sfcwind`, `hurs`, `rlds`, `pr`

## Usage

### Basic Usage

```python
import pandas as pd
import integrated_temperature_infiltration as itm

# Load your meteorological data
sim_df = pd.read_csv('your_forcing_data.csv', parse_dates=['date'])

# Define parameter file paths
temp_params = 'input_data/temperature_parameters.ini'
wf_params = 'input_data/water_flow_parameters.ini'

# Run the integrated model
results_df, water_flow_df = itm.run_integrated_model(
    sim_df=sim_df,
    temp_params_file=temp_params,
    wf_params_file=wf_params,
    rainfall_column='Rainfall',
    rainfall_unit='mm',
    climate_change=False
)

# Save results
results_df.to_csv('output/integrated_results.csv', index=False)
water_flow_df.to_csv('output/water_flow_results.csv', index=False)
```

### Running the Example

```bash
cd temperature_water_infiltration
python example_run_integrated_model.py
```

The example script will:
1. Check for required parameter files
2. Load or generate meteorological data
3. Run the integrated simulation
4. Save results to the output directory
5. Display summary statistics

## Model Integration Details

### How the Models are Coupled

1. **Water Flow State**: At each timestep, the water flow model computes:
   - Ponding depth (hp)
   - Saturation in unsaturated zone (s)
   - Saturated zone depth (hsz)
   - Unsaturated zone depth (husz)
   - Zone-specific porosities (nusz, nsz)

2. **Volumetric Water Content**: Computed for each temperature model node:
   ```
   θ = s * n  (for unsaturated zones)
   θ = n      (for saturated zones)
   ```

3. **Thermal Property Updates**: Material properties adjusted using:
   ```
   ρ_eff = (1-θ) * ρ_dry + θ * ρ_water
   (ρc)_eff = (1-θ) * ρ_dry * c_dry + θ * ρ_water * c_water
   λ_eff = λ_dry^(1-θ) * λ_water^θ
   ```

4. **Heat Transfer**: Temperature equation solved with updated properties:
   ```
   ρc * ∂T/∂t = ∇·(λ∇T) + Q_surface
   ```

5. **State Updates**: Both models advance forward in time, maintaining consistency

### Key Assumptions

1. **Spatial Alignment**: Temperature model grid is mapped to water flow zones
2. **Thermal Equilibrium**: Water and solid matrix are at the same temperature locally
3. **No Phase Change**: Water remains liquid (no freezing/melting)
4. **Quasi-steady Infiltration**: Water flow instantaneous compared to thermal timescales
5. **Geometric Mean Conductivity**: Used for moisture-dependent thermal conductivity

## Output Variables

### Integrated Results DataFrame

| Column | Description | Units |
|--------|-------------|-------|
| `date` | Timestamp | datetime |
| `h_s` | Solar radiation absorbed | W/m² |
| `h_li` | Incoming longwave radiation | W/m² |
| `h_l0` | Outgoing longwave radiation | W/m² |
| `h_rad` | Net radiation | W/m² |
| `h_evap` | Evaporative heat flux | W/m² |
| `h_conv` | Convective heat flux | W/m² |
| `h_r0` | Rainfall heat exchange | W/m² |
| `h_net` | Net heat flux to surface | W/m² |
| `surface_temp` | Surface temperature | °C |
| `subsurface_temp` | Average subsurface temperature | °C |
| `water_temp_surface` | Surface water temperature | °C |
| `water_temp_infil` | Infiltrating water temperature | °C |
| `theta_surface` | Surface volumetric water content | - |
| `theta_subsurface` | Subsurface volumetric water content | - |
| `ponding_depth` | Depth of ponded water | m |
| `saturation` | Unsaturated zone saturation | - |

### Water Flow Results DataFrame

| Column | Description | Units |
|--------|-------------|-------|
| `date` | Timestamp | datetime |
| `Qpipe` | Pipe/underdrain outflow | m³/s |
| `Qover` | Surface overflow | m³/s |
| `Qpf` | Percolation through filtration | m³/s |
| `Qfs` | Free storage flow | m³/s |
| `s` | Saturation degree | - |
| `hsz` | Saturated zone depth | m |
| `husz` | Unsaturated zone depth | m |
| `hp` | Ponding depth | m |

## Calibration and Validation

### Temperature Model
The temperature parameters in `temperature_parameters.ini` have been calibrated against observed pavement temperatures. Key calibrated parameters:
- `reflectivity`: Surface solar reflectance
- `emissivity`: Surface thermal emissivity
- `h_forced`: Forced convection coefficient
- `f_lam_layer1`: Thermal conductivity multiplier for surface layer

### Water Flow Model
The water flow parameters in `water_flow_parameters.ini` were calibrated using:
- Genetic algorithm optimization
- Observed outflow hydrographs
- Multiple rain events
- Particle filter validation

Source: GreenDT project (calibrated and validated model)

### Integration
The integrated model couples these validated components. For new sites or conditions:
1. Keep water flow parameters from calibration
2. Adjust temperature parameters if needed
3. Verify thermal property moisture dependencies against lab data if available

## Troubleshooting

### Common Issues

**Issue**: `ImportError: No module named 'temperature_model'`
- **Solution**: Ensure `temperature_model.py` exists in parent directory, or add it to Python path

**Issue**: `FileNotFoundError` for parameter files
- **Solution**: Check paths in `input_data/` directory, ensure both `.ini` files exist

**Issue**: Model produces unrealistic temperatures
- **Solution**:
  - Verify input data has correct units
  - Check initial temperature setting
  - Review calibration parameters

**Issue**: Water content exceeds porosity
- **Solution**:
  - Verify water flow parameter values
  - Check layer depth compatibility between models

**Issue**: Slow execution
- **Solution**:
  - Reduce spatial resolution (`dx` in temperature_parameters.ini)
  - Use shorter time series for testing
  - Consider vectorizing loops if modifying code

### Getting Help

1. Check parameter file formats match the expected structure
2. Verify input data has all required columns with correct names
3. Review example script for proper usage patterns
4. Check that timesteps (dt) are consistent between models

## References

### Temperature Model
- Original implementation in this project
- Based on Herb et al. (2008) for rainfall heat exchange
- Crank-Nicolson numerical scheme for stability

### Water Flow Model
- Adapted from GreenDT project: https://github.com/[repository]
- Based on bioretention/permeable pavement hydrology
- Multi-zone conceptual model with Richards-type equations

### Integration Approach
- Moisture-dependent thermal properties: De Vries (1963)
- Coupled heat-moisture transfer: Philip & De Vries (1957)
- Geometric mean thermal conductivity: Johansen (1975)

## Future Enhancements

Potential improvements to the integrated model:
1. **Latent heat of infiltration**: Explicit accounting for energy carried by water flux
2. **Vapor transport**: Include vapor diffusion in unsaturated zones
3. **Freezing/thawing**: Phase change effects in cold climates
4. **Root water uptake**: For vegetated permeable pavements
5. **2D/3D extension**: Lateral heat and water flow
6. **Optimization**: Vectorized operations for computational efficiency

## License

This integrated model combines components from:
- Pavement temperature model (this project)
- GreenDT water flow model (adapted)

Please refer to original project licenses for usage terms.

## Contact

For questions or issues related to this integration:
- Check this README and example scripts first
- Review parameter file documentation
- Verify all dependencies are correctly installed

---

*Last updated: 2025*
