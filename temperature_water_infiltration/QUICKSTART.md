# Quick Start Guide: Integrated Temperature and Water Infiltration Model

## What Was Created

A new folder `temperature_water_infiltration/` containing an integrated model that combines:
- Your pavement temperature model with heat transfer
- The calibrated water infiltration model from GreenDT

This addresses your concerns about the original model not accounting for water volume and infiltration.

## Files Created

```
temperature_water_infiltration/
├── integrated_temperature_infiltration.py    # Main integrated model
├── water_flow_module.py                     # Water infiltration (from GreenDT)
├── example_run_integrated_model.py          # Example script
├── README.md                                # Full documentation
├── INTEGRATION_ANALYSIS.md                  # Technical details
├── QUICKSTART.md                            # This file
├── input_data/
│   ├── temperature_parameters.ini          # Temperature config
│   └── water_flow_parameters.ini           # Water flow config
└── output/                                  # Results go here
```

## Quick Test (5 minutes)

### 1. Navigate to the folder
```bash
cd C:\Users\Artur\PycharmProjects\pavement_temperature_model\temperature_water_infiltration
```

### 2. Run the example
```bash
python example_run_integrated_model.py
```

### 3. Check results
Results will be saved in `output/`:
- `integrated_results.csv` - Temperature, heat fluxes, moisture content
- `water_flow_results.csv` - Water flow details

## Using With Your Own Data

### Option 1: Use existing input data
```python
import pandas as pd
import integrated_temperature_infiltration as itm

# Load your forcing data
sim_df = pd.read_csv('../input_data/1h/input_data_PICP2.csv', parse_dates=['date'])

# Run model
results_df, water_df = itm.run_integrated_model(
    sim_df=sim_df,
    temp_params_file='input_data/temperature_parameters.ini',
    wf_params_file='input_data/water_flow_parameters.ini',
    rainfall_column='Rainfall',
    rainfall_unit='mm'
)

# Save results
results_df.to_csv('output/my_results.csv', index=False)
```

### Option 2: Custom data
Your CSV needs these columns:
- `date` - Timestamp
- `AirTemperature` - °C
- `SolarRadiation` - W/m²
- `WindSpeed` - m/s
- `RelativeHumidity` - fraction (0-1)
- `CloudCoverage` - fraction (0-1)
- `DewPoint` - °C
- `Rainfall` - mm/hour

Then use the same code as Option 1.

## Key Improvements Over Original Model

| Original Model | Integrated Model |
|----------------|------------------|
| Assumes water leaves immediately | Tracks water storage in layers |
| No infiltration | Multi-zone infiltration model |
| Fixed thermal properties | Properties vary with moisture |
| Simple water temperature | Detailed heat-moisture coupling |

## What to Check in Results

### Temperature Results
- `surface_temp` - Pavement surface temperature (°C)
- `subsurface_temp` - Average subsurface temperature (°C)
- `h_rad`, `h_conv`, `h_evap` - Heat flux components (W/m²)

### Moisture Results
- `theta_surface` - Volumetric water content at surface
- `saturation` - Degree of saturation in unsaturated zone
- `ponding_depth` - Water ponding on surface (m)

### Water Flow Results
- `Qpipe` - Underdrain outflow (m³/s)
- `hsz` - Saturated zone depth (m)
- `s` - Saturation degree (-)

## Troubleshooting

**Problem**: ImportError for temperature_model
**Solution**: The model needs access to the parent directory. The example script handles this automatically.

**Problem**: No output files
**Solution**: Check console for error messages. Verify parameter files exist in `input_data/`.

**Problem**: Unrealistic results
**Solution**:
1. Check your input data units
2. Verify parameter files match your pavement structure
3. Review initial temperature setting

## Next Steps

1. **Test**: Run the example script first
2. **Validate**: Compare surface temperatures to observations (if available)
3. **Apply**: Use with your actual forcing data
4. **Analyze**: Plot temperature vs. moisture relationships
5. **Customize**: Adjust parameters for your specific pavement if needed

## Documentation

- **README.md** - Complete user guide with detailed documentation
- **INTEGRATION_ANALYSIS.md** - Technical analysis, dependencies, equations
- **This file (QUICKSTART.md)** - Get started quickly

## Support

If you encounter issues:
1. Check that all parameter files exist
2. Verify your forcing data has required columns
3. Review the error message - it usually indicates the problem
4. Consult INTEGRATION_ANALYSIS.md for technical details

## Example Output

When you run the example, you should see:
```
================================================================================
INTEGRATED TEMPERATURE AND WATER INFILTRATION MODEL - EXAMPLE
================================================================================

Using parameter files:
  Temperature: ...\input_data\temperature_parameters.ini
  Water Flow:  ...\input_data\water_flow_parameters.ini

Loading meteorological data...
...

Running integrated model...
(This may take a few moments)
Model run completed successfully!

Saving results...
Temperature results saved to: ...\output\integrated_results.csv
Water flow results saved to:  ...\output\water_flow_results.csv

RESULTS SUMMARY
--------------------------------------------------------------------------------

Temperature Statistics:
  Surface Temperature:
    Mean:    XX.XX °C
    Min:     XX.XX °C
    Max:     XX.XX °C
    ...

SIMULATION COMPLETE
================================================================================
```

## Questions?

The integrated model is ready to use! All files are in place with correct paths configured.

Start with the example script to verify everything works, then apply it to your research data.
