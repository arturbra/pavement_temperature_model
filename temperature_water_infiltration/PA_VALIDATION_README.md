# PA Pavement Validation Guide

## Overview

This validation script (`validate_PA_pavement.py`) runs the integrated temperature and water infiltration model using actual data from PA pavement and compares the results against observed temperatures.

## Quick Start

### Run the Validation

```bash
cd C:\Users\Artur\PycharmProjects\pavement_temperature_model\temperature_water_infiltration
python validate_PA_pavement.py
```

## Input Files (Already in place)

### Data File
- **input_data/input_data_PA.csv** - Contains:
  - Meteorological forcing (AirTemperature, SolarRadiation, WindSpeed, etc.)
  - Observed PavementTemperature (surface)
  - Observed BottomTemperature (subsurface)
  - Observed WaterTemperature
  - Rainfall data
  - Timestep: 5 minutes (300 seconds)

### Parameter Files
- **input_data/temperature_parameters_PA.ini** - Temperature model parameters
  - dt = 300 seconds
  - PA-specific thermal properties

- **input_data/water_flow_calibrated_parameters_PA.ini** - Water flow parameters
  - dt = 300 seconds
  - Calibrated hydraulic parameters for PA pavement

## What the Script Does

### 1. Loads Data
- Reads observed data from input_data_PA.csv
- Checks data availability for each variable
- Reports data coverage and rainfall events

### 2. Runs Integrated Model
- Executes the coupled temperature-infiltration model
- Uses 5-minute timesteps matching the observations
- Simulates the entire data period

### 3. Calculates Performance Metrics

For each variable (Surface Temperature, Bottom Temperature, Water Temperature):
- **NSE** (Nash-Sutcliffe Efficiency): Overall model performance
- **RMSE** (Root Mean Square Error): Average prediction error
- **MAE** (Mean Absolute Error): Average absolute error
- **R²** (Coefficient of Determination): Correlation strength
- **Bias**: Systematic over/under-prediction
- **Percent Bias**: Relative systematic error

### 4. Creates Comparison Plots

**Surface Temperature:**
- Time series comparison (observed vs. modeled)
- Scatter plot with 1:1 line

**Bottom Temperature** (if available):
- Time series comparison
- Scatter plot with 1:1 line

**Water Temperature** (if available):
- Time series comparison
- Scatter plot with 1:1 line

**Residuals Analysis:**
- Residuals vs. time
- Residuals distribution (histogram)
- Residuals vs. observed values
- Residuals vs. modeled values

**Model Diagnostics:**
- Heat flux components evolution
- Moisture content (surface and subsurface)
- Saturation and ponding depth

### 5. Generates Reports
- CSV file with performance metrics
- Complete model results CSV
- Water flow results CSV
- Text validation report with interpretation

## Output Files

All results are saved to: `output/PA_validation/`

### Data Files
- **model_results.csv** - Full model output (temperature, heat fluxes, moisture)
- **water_flow_results.csv** - Water flow details (outflow, saturation, depths)
- **performance_metrics.csv** - Statistical comparison metrics

### Report
- **validation_report.txt** - Comprehensive validation report with:
  - Data summary
  - Performance metrics for all variables
  - Model interpretation guidelines

### Plots
- **surface_temperature_comparison.png** - Surface temperature validation
- **bottom_temperature_comparison.png** - Subsurface temperature validation (if data available)
- **water_temperature_comparison.png** - Water temperature validation (if data available)
- **residuals_analysis.png** - Error analysis (4 subplots)
- **heat_flux_components.png** - Energy balance components
- **moisture_content.png** - Water content evolution

## Interpreting Results

### NSE (Nash-Sutcliffe Efficiency)
- **NSE = 1.0**: Perfect match
- **NSE > 0.75**: Very good performance
- **NSE > 0.65**: Good performance
- **NSE > 0.50**: Satisfactory performance
- **NSE < 0.50**: Unsatisfactory performance

### RMSE (Root Mean Square Error)
- Lower is better
- In units of °C for temperature
- Compare to observed temperature range

### R² (Coefficient of Determination)
- **R² = 1.0**: Perfect correlation
- **R² > 0.85**: Very strong correlation
- **R² > 0.70**: Strong correlation
- **R² > 0.50**: Moderate correlation

### Bias
- **Positive**: Model over-predicts on average
- **Negative**: Model under-predicts on average
- **Near zero**: Unbiased predictions

## Expected Results

Based on the data structure:

### Surface Temperature (PavementTemperature)
- Should have good coverage (appears to be continuous)
- Expected NSE: > 0.7 (good models typically achieve this)
- Expected RMSE: 1-3°C (typical for calibrated models)

### Bottom Temperature (BottomTemperature)
- May have limited data points
- Subsurface temperatures are typically easier to predict (less dynamic)
- Expected NSE: > 0.6

### Water Temperature (WaterTemperature)
- May be sparse (only during rainfall events)
- More challenging to predict
- Evaluate separately during rain events

## Troubleshooting

### Script Runs Slowly
- The 5-minute timestep means many iterations
- For the full dataset, expect several minutes of runtime
- This is normal for high-resolution simulations

### Missing Plots
- Some plots only appear if sufficient data exists
- Bottom and water temperature plots require at least 10 observations
- Check console output for data availability

### Poor Performance
If metrics are unsatisfactory:
1. Check parameter values match PA pavement characteristics
2. Verify initial temperature is reasonable
3. Review forcing data for quality issues
4. Examine residuals plots for systematic patterns

## Next Steps After Validation

### If Results are Good (NSE > 0.65)
1. Use the model for scenario analysis
2. Apply to future climate scenarios
3. Evaluate drainage performance
4. Analyze heat-moisture interactions

### If Results Need Improvement
1. Review parameter values
2. Consider calibrating thermal parameters
3. Check for data quality issues
4. Examine residuals for systematic errors

## Code Structure

The validation script (`validate_PA_pavement.py`) includes:

### Functions

**calculate_performance_metrics(observed, modeled, variable_name)**
- Calculates NSE, RMSE, MAE, R², bias
- Handles missing data (NaN values)
- Returns comprehensive statistics

**create_comparison_plots(results_df, obs_df, output_dir)**
- Generates all comparison plots
- Handles missing data gracefully
- Saves high-resolution PNG files (300 DPI)

**main()**
- Orchestrates the entire validation workflow
- Loads data and parameters
- Runs model
- Calculates metrics
- Creates plots
- Generates reports

## Customization

### Plotting Specific Time Periods

To focus on a specific event or period, edit the script:

```python
# After loading obs_df, filter to desired period
start_date = '2023-07-14 12:00:00'
end_date = '2023-07-15 12:00:00'
obs_df = obs_df[(obs_df['date'] >= start_date) & (obs_df['date'] <= end_date)]
```

### Additional Metrics

Add custom metrics in the `calculate_performance_metrics` function:

```python
# Example: Peak temperature error
peak_error = abs(obs.max() - mod.max())
```

### Custom Plots

Add to `create_comparison_plots` function:

```python
# Example: Temperature difference plot
fig, ax = plt.subplots(figsize=(12, 6))
temp_diff = results_df['surface_temp'] - obs_df['PavementTemperature']
ax.plot(obs_df['date'], temp_diff)
ax.set_title('Temperature Difference (Modeled - Observed)')
plt.savefig(os.path.join(output_dir, 'temperature_difference.png'), dpi=300)
```

## Contact

If validation reveals issues:
1. Check the validation report for detailed statistics
2. Examine residuals plots for patterns
3. Review parameter files for PA pavement
4. Verify data quality in input_data_PA.csv

---

*Ready to run! Just execute: `python validate_PA_pavement.py`*
