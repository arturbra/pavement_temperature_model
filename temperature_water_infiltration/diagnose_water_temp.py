"""
Diagnostic script to understand water temperature errors during different conditions
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import integrated_temperature_infiltration as itm

# Load PA data
base_dir = os.path.dirname(os.path.abspath(__file__))
pavement = "PA"
input_dir = os.path.join(base_dir, "input_data", pavement)

data_file = os.path.join(input_dir, f"input_data_{pavement}.csv")
temp_params_file = os.path.join(input_dir, f"temperature_parameters_{pavement}.ini")
wf_params_file = os.path.join(input_dir, f"water_flow_calibrated_parameters_{pavement}.ini")

print("Loading data and running model...")
obs_df = pd.read_csv(data_file, parse_dates=['date'])

results_df, water_flow_df = itm.run_integrated_model(
    sim_df=obs_df,
    temp_params_file=temp_params_file,
    wf_params_file=wf_params_file,
    rainfall_column='Rainfall',
    rainfall_unit='mm',
    climate_change=False
)

# Merge observed and modeled
merged = pd.merge(
    obs_df[['date', 'Rainfall', 'WaterTemperature', 'PavementTemperature']],
    results_df[['date', 'water_temp_infil', 'surface_temp', 'subsurface_temp']],
    on='date',
    how='inner'
)

# Only analyze where we have observations
merged = merged[merged['WaterTemperature'].notna()].copy()

# Calculate errors
merged['error'] = merged['water_temp_infil'] - merged['WaterTemperature']
merged['abs_error'] = merged['error'].abs()

# Classify timesteps
merged['has_rain'] = merged['Rainfall'] > 0
merged['hours_since_rain'] = 0.0

# Calculate hours since last rain
last_rain_idx = -1000
for i in range(len(merged)):
    if merged['Rainfall'].iloc[i] > 0:
        last_rain_idx = i
        merged.loc[merged.index[i], 'hours_since_rain'] = 0
    else:
        if last_rain_idx >= 0:
            time_diff = (merged['date'].iloc[i] - merged['date'].iloc[last_rain_idx]).total_seconds() / 3600
            merged.loc[merged.index[i], 'hours_since_rain'] = time_diff
        else:
            merged.loc[merged.index[i], 'hours_since_rain'] = 999

print("\n" + "="*80)
print("DIAGNOSTIC ANALYSIS: Water Temperature Errors")
print("="*80)

# Overall statistics
print(f"\nOverall Statistics:")
print(f"  Total observations: {len(merged)}")
print(f"  RMSE: {np.sqrt((merged['error']**2).mean()):.3f}°C")
print(f"  MAE: {merged['abs_error'].mean():.3f}°C")
print(f"  Bias: {merged['error'].mean():.3f}°C")

# During active rainfall
rain_data = merged[merged['has_rain']]
if len(rain_data) > 0:
    print(f"\nDuring Active Rainfall (n={len(rain_data)}):")
    print(f"  RMSE: {np.sqrt((rain_data['error']**2).mean()):.3f}°C")
    print(f"  MAE: {rain_data['abs_error'].mean():.3f}°C")
    print(f"  Bias: {rain_data['error'].mean():.3f}°C")

# After rainfall (within 6 hours)
recent_data = merged[(~merged['has_rain']) & (merged['hours_since_rain'] <= 6)]
if len(recent_data) > 0:
    print(f"\nWithin 6 Hours After Rain (n={len(recent_data)}):")
    print(f"  RMSE: {np.sqrt((recent_data['error']**2).mean()):.3f}°C")
    print(f"  MAE: {recent_data['abs_error'].mean():.3f}°C")
    print(f"  Bias: {recent_data['error'].mean():.3f}°C")

# Long after rainfall (>24 hours)
old_data = merged[(~merged['has_rain']) & (merged['hours_since_rain'] > 24)]
if len(old_data) > 0:
    print(f"\n>24 Hours After Rain (n={len(old_data)}):")
    print(f"  RMSE: {np.sqrt((old_data['error']**2).mean()):.3f}°C")
    print(f"  MAE: {old_data['abs_error'].mean():.3f}°C")
    print(f"  Bias: {old_data['error'].mean():.3f}°C")

# Check correlation with surface/subsurface temps
print("\n" + "="*80)
print("TEMPERATURE RELATIONSHIPS")
print("="*80)

print(f"\nObserved Water Temperature:")
print(f"  Range: {merged['WaterTemperature'].min():.1f} to {merged['WaterTemperature'].max():.1f}°C")
print(f"  Mean: {merged['WaterTemperature'].mean():.1f}°C")

print(f"\nModeled Water Temperature:")
print(f"  Range: {merged['water_temp_infil'].min():.1f} to {merged['water_temp_infil'].max():.1f}°C")
print(f"  Mean: {merged['water_temp_infil'].mean():.1f}°C")

print(f"\nPavement Surface Temperature (at water obs times):")
print(f"  Range: {merged['surface_temp'].min():.1f} to {merged['surface_temp'].max():.1f}°C")
print(f"  Mean: {merged['surface_temp'].mean():.1f}°C")

print(f"\nSubsurface Temperature (at water obs times):")
print(f"  Range: {merged['subsurface_temp'].min():.1f} to {merged['subsurface_temp'].max():.1f}°C")
print(f"  Mean: {merged['subsurface_temp'].mean():.1f}°C")

# Check which temperature observed water is closer to
diff_to_surface = (merged['WaterTemperature'] - merged['surface_temp']).abs().mean()
diff_to_subsurface = (merged['WaterTemperature'] - merged['subsurface_temp']).abs().mean()

print(f"\nObserved water temperature is closer to:")
print(f"  Surface temp: MAE = {diff_to_surface:.2f}°C")
print(f"  Subsurface temp: MAE = {diff_to_subsurface:.2f}°C")
if diff_to_surface < diff_to_subsurface:
    print(f"  → Observed water is {diff_to_subsurface - diff_to_surface:.2f}°C CLOSER to surface!")
else:
    print(f"  → Observed water is {diff_to_surface - diff_to_subsurface:.2f}°C CLOSER to subsurface!")

# Create diagnostic plot
print("\n" + "="*80)
print("Creating diagnostic plots...")
print("="*80)

fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# Plot 1: Time series during first rain event
rain_events = merged[merged['has_rain']]
if len(rain_events) > 0:
    first_rain_date = rain_events['date'].iloc[0]
    event_start = first_rain_date - pd.Timedelta(hours=12)
    event_end = first_rain_date + pd.Timedelta(hours=48)

    event_data = merged[(merged['date'] >= event_start) & (merged['date'] <= event_end)]

    ax = axes[0]
    ax.plot(event_data['date'], event_data['WaterTemperature'],
            'o-', label='Observed Water', color='blue', linewidth=2, markersize=4)
    ax.plot(event_data['date'], event_data['water_temp_infil'],
            's--', label='Modeled Water', color='red', linewidth=2, markersize=4)
    ax.plot(event_data['date'], event_data['surface_temp'],
            ':', label='Surface Temp', color='orange', linewidth=1, alpha=0.7)
    ax.plot(event_data['date'], event_data['subsurface_temp'],
            ':', label='Subsurface Temp', color='green', linewidth=1, alpha=0.7)

    ax.set_ylabel('Temperature (°C)', fontsize=12)
    ax.set_title('Water Temperature During First Rain Event', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

# Plot 2: Error vs surface temperature
ax = axes[1]
ax.scatter(merged['surface_temp'], merged['error'], alpha=0.5, s=20)
ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
ax.set_xlabel('Surface Temperature (°C)', fontsize=12)
ax.set_ylabel('Error (Modeled - Observed) (°C)', fontsize=12)
ax.set_title('Water Temperature Error vs Surface Temperature', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)

# Plot 3: Error vs subsurface temperature
ax = axes[2]
ax.scatter(merged['subsurface_temp'], merged['error'], alpha=0.5, s=20, color='green')
ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
ax.set_xlabel('Subsurface Temperature (°C)', fontsize=12)
ax.set_ylabel('Error (Modeled - Observed) (°C)', fontsize=12)
ax.set_title('Water Temperature Error vs Subsurface Temperature', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)

plt.tight_layout()
output_dir = os.path.join(base_dir, "output", f"{pavement}_validation")
plt.savefig(os.path.join(output_dir, "water_temp_diagnostics.png"), dpi=300, bbox_inches='tight')
print(f"  Saved: {os.path.join(output_dir, 'water_temp_diagnostics.png')}")

plt.close()

print("\n" + "="*80)
print("DIAGNOSIS COMPLETE")
print("="*80)
