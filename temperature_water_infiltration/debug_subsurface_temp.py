"""Debug script to check subsurface temperature values"""
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import integrated_temperature_infiltration as itm

# Load PA data - just first 500 timesteps for speed
pavement = "PA"
base_dir = os.path.dirname(os.path.abspath(__file__))
input_dir = os.path.join(base_dir, "input_data", pavement)

data_file = os.path.join(input_dir, f"input_data_{pavement}.csv")
temp_params_file = os.path.join(input_dir, f"temperature_parameters_{pavement}.ini")
wf_params_file = os.path.join(input_dir, f"water_flow_calibrated_parameters_{pavement}.ini")

print("Loading parameters to check grid spacing...")

# Load to check grid
temp_params, wf_params = itm.load_parameters(temp_params_file, wf_params_file)

total_depth = temp_params['pavement']['total_depth']
dx = temp_params['pavement']['dx']
N = int(np.ceil(total_depth / dx)) + 1
x = np.linspace(0, total_depth, N)

print(f"\nGrid Information:")
print(f"  Total depth: {total_depth} m")
print(f"  Grid spacing (dx): {dx} m")
print(f"  Number of nodes: {N}")
print(f"  Depth array: {x[:10]} ... {x[-3:]}")

# Check which nodes are "shallow"
shallow_limit_30cm = 0.30
shallow_limit_20cm = 0.20
shallow_limit_15cm = 0.15

idx_30 = np.where((x > 0) & (x <= shallow_limit_30cm))[0]
idx_20 = np.where((x > 0) & (x <= shallow_limit_20cm))[0]
idx_15 = np.where((x > 0) & (x <= shallow_limit_15cm))[0]

print(f"\nNodes in different depth ranges:")
print(f"  0-15cm: {len(idx_15)} nodes at depths {x[idx_15] if len(idx_15) > 0 else 'NONE'}")
print(f"  0-20cm: {len(idx_20)} nodes at depths {x[idx_20] if len(idx_20) > 0 else 'NONE'}")
print(f"  0-30cm: {len(idx_30)} nodes at depths {x[idx_30] if len(idx_30) > 0 else 'NONE'}")
print(f"  All subsurface (>0): {len(x[1:])} nodes")

# Load a small amount of data and run model
obs_df = pd.read_csv(data_file, parse_dates=['date'], nrows=1000)

print(f"\nRunning model on first 1000 timesteps...")
results_df, _ = itm.run_integrated_model(
    sim_df=obs_df,
    temp_params_file=temp_params_file,
    wf_params_file=wf_params_file,
    rainfall_column='Rainfall',
    rainfall_unit='mm',
    climate_change=False
)

# Merge and analyze
merged = pd.merge(
    obs_df[['date', 'WaterTemperature', 'PavementTemperature']],
    results_df[['date', 'water_temp_infil', 'surface_temp', 'subsurface_temp']],
    on='date',
    how='inner'
)
merged = merged[merged['WaterTemperature'].notna()]

if len(merged) > 0:
    print(f"\nTemperature Statistics (where water obs exist, n={len(merged)}):")
    print(f"  Observed water:   {merged['WaterTemperature'].mean():.2f}°C (range: {merged['WaterTemperature'].min():.1f}-{merged['WaterTemperature'].max():.1f})")
    print(f"  Modeled water:    {merged['water_temp_infil'].mean():.2f}°C (range: {merged['water_temp_infil'].min():.1f}-{merged['water_temp_infil'].max():.1f})")
    print(f"  Surface temp:     {merged['surface_temp'].mean():.2f}°C")
    print(f"  Subsurface temp:  {merged['subsurface_temp'].mean():.2f}°C")

    # The "subsurface_temp" in results is calculated differently - check what it is
    print(f"\n  Observed water vs surface: diff = {(merged['WaterTemperature'] - merged['surface_temp']).mean():.2f}°C")
    print(f"  Observed water vs subsurface: diff = {(merged['WaterTemperature'] - merged['subsurface_temp']).mean():.2f}°C")
