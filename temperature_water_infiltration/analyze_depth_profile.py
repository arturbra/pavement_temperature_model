"""
Analyze which depth's temperature best matches observed water temperature
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# We need to modify the model to output temperature at each depth
# For now, let's analyze using the available outputs

pavement = "PA"
base_dir = os.path.dirname(os.path.abspath(__file__))
input_dir = os.path.join(base_dir, "input_data", pavement)

data_file = os.path.join(input_dir, f"input_data_{pavement}.csv")

# Load validation results that were already generated
output_dir = os.path.join(base_dir, "output", f"{pavement}_validation")
results_file = os.path.join(output_dir, "model_results.csv")

if os.path.exists(results_file):
    print("Loading existing results...")
    results_df = pd.read_csv(results_file, parse_dates=['date'])
    obs_df = pd.read_csv(data_file, parse_dates=['date'])

    merged = pd.merge(
        obs_df[['date', 'WaterTemperature', 'PavementTemperature']],
        results_df[['date', 'water_temp_infil', 'surface_temp', 'subsurface_temp']],
        on='date',
        how='inner'
    )
    merged = merged[merged['WaterTemperature'].notna()]

    print(f"\nAnalyzing {len(merged)} observations...")

    # Check correlations
    print(f"\nTemperature Correlations with Observed Water:")
    corr_surface = merged[['WaterTemperature', 'surface_temp']].corr().iloc[0, 1]
    corr_subsurface = merged[['WaterTemperature', 'subsurface_temp']].corr().iloc[0, 1]
    corr_modeled = merged[['WaterTemperature', 'water_temp_infil']].corr().iloc[0, 1]

    print(f"  Surface temp:     r = {corr_surface:.3f}")
    print(f"  Subsurface temp:  r = {corr_subsurface:.3f}")
    print(f"  Modeled water:    r = {corr_modeled:.3f}")

    # MAE to different references
    mae_surface = (merged['WaterTemperature'] - merged['surface_temp']).abs().mean()
    mae_subsurface = (merged['WaterTemperature'] - merged['subsurface_temp']).abs().mean()
    mae_modeled = (merged['WaterTemperature'] - merged['water_temp_infil']).abs().mean()

    print(f"\nMean Absolute Error:")
    print(f"  Surface temp:     {mae_surface:.2f}°C")
    print(f"  Subsurface temp:  {mae_subsurface:.2f}°C")
    print(f"  Modeled water:    {mae_modeled:.2f}°C")

    # Temperature statistics
    print(f"\nTemperature Statistics:")
    print(f"  Observed water:   {merged['WaterTemperature'].mean():.1f}°C ± {merged['WaterTemperature'].std():.1f}")
    print(f"  Surface temp:     {merged['surface_temp'].mean():.1f}°C ± {merged['surface_temp'].std():.1f}")
    print(f"  Subsurface temp:  {merged['subsurface_temp'].mean():.1f}°C ± {merged['subsurface_temp'].std():.1f}")
    print(f"  Modeled water:    {merged['water_temp_infil'].mean():.1f}°C ± {merged['water_temp_infil'].std():.1f}")

    # The key question: is subsurface_temp systematically too cold?
    print(f"\nBias Analysis:")
    print(f"  Observed water - Subsurface: {(merged['WaterTemperature'] - merged['subsurface_temp']).mean():.2f}°C")
    print(f"  Observed water - Surface:    {(merged['WaterTemperature'] - merged['surface_temp']).mean():.2f}°C")

    # Create diagnostic plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Observed water vs subsurface temp
    ax = axes[0, 0]
    ax.scatter(merged['subsurface_temp'], merged['WaterTemperature'], alpha=0.3, s=10)
    lims = [min(merged['subsurface_temp'].min(), merged['WaterTemperature'].min()),
            max(merged['subsurface_temp'].max(), merged['WaterTemperature'].max())]
    ax.plot(lims, lims, 'r--', lw=2, label='1:1 line')
    ax.set_xlabel('Subsurface Temperature (°C)')
    ax.set_ylabel('Observed Water Temperature (°C)')
    ax.set_title(f'Water vs Subsurface (r={corr_subsurface:.3f})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Observed water vs surface temp
    ax = axes[0, 1]
    ax.scatter(merged['surface_temp'], merged['WaterTemperature'], alpha=0.3, s=10)
    lims = [min(merged['surface_temp'].min(), merged['WaterTemperature'].min()),
            max(merged['surface_temp'].max(), merged['WaterTemperature'].max())]
    ax.plot(lims, lims, 'r--', lw=2, label='1:1 line')
    ax.set_xlabel('Surface Temperature (°C)')
    ax.set_ylabel('Observed Water Temperature (°C)')
    ax.set_title(f'Water vs Surface (r={corr_surface:.3f})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Time series sample
    ax = axes[1, 0]
    sample = merged.iloc[1000:1500]  # Sample period
    ax.plot(sample['date'], sample['WaterTemperature'], 'o-', label='Observed Water', markersize=3)
    ax.plot(sample['date'], sample['water_temp_infil'], 's--', label='Modeled Water', markersize=3)
    ax.plot(sample['date'], sample['subsurface_temp'], ':', label='Subsurface Temp', alpha=0.7)
    ax.set_ylabel('Temperature (°C)')
    ax.set_title('Sample Time Series')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    # Plot 4: Residuals vs subsurface temp
    ax = axes[1, 1]
    residuals = merged['water_temp_infil'] - merged['WaterTemperature']
    ax.scatter(merged['subsurface_temp'], residuals, alpha=0.3, s=10)
    ax.axhline(y=0, color='r', linestyle='--', lw=2)
    ax.set_xlabel('Subsurface Temperature (°C)')
    ax.set_ylabel('Residual (Modeled - Observed) (°C)')
    ax.set_title('Residuals vs Subsurface Temperature')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'depth_analysis.png'), dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {os.path.join(output_dir, 'depth_analysis.png')}")

else:
    print(f"Results file not found: {results_file}")
    print("Please run validate_PA_pavement.py first")
