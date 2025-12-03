"""
Validation Script for PA Pavement - Integrated Temperature and Water Infiltration Model

This script:
1. Loads observed data from input_data_PA.csv
2. Runs the integrated model with PA-specific parameters
3. Compares modeled vs. observed temperatures (surface, bottom, water)
4. Calculates performance metrics (NSE, RMSE, MAE, R², bias)
5. Creates comparison plots
6. Generates a validation report

Author: Validation for PA Pavement
Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime

# Add parent directory to path to import temperature_model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import integrated_temperature_infiltration as itm


def calculate_performance_metrics(observed, modeled, variable_name="Variable"):
    """
    Calculate comprehensive performance metrics.

    Args:
        observed: Array of observed values
        modeled: Array of modeled values
        variable_name: Name of the variable for reporting

    Returns:
        Dictionary with performance metrics
    """
    # Remove NaN values (where observations don't exist)
    mask = ~np.isnan(observed) & ~np.isnan(modeled)
    obs = observed[mask]
    mod = modeled[mask]

    if len(obs) == 0:
        return {
            'variable': variable_name,
            'n_points': 0,
            'nse': np.nan,
            'rmse': np.nan,
            'mae': np.nan,
            'r2': np.nan,
            'bias': np.nan,
            'percent_bias': np.nan
        }

    # Nash-Sutcliffe Efficiency
    nse = 1 - np.sum((obs - mod) ** 2) / np.sum((obs - np.mean(obs)) ** 2)

    # Root Mean Square Error
    rmse = np.sqrt(np.mean((obs - mod) ** 2))

    # Mean Absolute Error
    mae = np.mean(np.abs(obs - mod))

    # R-squared (coefficient of determination)
    corr_matrix = np.corrcoef(obs, mod)
    r2 = corr_matrix[0, 1] ** 2 if corr_matrix.shape == (2, 2) else np.nan

    # Bias (mean error)
    bias = np.mean(mod - obs)

    # Percent bias
    percent_bias = 100 * np.sum(mod - obs) / np.sum(obs) if np.sum(obs) != 0 else np.nan

    return {
        'variable': variable_name,
        'n_points': len(obs),
        'nse': nse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'bias': bias,
        'percent_bias': percent_bias,
        'obs_mean': np.mean(obs),
        'obs_std': np.std(obs),
        'mod_mean': np.mean(mod),
        'mod_std': np.std(mod),
        'obs_min': np.min(obs),
        'obs_max': np.max(obs),
        'mod_min': np.min(mod),
        'mod_max': np.max(mod)
    }


def create_comparison_plots(results_df, obs_df, output_dir):
    """
    Create comprehensive comparison plots.

    Args:
        results_df: DataFrame with model results
        obs_df: DataFrame with observations
        output_dir: Directory to save plots
    """

    # Create figure directory
    os.makedirs(output_dir, exist_ok=True)

    # === 1. Surface Temperature Time Series ===
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Full time series
    ax = axes[0]
    ax.plot(obs_df['date'], obs_df['PavementTemperature'],
            'o', markersize=2, alpha=0.5, label='Observed', color='black')
    ax.plot(results_df['date'], results_df['surface_temp'],
            '-', linewidth=1.5, label='Modeled', color='red')
    ax.set_ylabel('Surface Temperature (°C)', fontsize=12)
    ax.set_title('Surface Temperature: Observed vs. Modeled', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Scatter plot
    ax = axes[1]
    mask = ~np.isnan(obs_df['PavementTemperature'])
    ax.scatter(obs_df.loc[mask, 'PavementTemperature'],
              results_df.loc[mask, 'surface_temp'],
              alpha=0.3, s=10, color='blue')

    # 1:1 line
    min_val = min(obs_df.loc[mask, 'PavementTemperature'].min(),
                  results_df.loc[mask, 'surface_temp'].min())
    max_val = max(obs_df.loc[mask, 'PavementTemperature'].max(),
                  results_df.loc[mask, 'surface_temp'].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='1:1 Line')

    ax.set_xlabel('Observed Temperature (°C)', fontsize=12)
    ax.set_ylabel('Modeled Temperature (°C)', fontsize=12)
    ax.set_title('Surface Temperature: Scatter Plot', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'surface_temperature_comparison.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    # === 2. Bottom Temperature (if available) ===
    if obs_df['BottomTemperature'].notna().sum() > 10:
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        # Time series
        ax = axes[0]
        mask_bottom = ~np.isnan(obs_df['BottomTemperature'])
        ax.plot(obs_df.loc[mask_bottom, 'date'],
                obs_df.loc[mask_bottom, 'BottomTemperature'],
                'o', markersize=3, alpha=0.5, label='Observed', color='black')
        ax.plot(results_df.loc[mask_bottom, 'date'],
                results_df.loc[mask_bottom, 'subsurface_temp'],
                '-', linewidth=1.5, label='Modeled', color='green')
        ax.set_ylabel('Bottom Temperature (°C)', fontsize=12)
        ax.set_title('Bottom Temperature: Observed vs. Modeled', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

        # Scatter plot
        ax = axes[1]
        ax.scatter(obs_df.loc[mask_bottom, 'BottomTemperature'],
                  results_df.loc[mask_bottom, 'subsurface_temp'],
                  alpha=0.3, s=10, color='green')

        # 1:1 line
        min_val = min(obs_df.loc[mask_bottom, 'BottomTemperature'].min(),
                      results_df.loc[mask_bottom, 'subsurface_temp'].min())
        max_val = max(obs_df.loc[mask_bottom, 'BottomTemperature'].max(),
                      results_df.loc[mask_bottom, 'subsurface_temp'].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='1:1 Line')

        ax.set_xlabel('Observed Temperature (°C)', fontsize=12)
        ax.set_ylabel('Modeled Temperature (°C)', fontsize=12)
        ax.set_title('Bottom Temperature: Scatter Plot', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'bottom_temperature_comparison.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    # === 3. Water Temperature (if available) ===
    if obs_df['WaterTemperature'].notna().sum() > 10:
        fig, axes = plt.subplots(3, 1, figsize=(14, 14))

        # Time series - Infiltrated water (main output)
        ax = axes[0]
        mask_water = ~np.isnan(obs_df['WaterTemperature'])
        ax.plot(obs_df.loc[mask_water, 'date'],
                obs_df.loc[mask_water, 'WaterTemperature'],
                'o', markersize=3, alpha=0.5, label='Observed', color='black')
        ax.plot(results_df.loc[mask_water, 'date'],
                results_df.loc[mask_water, 'water_temp_infil'],
                '-', linewidth=1.5, label='Modeled (Infiltrated)', color='blue')
        ax.set_ylabel('Water Temperature (°C)', fontsize=12)
        ax.set_title('Water Temperature (Infiltrated): Observed vs. Modeled', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

        # Comparison: Surface vs. Infiltrated water temperature
        ax = axes[1]
        ax.plot(results_df.loc[mask_water, 'date'],
                results_df.loc[mask_water, 'water_temp_surface'],
                '-', linewidth=1.5, label='Surface Water', color='red', alpha=0.7)
        ax.plot(results_df.loc[mask_water, 'date'],
                results_df.loc[mask_water, 'water_temp_infil'],
                '-', linewidth=1.5, label='Infiltrated Water', color='blue', alpha=0.7)
        ax.plot(obs_df.loc[mask_water, 'date'],
                obs_df.loc[mask_water, 'WaterTemperature'],
                'o', markersize=3, alpha=0.5, label='Observed', color='black')
        ax.set_ylabel('Water Temperature (°C)', fontsize=12)
        ax.set_title('Water Temperature Evolution: Surface vs. Infiltrated', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

        # Scatter plot - Infiltrated water
        ax = axes[2]
        mask_valid = mask_water & ~np.isnan(results_df['water_temp_infil'])
        ax.scatter(obs_df.loc[mask_valid, 'WaterTemperature'],
                  results_df.loc[mask_valid, 'water_temp_infil'],
                  alpha=0.3, s=10, color='blue', label='Infiltrated')

        # 1:1 line
        if mask_valid.sum() > 0:
            min_val = min(obs_df.loc[mask_valid, 'WaterTemperature'].min(),
                          results_df.loc[mask_valid, 'water_temp_infil'].min())
            max_val = max(obs_df.loc[mask_valid, 'WaterTemperature'].max(),
                          results_df.loc[mask_valid, 'water_temp_infil'].max())
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='1:1 Line')

        ax.set_xlabel('Observed Temperature (°C)', fontsize=12)
        ax.set_ylabel('Modeled Temperature (°C)', fontsize=12)
        ax.set_title('Water Temperature (Infiltrated): Scatter Plot', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'water_temperature_comparison.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    # === 4. Residuals Analysis ===
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Surface temperature residuals vs. time
    ax = axes[0, 0]
    mask = ~np.isnan(obs_df['PavementTemperature'])
    residuals = results_df.loc[mask, 'surface_temp'] - obs_df.loc[mask, 'PavementTemperature']
    ax.plot(obs_df.loc[mask, 'date'], residuals, 'o', markersize=2, alpha=0.5)
    ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax.axhline(y=residuals.mean(), color='g', linestyle='--', linewidth=1, label=f'Mean: {residuals.mean():.2f}°C')
    ax.set_ylabel('Residual (Modeled - Observed) (°C)', fontsize=10)
    ax.set_title('Surface Temperature Residuals vs. Time', fontsize=11, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Residuals histogram
    ax = axes[0, 1]
    ax.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax.axvline(x=residuals.mean(), color='g', linestyle='--', linewidth=1, label=f'Mean: {residuals.mean():.2f}°C')
    ax.set_xlabel('Residual (°C)', fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)
    ax.set_title('Surface Temperature Residuals Distribution', fontsize=11, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Residuals vs. observed
    ax = axes[1, 0]
    ax.scatter(obs_df.loc[mask, 'PavementTemperature'], residuals, alpha=0.3, s=10)
    ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax.set_xlabel('Observed Temperature (°C)', fontsize=10)
    ax.set_ylabel('Residual (°C)', fontsize=10)
    ax.set_title('Residuals vs. Observed Temperature', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Residuals vs. modeled
    ax = axes[1, 1]
    ax.scatter(results_df.loc[mask, 'surface_temp'], residuals, alpha=0.3, s=10)
    ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax.set_xlabel('Modeled Temperature (°C)', fontsize=10)
    ax.set_ylabel('Residual (°C)', fontsize=10)
    ax.set_title('Residuals vs. Modeled Temperature', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'residuals_analysis.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    # === 5. Heat Flux Components ===
    fig, ax = plt.subplots(figsize=(14, 6))

    # Plot heat flux components
    ax.plot(results_df['date'], results_df['h_rad'], label='Net Radiation', linewidth=1)
    ax.plot(results_df['date'], results_df['h_conv'], label='Convection', linewidth=1)
    ax.plot(results_df['date'], results_df['h_evap'], label='Evaporation', linewidth=1)
    ax.plot(results_df['date'], results_df['h_r0'], label='Rainfall (Surface)', linewidth=1)
    ax.plot(results_df['date'], results_df['h_infil'], label='Infiltration (Subsurface)', linewidth=1)
    ax.plot(results_df['date'], results_df['h_net'], label='Net Heat Flux', linewidth=2, color='black')

    ax.set_ylabel('Heat Flux (W/m²)', fontsize=12)
    ax.set_title('Heat Flux Components (Including Infiltration)', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'heat_flux_components.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    # === 6. Moisture Content ===
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Moisture content
    ax = axes[0]
    ax.plot(results_df['date'], results_df['theta_surface'], label='Surface θ', linewidth=1.5)
    ax.plot(results_df['date'], results_df['theta_subsurface'], label='Subsurface θ', linewidth=1.5)
    ax.set_ylabel('Volumetric Water Content (-)', fontsize=12)
    ax.set_title('Moisture Content Evolution', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Saturation and ponding
    ax = axes[1]
    ax.plot(results_df['date'], results_df['saturation'], label='Saturation', linewidth=1.5, color='blue')
    ax.set_ylabel('Saturation (-)', fontsize=12, color='blue')
    ax.tick_params(axis='y', labelcolor='blue')
    ax.set_title('Saturation and Ponding Depth', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Secondary axis for ponding
    ax2 = ax.twinx()
    ax2.plot(results_df['date'], results_df['ponding_depth'] * 1000,
             label='Ponding Depth', linewidth=1.5, color='red')
    ax2.set_ylabel('Ponding Depth (mm)', fontsize=12, color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    # Combine legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'moisture_content.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Plots saved to: {output_dir}")


def main(pavement):
    """Main validation function."""

    print("=" * 80)
    print("PA PAVEMENT VALIDATION - INTEGRATED MODEL")
    print("=" * 80)

    start_time = datetime.now()

    # Define paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(base_dir, "input_data", pavement)
    output_dir = os.path.join(base_dir, "output", f"{pavement}_validation")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Input files
    data_file = os.path.join(input_dir, f"input_data_{pavement}.csv")
    temp_params_file = os.path.join(input_dir, f"temperature_parameters_{pavement}.ini")
    wf_params_file = os.path.join(input_dir, f"water_flow_calibrated_parameters_{pavement}.ini")

    print(f"\nInput Files:")
    print(f"  Data:        {data_file}")
    print(f"  Temperature: {temp_params_file}")
    print(f"  Water Flow:  {wf_params_file}")

    # Check files exist
    for file_path in [data_file, temp_params_file, wf_params_file]:
        if not os.path.exists(file_path):
            print(f"\nERROR: File not found: {file_path}")
            return

    # Load observed data
    print("\n" + "-" * 80)
    print("Loading observed data...")

    obs_df = pd.read_csv(data_file, parse_dates=['date'])
    print(f"  Loaded {len(obs_df)} timesteps")
    print(f"  Period: {obs_df['date'].iloc[0]} to {obs_df['date'].iloc[-1]}")
    print(f"  Duration: {obs_df['date'].iloc[-1] - obs_df['date'].iloc[0]}")

    # Data availability
    print(f"\n  Data Availability:")
    print(f"    PavementTemperature: {obs_df['PavementTemperature'].notna().sum()} / {len(obs_df)} ({100*obs_df['PavementTemperature'].notna().sum()/len(obs_df):.1f}%)")
    print(f"    BottomTemperature:   {obs_df['BottomTemperature'].notna().sum()} / {len(obs_df)} ({100*obs_df['BottomTemperature'].notna().sum()/len(obs_df):.1f}%)")
    print(f"    WaterTemperature:    {obs_df['WaterTemperature'].notna().sum()} / {len(obs_df)} ({100*obs_df['WaterTemperature'].notna().sum()/len(obs_df):.1f}%)")
    print(f"    Rainfall events:     {(obs_df['Rainfall'] > 0).sum()} timesteps")
    print(f"    Total rainfall:      {obs_df['Rainfall'].sum():.1f} mm")

    # Run the integrated model
    print("\n" + "-" * 80)
    print("Running integrated model...")
    print("(This may take several minutes for 5-minute timesteps)")

    try:
        results_df, water_flow_df = itm.run_integrated_model(
            sim_df=obs_df,
            temp_params_file=temp_params_file,
            wf_params_file=wf_params_file,
            rainfall_column='Rainfall',
            rainfall_unit='mm',
            climate_change=False
        )

        print("  Model run completed successfully!")

    except Exception as e:
        print(f"\nERROR during model execution:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return

    # Calculate performance metrics
    print("\n" + "-" * 80)
    print("Calculating performance metrics...")

    metrics = []

    # Surface temperature
    surf_metrics = calculate_performance_metrics(
        obs_df['PavementTemperature'].values,
        results_df['surface_temp'].values,
        "Surface Temperature"
    )
    metrics.append(surf_metrics)

    # Bottom temperature
    bottom_metrics = calculate_performance_metrics(
        obs_df['BottomTemperature'].values,
        results_df['subsurface_temp'].values,
        "Bottom Temperature"
    )
    metrics.append(bottom_metrics)

    # Water temperature - Use infiltrated water temperature (the main output)
    water_metrics = calculate_performance_metrics(
        obs_df['WaterTemperature'].values,
        results_df['water_temp_infil'].values,
        "Water Temperature (Infiltrated)"
    )
    metrics.append(water_metrics)

    # Also check surface water temperature for completeness
    water_surface_metrics = calculate_performance_metrics(
        obs_df['WaterTemperature'].values,
        results_df['water_temp_surface'].values,
        "Water Temperature (Surface)"
    )
    metrics.append(water_surface_metrics)

    # Display metrics
    print("\n" + "=" * 80)
    print("PERFORMANCE METRICS")
    print("=" * 80)

    for m in metrics:
        if m['n_points'] > 0:
            print(f"\n{m['variable']}:")
            print(f"  Number of points: {m['n_points']}")
            print(f"  NSE:              {m['nse']:.4f}")
            print(f"  RMSE:             {m['rmse']:.4f} °C")
            print(f"  MAE:              {m['mae']:.4f} °C")
            print(f"  R²:               {m['r2']:.4f}")
            print(f"  Bias:             {m['bias']:.4f} °C")
            print(f"  Percent Bias:     {m['percent_bias']:.2f} %")
            print(f"\n  Observed:  Mean={m['obs_mean']:.2f}°C, Std={m['obs_std']:.2f}°C, Range=[{m['obs_min']:.2f}, {m['obs_max']:.2f}]°C")
            print(f"  Modeled:   Mean={m['mod_mean']:.2f}°C, Std={m['mod_std']:.2f}°C, Range=[{m['mod_min']:.2f}, {m['mod_max']:.2f}]°C")
        else:
            print(f"\n{m['variable']}: No data available for comparison")

    # Save metrics to CSV
    metrics_df = pd.DataFrame(metrics)
    metrics_file = os.path.join(output_dir, "performance_metrics.csv")
    metrics_df.to_csv(metrics_file, index=False)
    print(f"\n  Metrics saved to: {metrics_file}")

    # Save full results
    print("\n" + "-" * 80)
    print("Saving results...")

    results_file = os.path.join(output_dir, "model_results.csv")
    results_df.to_csv(results_file, index=False)
    print(f"  Model results saved to: {results_file}")

    water_flow_file = os.path.join(output_dir, "water_flow_results.csv")
    water_flow_df.to_csv(water_flow_file, index=False)
    print(f"  Water flow results saved to: {water_flow_file}")

    # Create comparison plots
    print("\n" + "-" * 80)
    print("Creating comparison plots...")

    create_comparison_plots(results_df, obs_df, output_dir)

    # Generate validation report
    print("\n" + "-" * 80)
    print("Generating validation report...")

    report_file = os.path.join(output_dir, "validation_report.txt")
    with open(report_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("PA PAVEMENT VALIDATION REPORT\n")
        f.write("Integrated Temperature and Water Infiltration Model\n")
        f.write("=" * 80 + "\n")
        f.write(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"\nData Period: {obs_df['date'].iloc[0]} to {obs_df['date'].iloc[-1]}\n")
        f.write(f"Number of Timesteps: {len(obs_df)}\n")
        f.write(f"Timestep: 5 minutes (300 seconds)\n")

        f.write("\n" + "-" * 80 + "\n")
        f.write("DATA AVAILABILITY\n")
        f.write("-" * 80 + "\n")
        f.write(f"PavementTemperature: {obs_df['PavementTemperature'].notna().sum()} / {len(obs_df)} ({100*obs_df['PavementTemperature'].notna().sum()/len(obs_df):.1f}%)\n")
        f.write(f"BottomTemperature:   {obs_df['BottomTemperature'].notna().sum()} / {len(obs_df)} ({100*obs_df['BottomTemperature'].notna().sum()/len(obs_df):.1f}%)\n")
        f.write(f"WaterTemperature:    {obs_df['WaterTemperature'].notna().sum()} / {len(obs_df)} ({100*obs_df['WaterTemperature'].notna().sum()/len(obs_df):.1f}%)\n")

        f.write("\n" + "-" * 80 + "\n")
        f.write("PERFORMANCE METRICS\n")
        f.write("-" * 80 + "\n")

        for m in metrics:
            if m['n_points'] > 0:
                f.write(f"\n{m['variable']}:\n")
                f.write(f"  Number of points: {m['n_points']}\n")
                f.write(f"  NSE:              {m['nse']:.4f}\n")
                f.write(f"  RMSE:             {m['rmse']:.4f} °C\n")
                f.write(f"  MAE:              {m['mae']:.4f} °C\n")
                f.write(f"  R²:               {m['r2']:.4f}\n")
                f.write(f"  Bias:             {m['bias']:.4f} °C\n")
                f.write(f"  Percent Bias:     {m['percent_bias']:.2f} %\n")
                f.write(f"\n  Observed Statistics:\n")
                f.write(f"    Mean: {m['obs_mean']:.2f} °C\n")
                f.write(f"    Std:  {m['obs_std']:.2f} °C\n")
                f.write(f"    Min:  {m['obs_min']:.2f} °C\n")
                f.write(f"    Max:  {m['obs_max']:.2f} °C\n")
                f.write(f"\n  Modeled Statistics:\n")
                f.write(f"    Mean: {m['mod_mean']:.2f} °C\n")
                f.write(f"    Std:  {m['mod_std']:.2f} °C\n")
                f.write(f"    Min:  {m['mod_min']:.2f} °C\n")
                f.write(f"    Max:  {m['mod_max']:.2f} °C\n")

        f.write("\n" + "-" * 80 + "\n")
        f.write("MODEL INTERPRETATION\n")
        f.write("-" * 80 + "\n")
        f.write("\nNSE (Nash-Sutcliffe Efficiency):\n")
        f.write("  NSE = 1:    Perfect match\n")
        f.write("  NSE > 0.75: Very good\n")
        f.write("  NSE > 0.65: Good\n")
        f.write("  NSE > 0.50: Satisfactory\n")
        f.write("  NSE < 0.50: Unsatisfactory\n")

        f.write("\nR² (Coefficient of Determination):\n")
        f.write("  R² = 1:    Perfect correlation\n")
        f.write("  R² > 0.85: Very strong correlation\n")
        f.write("  R² > 0.70: Strong correlation\n")
        f.write("  R² > 0.50: Moderate correlation\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")

    print(f"  Validation report saved to: {report_file}")

    # Summary
    elapsed_time = datetime.now() - start_time

    print("\n" + "=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)
    print(f"\nElapsed time: {elapsed_time}")
    print(f"\nAll results saved to: {output_dir}")
    print("\nFiles created:")
    print(f"  - model_results.csv")
    print(f"  - water_flow_results.csv")
    print(f"  - performance_metrics.csv")
    print(f"  - validation_report.txt")
    print(f"  - surface_temperature_comparison.png")
    if obs_df['BottomTemperature'].notna().sum() > 10:
        print(f"  - bottom_temperature_comparison.png")
    if obs_df['WaterTemperature'].notna().sum() > 10:
        print(f"  - water_temperature_comparison.png")
    print(f"  - residuals_analysis.png")
    print(f"  - heat_flux_components.png")
    print(f"  - moisture_content.png")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    pavement = "CP"

    main(pavement)
