"""
Calibration and Validation Script - Split Dataset Approach

This script:
1. Splits data into calibration (40%) and validation (60%) sets
2. Runs the integrated model on both datasets separately
3. Calculates performance metrics for each dataset
4. Saves results for all pavements

Author: Calibration/Validation Analysis
Date: 2025
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

# Add parent directory to path
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


def process_pavement(pavement, calib_fraction=0.4):
    """
    Process a single pavement type with calibration/validation split.

    Args:
        pavement: Pavement type (e.g., 'PA', 'PC', 'PICP', 'PGr', 'CP')
        calib_fraction: Fraction of data for calibration (default 0.4 = 40%)
    """
    print("\n" + "=" * 80)
    print(f"PROCESSING {pavement} PAVEMENT")
    print("=" * 80)

    # Define paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(base_dir, "input_data", pavement)
    output_dir = os.path.join(base_dir, "output", f"{pavement}_calib_valid")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Input files
    data_file = os.path.join(input_dir, f"input_data_{pavement}.csv")
    temp_params_file = os.path.join(input_dir, f"temperature_parameters_{pavement}.ini")
    wf_params_file = os.path.join(input_dir, f"water_flow_calibrated_parameters_{pavement}.ini")

    # Check files exist
    for file_path in [data_file, temp_params_file, wf_params_file]:
        if not os.path.exists(file_path):
            print(f"  ERROR: File not found: {file_path}")
            return None

    # Load observed data
    print("\nLoading observed data...")
    obs_df = pd.read_csv(data_file, parse_dates=['date'])
    print(f"  Total timesteps: {len(obs_df)}")
    print(f"  Period: {obs_df['date'].iloc[0]} to {obs_df['date'].iloc[-1]}")

    # Split data into calibration and validation
    n_total = len(obs_df)
    n_calib = int(n_total * calib_fraction)

    calib_df = obs_df.iloc[:n_calib].copy()
    valid_df = obs_df.iloc[n_calib:].copy()

    print(f"\nData split:")
    print(f"  Calibration:  {len(calib_df)} timesteps ({100*calib_fraction:.0f}%)")
    print(f"    Period: {calib_df['date'].iloc[0]} to {calib_df['date'].iloc[-1]}")
    print(f"  Validation:   {len(valid_df)} timesteps ({100*(1-calib_fraction):.0f}%)")
    print(f"    Period: {valid_df['date'].iloc[0]} to {valid_df['date'].iloc[-1]}")

    # Dictionary to store results
    results = {
        'pavement': pavement,
        'calibration': {},
        'validation': {}
    }

    # Process calibration dataset
    print("\n" + "-" * 80)
    print("Running model on CALIBRATION dataset...")

    try:
        calib_results, _ = itm.run_integrated_model(
            sim_df=calib_df,
            temp_params_file=temp_params_file,
            wf_params_file=wf_params_file,
            rainfall_column='Rainfall',
            rainfall_unit='mm',
            climate_change=False
        )
        print("  Calibration run completed successfully!")

        # Calculate calibration metrics
        calib_metrics = []

        # Surface temperature
        surf_metrics = calculate_performance_metrics(
            calib_df['PavementTemperature'].values,
            calib_results['surface_temp'].values,
            "Surface Temperature"
        )
        calib_metrics.append(surf_metrics)

        # Bottom temperature (if available)
        if calib_df['BottomTemperature'].notna().sum() > 10:
            bottom_metrics = calculate_performance_metrics(
                calib_df['BottomTemperature'].values,
                calib_results['subsurface_temp'].values,
                "Bottom Temperature"
            )
            calib_metrics.append(bottom_metrics)

        # Water temperature (if available)
        if calib_df['WaterTemperature'].notna().sum() > 10:
            water_metrics = calculate_performance_metrics(
                calib_df['WaterTemperature'].values,
                calib_results['water_temp_infil'].values,
                "Water Temperature"
            )
            calib_metrics.append(water_metrics)

        results['calibration']['metrics'] = calib_metrics
        results['calibration']['results'] = calib_results

        # Display calibration metrics
        print("\n  CALIBRATION Performance:")
        for m in calib_metrics:
            if m['n_points'] > 0:
                print(f"    {m['variable']}: NSE={m['nse']:.4f}, RMSE={m['rmse']:.3f}°C, n={m['n_points']}")

    except Exception as e:
        print(f"  ERROR during calibration model run: {e}")
        return None

    # Process validation dataset
    print("\n" + "-" * 80)
    print("Running model on VALIDATION dataset...")

    try:
        valid_results, _ = itm.run_integrated_model(
            sim_df=valid_df,
            temp_params_file=temp_params_file,
            wf_params_file=wf_params_file,
            rainfall_column='Rainfall',
            rainfall_unit='mm',
            climate_change=False
        )
        print("  Validation run completed successfully!")

        # Calculate validation metrics
        valid_metrics = []

        # Surface temperature
        surf_metrics = calculate_performance_metrics(
            valid_df['PavementTemperature'].values,
            valid_results['surface_temp'].values,
            "Surface Temperature"
        )
        valid_metrics.append(surf_metrics)

        # Bottom temperature (if available)
        if valid_df['BottomTemperature'].notna().sum() > 10:
            bottom_metrics = calculate_performance_metrics(
                valid_df['BottomTemperature'].values,
                valid_results['subsurface_temp'].values,
                "Bottom Temperature"
            )
            valid_metrics.append(bottom_metrics)

        # Water temperature (if available)
        if valid_df['WaterTemperature'].notna().sum() > 10:
            water_metrics = calculate_performance_metrics(
                valid_df['WaterTemperature'].values,
                valid_results['water_temp_infil'].values,
                "Water Temperature"
            )
            valid_metrics.append(water_metrics)

        results['validation']['metrics'] = valid_metrics
        results['validation']['results'] = valid_results

        # Display validation metrics
        print("\n  VALIDATION Performance:")
        for m in valid_metrics:
            if m['n_points'] > 0:
                print(f"    {m['variable']}: NSE={m['nse']:.4f}, RMSE={m['rmse']:.3f}°C, n={m['n_points']}")

    except Exception as e:
        print(f"  ERROR during validation model run: {e}")
        return None

    # Save results
    print("\n" + "-" * 80)
    print("Saving results...")

    # Save calibration metrics
    calib_metrics_df = pd.DataFrame(results['calibration']['metrics'])
    calib_metrics_file = os.path.join(output_dir, "calibration_metrics.csv")
    calib_metrics_df.to_csv(calib_metrics_file, index=False)
    print(f"  Calibration metrics: {calib_metrics_file}")

    # Save validation metrics
    valid_metrics_df = pd.DataFrame(results['validation']['metrics'])
    valid_metrics_file = os.path.join(output_dir, "validation_metrics.csv")
    valid_metrics_df.to_csv(valid_metrics_file, index=False)
    print(f"  Validation metrics: {valid_metrics_file}")

    # Save combined summary for LaTeX table
    summary = []
    for phase in ['calibration', 'validation']:
        for m in results[phase]['metrics']:
            if m['variable'] == 'Surface Temperature' and m['n_points'] > 0:
                summary.append({
                    'pavement': pavement,
                    'phase': phase,
                    'variable': m['variable'],
                    'nse': m['nse'],
                    'rmse': m['rmse'],
                    'mae': m['mae'],
                    'r2': m['r2'],
                    'n_points': m['n_points']
                })

    summary_df = pd.DataFrame(summary)
    summary_file = os.path.join(output_dir, "summary.csv")
    summary_df.to_csv(summary_file, index=False)
    print(f"  Summary: {summary_file}")

    return results


def main():
    """Main function to process all pavements."""

    print("=" * 80)
    print("CALIBRATION AND VALIDATION ANALYSIS")
    print("Data Split: 40% Calibration / 60% Validation")
    print("=" * 80)

    start_time = datetime.now()

    # List of pavements to process
    pavements = ['CP', 'PICP', 'PGr', 'PA', 'PC']

    # Store all results
    all_results = {}

    # Process each pavement
    for pavement in pavements:
        result = process_pavement(pavement, calib_fraction=0.4)
        if result is not None:
            all_results[pavement] = result

    # Create combined summary table for all pavements
    print("\n" + "=" * 80)
    print("CREATING COMBINED SUMMARY")
    print("=" * 80)

    combined_summary = []
    for pavement in pavements:
        if pavement in all_results:
            for phase in ['calibration', 'validation']:
                for m in all_results[pavement][phase]['metrics']:
                    if m['variable'] == 'Surface Temperature' and m['n_points'] > 0:
                        combined_summary.append({
                            'pavement': pavement,
                            'phase': phase,
                            'nse': m['nse'],
                            'rmse': m['rmse'],
                            'mae': m['mae'],
                            'r2': m['r2']
                        })

    combined_df = pd.DataFrame(combined_summary)

    # Save combined summary
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, "output")
    combined_file = os.path.join(output_dir, "all_pavements_calib_valid_summary.csv")
    combined_df.to_csv(combined_file, index=False)
    print(f"\nCombined summary saved to: {combined_file}")

    # Display summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE (Surface Temperature)")
    print("=" * 80)
    print(f"\n{'Pavement':<10} {'Phase':<15} {'RMSE (°C)':<12} {'NSE':<8}")
    print("-" * 50)

    for pavement in pavements:
        if pavement in all_results:
            # Calibration
            calib_surf = [m for m in all_results[pavement]['calibration']['metrics']
                         if m['variable'] == 'Surface Temperature' and m['n_points'] > 0]
            if calib_surf:
                m = calib_surf[0]
                print(f"{pavement:<10} {'Calibration':<15} {m['rmse']:>8.2f}    {m['nse']:>6.3f}")

            # Validation
            valid_surf = [m for m in all_results[pavement]['validation']['metrics']
                         if m['variable'] == 'Surface Temperature' and m['n_points'] > 0]
            if valid_surf:
                m = valid_surf[0]
                print(f"{pavement:<10} {'Validation':<15} {m['rmse']:>8.2f}    {m['nse']:>6.3f}")

    # Execution time
    elapsed_time = datetime.now() - start_time

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nElapsed time: {elapsed_time}")
    print(f"\nResults saved to: {output_dir}")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
