"""
Run Integrated Temperature-Infiltration Model for Climate Change Scenarios

This script:
1. Loads climate change input data (historical, ssp126, ssp585)
2. Runs the integrated model for all pavements with dt=3600s (hourly timestep)
3. Saves results similar to climate_change/output/ structure

IMPORTANT:
- Uses *_climate.ini parameter files with dt=3600s for hourly climate data
- Regular calibration/validation uses dt=300s (5-minute data)
- Climate scenarios use 'pr' column for precipitation
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import integrated_temperature_infiltration as itm


def run_climate_scenario(pavement, scenario, climate_data_file, output_dir):
    """
    Run integrated model for a specific pavement and climate scenario.

    Args:
        pavement: Pavement type (e.g., 'PA', 'PC', 'PICP', 'PGr', 'CP')
        scenario: Climate scenario ('historical', 'ssp126', 'ssp585')
        climate_data_file: Path to climate change input CSV
        output_dir: Directory to save results

    Returns:
        Boolean indicating success
    """
    print(f"\n{'='*80}")
    print(f"Processing: {pavement} - {scenario}")
    print(f"{'='*80}")

    # Define paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(base_dir, "input_data", pavement)

    # Parameter files - use climate-specific versions with dt=3600
    temp_params_file = os.path.join(input_dir, f"temperature_parameters_{pavement}_climate.ini")
    wf_params_file = os.path.join(input_dir, f"water_flow_calibrated_parameters_{pavement}_climate.ini")

    # Check files exist
    if not os.path.exists(temp_params_file):
        print(f"  WARNING: Temperature parameters not found: {temp_params_file}")
        return False

    if not os.path.exists(wf_params_file):
        print(f"  WARNING: Water flow parameters not found: {wf_params_file}")
        return False

    if not os.path.exists(climate_data_file):
        print(f"  ERROR: Climate data file not found: {climate_data_file}")
        return False

    # Load climate change data
    print(f"  Loading climate data from: {climate_data_file}")
    try:
        climate_df = pd.read_csv(climate_data_file, parse_dates=['date'])
        print(f"    Loaded {len(climate_df)} timesteps")
        print(f"    Period: {climate_df['date'].iloc[0]} to {climate_df['date'].iloc[-1]}")
    except Exception as e:
        print(f"  ERROR loading climate data: {e}")
        return False

    # Run integrated model
    print(f"  Running integrated model...")
    print(f"    (This may take several minutes)")

    try:
        results_df, water_flow_df = itm.run_integrated_model(
            sim_df=climate_df,
            temp_params_file=temp_params_file,
            wf_params_file=wf_params_file,
            rainfall_column='pr',  # Climate data uses 'pr' for precipitation
            rainfall_unit='mm',
            climate_change=True
        )

        print(f"  Model run completed successfully!")

        # Save results in the same format as climate_change/output/
        output_file = os.path.join(output_dir, f"{pavement}_scenario{scenario}.csv")

        # Select columns to match the original climate_change output format
        output_columns = [
            'h_s', 'h_li', 'h_l0', 'h_rad', 'h_evap', 'h_conv', 'h_r0',
            'h_net', 'surface_temp', 'rainfall', 'water_temp_surface',
            'water_temp_infil', 'date'
        ]

        # Ensure all columns exist
        for col in output_columns:
            if col not in results_df.columns:
                if col == 'rainfall':
                    # Map pr to rainfall if needed
                    results_df['rainfall'] = climate_df['pr'] if 'pr' in climate_df.columns else 0
                else:
                    results_df[col] = np.nan

        # Save with selected columns
        results_df[output_columns].to_csv(output_file, index=False)
        print(f"  Saved results to: {output_file}")

        # Also save full results for reference
        full_output_file = os.path.join(output_dir, f"{pavement}_scenario{scenario}_full.csv")
        results_df.to_csv(full_output_file, index=False)
        print(f"  Saved full results to: {full_output_file}")

        # Save water flow results
        water_flow_file = os.path.join(output_dir, f"{pavement}_scenario{scenario}_waterflow.csv")
        water_flow_df.to_csv(water_flow_file, index=False)
        print(f"  Saved water flow to: {water_flow_file}")

        return True

    except Exception as e:
        print(f"  ERROR during model execution: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function to run all climate scenarios for all pavements."""

    print("="*80)
    print("CLIMATE CHANGE SCENARIOS - INTEGRATED MODEL")
    print("="*80)

    start_time = datetime.now()

    # Define pavements and scenarios
    pavements = ["CP", "PICP", "PGr", "PA", "PC"]
    scenarios = ["historical", "ssp126", "ssp585"]

    # Define paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_data_dir = os.path.join(base_dir, "input_data")
    output_dir = os.path.join(base_dir, "output", "climate_change")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nConfiguration:")
    print(f"  Pavements: {', '.join(pavements)}")
    print(f"  Scenarios: {', '.join(scenarios)}")
    print(f"  Output directory: {output_dir}")

    # Climate data files
    climate_files = {
        'historical': os.path.join(input_data_dir, 'climate_change_historical.csv'),
        'ssp126': os.path.join(input_data_dir, 'climate_change_ssp126.csv'),
        'ssp585': os.path.join(input_data_dir, 'climate_change_ssp585.csv')
    }

    # Check climate files exist
    print(f"\nChecking climate data files:")
    for scenario, file_path in climate_files.items():
        if os.path.exists(file_path):
            print(f"  {scenario}: Found")
        else:
            print(f"  {scenario}: NOT FOUND - {file_path}")

    # Track results
    success_count = 0
    failure_count = 0
    results_summary = []

    # Process each combination
    total_runs = len(pavements) * len(scenarios)
    current_run = 0

    for pavement in pavements:
        for scenario in scenarios:
            current_run += 1
            print(f"\n{'='*80}")
            print(f"Run {current_run}/{total_runs}: {pavement} - {scenario}")
            print(f"{'='*80}")

            climate_file = climate_files[scenario]

            success = run_climate_scenario(
                pavement=pavement,
                scenario=scenario,
                climate_data_file=climate_file,
                output_dir=output_dir
            )

            if success:
                success_count += 1
                results_summary.append({
                    'pavement': pavement,
                    'scenario': scenario,
                    'status': 'Success'
                })
            else:
                failure_count += 1
                results_summary.append({
                    'pavement': pavement,
                    'scenario': scenario,
                    'status': 'Failed'
                })

    # Save summary
    summary_df = pd.DataFrame(results_summary)
    summary_file = os.path.join(output_dir, 'run_summary.csv')
    summary_df.to_csv(summary_file, index=False)

    # Final summary
    elapsed_time = datetime.now() - start_time

    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"\nTotal runs: {total_runs}")
    print(f"  Successful: {success_count}")
    print(f"  Failed: {failure_count}")
    print(f"\nElapsed time: {elapsed_time}")
    print(f"\nResults saved to: {output_dir}")
    print(f"Summary saved to: {summary_file}")

    # Display summary table
    print(f"\n{'='*80}")
    print("SUMMARY TABLE")
    print(f"{'='*80}")
    print(f"\n{'Pavement':<10} {'Historical':<12} {'SSP126':<12} {'SSP585':<12}")
    print("-"*50)

    for pavement in pavements:
        status_hist = summary_df[(summary_df['pavement'] == pavement) &
                                 (summary_df['scenario'] == 'historical')]['status'].values[0]
        status_126 = summary_df[(summary_df['pavement'] == pavement) &
                                (summary_df['scenario'] == 'ssp126')]['status'].values[0]
        status_585 = summary_df[(summary_df['pavement'] == pavement) &
                                (summary_df['scenario'] == 'ssp585')]['status'].values[0]

        print(f"{pavement:<10} {status_hist:<12} {status_126:<12} {status_585:<12}")

    print(f"\n{'='*80}")


if __name__ == "__main__":
    main()
