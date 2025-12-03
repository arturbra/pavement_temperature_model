"""
Example script for running the integrated temperature and water infiltration model.

This script demonstrates how to:
1. Load meteorological forcing data
2. Set up parameter files
3. Run the integrated model
4. Save and analyze results

Author: Integrated Model Example
Date: 2025
"""

import pandas as pd
import numpy as np
import os
import sys

# Add parent directory to path to import temperature_model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import integrated_temperature_infiltration as itm


def load_example_data(data_file=None):
    """
    Load or create example meteorological data.

    Args:
        data_file: Path to CSV file with meteorological data (optional)

    Returns:
        DataFrame with required columns
    """
    if data_file is not None and os.path.exists(data_file):
        # Load existing data
        df = pd.read_csv(data_file, parse_dates=['date'])
        return df
    else:
        # Create synthetic example data for testing
        print("Creating synthetic example data...")

        # Create a week of hourly data
        n_hours = 24 * 7  # 1 week
        dates = pd.date_range('2024-01-01', periods=n_hours, freq='H')

        # Synthetic meteorological data
        hours = np.arange(n_hours)
        day_hours = hours % 24

        # Temperature variation (°C)
        T_air = 20 + 10 * np.sin(2 * np.pi * hours / 24 - np.pi / 2)

        # Solar radiation (W/m²)
        solar = np.maximum(0, 800 * np.sin(np.pi * day_hours / 24))

        # Wind speed (m/s)
        wind = 2 + 1 * np.random.randn(n_hours) * 0.5
        wind = np.maximum(0.5, wind)

        # Relative humidity (-)
        RH = 0.6 + 0.2 * np.sin(2 * np.pi * hours / 24)

        # Cloud coverage (-)
        CC = 0.3 + 0.2 * np.random.randn(n_hours) * 0.3
        CC = np.clip(CC, 0, 1)

        # Dew point (°C)
        T_dew = T_air - 5

        # Rainfall (mm/hour) - add some rain events
        rainfall = np.zeros(n_hours)
        # Add 3 rain events
        rainfall[48:52] = 10  # Heavy rain
        rainfall[96:100] = 5  # Moderate rain
        rainfall[144:150] = 15  # Heavy rain

        # Create DataFrame
        df = pd.DataFrame({
            'date': dates,
            'AirTemperature': T_air,
            'SolarRadiation': solar,
            'WindSpeed': wind,
            'RelativeHumidity': RH,
            'CloudCoverage': CC,
            'DewPoint': T_dew,
            'Rainfall': rainfall
        })

        return df


def main():
    """Main execution function."""

    print("=" * 80)
    print("INTEGRATED TEMPERATURE AND WATER INFILTRATION MODEL - EXAMPLE")
    print("=" * 80)

    # Define paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(base_dir, "input_data")
    output_dir = os.path.join(base_dir, "output")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Parameter files
    temp_params_file = os.path.join(input_dir, "temperature_parameters.ini")
    wf_params_file = os.path.join(input_dir, "water_flow_parameters.ini")

    print(f"\nUsing parameter files:")
    print(f"  Temperature: {temp_params_file}")
    print(f"  Water Flow:  {wf_params_file}")

    # Check if parameter files exist
    if not os.path.exists(temp_params_file):
        print(f"\nERROR: Temperature parameter file not found: {temp_params_file}")
        print("Please ensure the file exists in the input_data directory.")
        return

    if not os.path.exists(wf_params_file):
        print(f"\nERROR: Water flow parameter file not found: {wf_params_file}")
        print("Please ensure the file exists in the input_data directory.")
        return

    # Load meteorological data
    print("\n" + "-" * 80)
    print("Loading meteorological data...")

    # Try to load from parent directory's input_data
    parent_input_file = os.path.join(
        os.path.dirname(base_dir),
        "input_data", "1h", "input_data_PICP2.csv"
    )

    if os.path.exists(parent_input_file):
        print(f"Loading data from: {parent_input_file}")
        sim_df = pd.read_csv(parent_input_file, parse_dates=['date'])
        # Limit to a subset for faster testing
        sim_df = sim_df.iloc[:168]  # First week
        print(f"Loaded {len(sim_df)} timesteps")
    else:
        print("Using synthetic test data")
        sim_df = load_example_data()

    # Verify required columns
    required_cols = ['date', 'AirTemperature', 'SolarRadiation', 'WindSpeed',
                    'RelativeHumidity', 'CloudCoverage', 'DewPoint', 'Rainfall']

    missing_cols = [col for col in required_cols if col not in sim_df.columns]
    if missing_cols:
        print(f"\nERROR: Missing required columns: {missing_cols}")
        return

    print(f"Data range: {sim_df['date'].iloc[0]} to {sim_df['date'].iloc[-1]}")
    print(f"Total rainfall: {sim_df['Rainfall'].sum():.1f} mm")

    # Run the integrated model
    print("\n" + "-" * 80)
    print("Running integrated model...")
    print("(This may take a few moments)")

    try:
        results_df, water_flow_df = itm.run_integrated_model(
            sim_df=sim_df,
            temp_params_file=temp_params_file,
            wf_params_file=wf_params_file,
            rainfall_column='Rainfall',
            rainfall_unit='mm',
            climate_change=False
        )

        print("Model run completed successfully!")

        # Save results
        print("\n" + "-" * 80)
        print("Saving results...")

        results_file = os.path.join(output_dir, "integrated_results.csv")
        water_flow_file = os.path.join(output_dir, "water_flow_results.csv")

        results_df.to_csv(results_file, index=False)
        water_flow_df.to_csv(water_flow_file, index=False)

        print(f"Temperature results saved to: {results_file}")
        print(f"Water flow results saved to:  {water_flow_file}")

        # Display summary statistics
        print("\n" + "-" * 80)
        print("RESULTS SUMMARY")
        print("-" * 80)

        print("\nTemperature Statistics:")
        print(f"  Surface Temperature:")
        print(f"    Mean:    {results_df['surface_temp'].mean():.2f} °C")
        print(f"    Min:     {results_df['surface_temp'].min():.2f} °C")
        print(f"    Max:     {results_df['surface_temp'].max():.2f} °C")
        print(f"    Std Dev: {results_df['surface_temp'].std():.2f} °C")

        print(f"\n  Subsurface Temperature:")
        print(f"    Mean:    {results_df['subsurface_temp'].mean():.2f} °C")
        print(f"    Min:     {results_df['subsurface_temp'].min():.2f} °C")
        print(f"    Max:     {results_df['subsurface_temp'].max():.2f} °C")
        print(f"    Std Dev: {results_df['subsurface_temp'].std():.2f} °C")

        print(f"\nHeat Flux Statistics (W/m²):")
        print(f"  Net Radiation:     {results_df['h_rad'].mean():.2f}")
        print(f"  Convection:        {results_df['h_conv'].mean():.2f}")
        print(f"  Evaporation:       {results_df['h_evap'].mean():.2f}")
        print(f"  Rainfall Exchange: {results_df['h_r0'].mean():.2f}")

        print(f"\nWater Content Statistics:")
        print(f"  Surface θ (mean):     {results_df['theta_surface'].mean():.3f}")
        print(f"  Subsurface θ (mean):  {results_df['theta_subsurface'].mean():.3f}")
        print(f"  Max Ponding Depth:    {results_df['ponding_depth'].max():.4f} m")
        print(f"  Mean Saturation:      {results_df['saturation'].mean():.3f}")

        print(f"\nWater Flow Statistics:")
        print(f"  Total Pipe Outflow: {water_flow_df['Qpipe'].sum() * 3600:.3f} m³")
        print(f"  Peak Pipe Flow:     {water_flow_df['Qpipe'].max():.6f} m³/s")
        print(f"  Max Sat. Zone Depth: {water_flow_df['hsz'].max():.4f} m")

        print("\n" + "=" * 80)
        print("SIMULATION COMPLETE")
        print("=" * 80)

    except Exception as e:
        print(f"\nERROR during model execution:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
