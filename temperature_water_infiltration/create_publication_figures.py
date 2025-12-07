"""
Create Publication-Quality Figures for Temperature and Water Infiltration Model

This script creates all figures needed for publication:
1. Event-specific plots for permeable pavements (similar to output/pavements/)
2. Multi-pavement comparison plots (similar to pavement_comparison_plots/)

Based on the original plotting code from temperature_evaluate.py and water_temperature_evaluate.py

Author: Publication Figures Generator
Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import sys
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import integrated_temperature_infiltration as itm


# Event dates for specific plots (from original output/pavements/)
# EVENT_DATES = {
#     'event1': ('2023-10-05 05:00', '2023-10-06 05:00'),
#     'event2': ('2023-11-09 21:00', '2023-11-10 15:00'),
#     'event3': ('2024-01-21 05:00', '2024-01-23 08:00'),
#     'event4': ('2024-01-31 22:00', '2024-02-03 10:00'),
#     'event5': ('2024-04-28 04:00', '2024-04-29 04:00'),
# }

EVENT_DATES = {
    'event4': ('2024-01-31 22:00', '2024-02-03 10:00'),
}

# Color mapping for pavements (from original code)
COLOR_MAPPING = {
    'CP': (0.121, 0.467, 0.706),
    'PICP': (1.0, 0.498, 0.055),
    'PGr': (0.173, 0.627, 0.173),
    'PA': (0.580, 0.404, 0.741),
    'PC': (0.839, 0.153, 0.157),
}

# Pavement names for titles
PAVEMENT_NAMES = {
    "CP": "Conventional Pavement",
    "PC": "Permeable Concrete",
    "PICP": "Permeable Interlocking Concrete Pavers",
    "PA": "Porous Asphalt",
    "PGr": "Permeable Gravel"
}

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd


def plot_event_for_pavement(combined_df, start_date, end_date, pavement_type="PICP",
                            y_min=None, y_max=None, save_path=None):
    """
    Plot three-panel event figure with rainfall, surface temperature, and water temperature.

    Adjustments made:
    - X-axis format: YYYY-Mon-DD
    - X-axis ticks: Multiples of 0, 6, 12, 18 hours.
    - Date Placement: First date left-aligned to Y-axis; subsequent dates centered on 00:00.
    """
    # Set font sizes
    TITLE_SIZE = 20
    LABEL_SIZE = 18
    LEGEND_SIZE = 18
    TICK_SIZE = 18

    # Handle global variable dependency safely
    try:
        pavement_full_name = PAVEMENT_NAMES.get(pavement_type, pavement_type)
    except NameError:
        pavement_full_name = pavement_type

    # Ensure date column is datetime
    if combined_df['date'].dtype != 'datetime64[ns]':
        combined_df['date'] = pd.to_datetime(combined_df['date'])

    # Filter data for selected date range
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Make dates timezone-aware if needed to match the data
    if hasattr(combined_df['date'].dtype, 'tz') and combined_df['date'].dt.tz is not None:
        start_date = start_date.tz_localize(combined_df['date'].dt.tz)
        end_date = end_date.tz_localize(combined_df['date'].dt.tz)

    mask = (combined_df['date'] >= start_date) & (combined_df['date'] <= end_date)
    event_data = combined_df[mask].copy()

    if len(event_data) == 0:
        print(f"No data for {pavement_type} in range {start_date} to {end_date}")
        return None

    # Calculate total rainfall for annotation
    total_rain = event_data['Rainfall'].sum() if 'Rainfall' in event_data.columns else 0.0

    # Create figure with 3 subplots (rainfall, surface temp, water temp)
    fig, axs = plt.subplots(3, 1, figsize=(20, 12), sharex=True,
                            gridspec_kw={'height_ratios': [1, 2, 2]})

    # --- Panel 1: Rainfall subplot (top) ---
    if 'Rainfall' in event_data.columns:
        axs[0].bar(event_data['date'], event_data['Rainfall'], width=0.008, color='black')
        axs[0].invert_yaxis()
        axs[0].set_ylabel('Rainfall (mm)', fontsize=LABEL_SIZE)
        axs[0].set_title('Rainfall', fontsize=TITLE_SIZE)
        axs[0].tick_params(axis='both', which='major', labelsize=TICK_SIZE)

        # Add total rainfall annotation
        axs[0].annotate(f'Total: {total_rain:.1f} mm',
                        xy=(0.99, 0.55), xycoords='axes fraction',
                        ha='right', va='top', fontsize=16, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.7))

    # --- Panel 2: Surface Temperature subplot ---
    # Plot observed data
    if 'PavementTemperature' in event_data.columns:
        axs[1].plot(event_data['date'], event_data['PavementTemperature'],
                    color='red', marker='o', markersize=3, label='Observed Surface')

    # Plot model results - dashed line
    if 'surface_temp' in event_data.columns:
        axs[1].plot(event_data['date'], event_data['surface_temp'],
                    color='red', linestyle='--', linewidth=2, label='Modeled Surface')

    axs[1].set_ylabel('Temperature (°C)', fontsize=LABEL_SIZE)
    axs[1].set_title(f'{pavement_type} Surface Temperature', fontsize=TITLE_SIZE)
    axs[1].grid(True, linestyle='--', alpha=0.7)
    axs[1].legend(loc='best', fontsize=LEGEND_SIZE)
    axs[1].tick_params(axis='both', which='major', labelsize=TICK_SIZE)
    axs[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}'))

    # --- Panel 3: Water Temperature subplot ---
    water_data = []
    if 'WaterTemperature' in event_data.columns:
        observed_water_data = event_data[['date', 'WaterTemperature']].dropna()

        if not observed_water_data.empty:
            water_obs_label = f'Observed Water'
            axs[2].plot(observed_water_data['date'], observed_water_data['WaterTemperature'],
                        color='blue', linestyle='-', linewidth=2, label=water_obs_label)

            water_data.extend(observed_water_data['WaterTemperature'].tolist())

            # Only plot model results for timestamps where we have observed data
            if 'water_temp_infil' in event_data.columns:
                common_dates = observed_water_data['date']
                filtered_results = event_data[event_data['date'].isin(common_dates)]

                if not filtered_results.empty:
                    water_model_label = f'Modeled Water'
                    axs[2].plot(filtered_results['date'], filtered_results['water_temp_infil'],
                                color='blue', linestyle='--', linewidth=2, label=water_model_label)
                    water_data.extend(filtered_results['water_temp_infil'].tolist())

    # Calculate y-axis limits for water temperature subplot
    min_temp = 0
    max_temp = max(water_data) if water_data else 40
    y_min_calc = min_temp
    y_max_calc = max_temp + 15
    axs[2].set_ylim(y_min_calc, y_max_calc)

    axs[2].set_ylabel('Temperature (°C)', fontsize=LABEL_SIZE)
    axs[2].set_title('Water Temperature', fontsize=TITLE_SIZE)
    axs[2].grid(True, linestyle='--', alpha=0.7)
    axs[2].legend(loc='best', fontsize=LEGEND_SIZE)
    axs[2].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}'))

    import matplotlib.dates as mdates

    # X-limits (use the same start/end you used for filtering)
    axs[2].set_xlim(start_date, end_date)

    # Major ticks at 0, 6, 12, 18 hours
    locator = mdates.HourLocator(byhour=[0, 6, 12, 18])
    formatter = mdates.DateFormatter('%H')
    axs[2].xaxis.set_major_locator(locator)
    axs[2].xaxis.set_major_formatter(formatter)
    axs[2].tick_params(axis='x', which='major', labelsize=TICK_SIZE)
    axs[2].tick_params(axis='y', which='major', labelsize=TICK_SIZE)

    # ---- Custom date labels ----
    DATE_SIZE = TICK_SIZE - 2           # slightly smaller than hour ticks
    date_y_pos = -0.18                  # vertical offset below hour labels
    date_fmt = '%Y-%b-%d'               # e.g. 2024-Feb-02

    # 1) First date: aligned with the Y-axis (left border)
    x_min, x_max = axs[2].get_xlim()
    first_date_dt = mdates.num2date(x_min)
    first_date_text = first_date_dt.strftime(date_fmt)

    axs[2].text(
        x_min, date_y_pos, first_date_text,
        ha='left', va='top',
        fontsize=DATE_SIZE,
        transform=axs[2].get_xaxis_transform()
    )

    # 2) Subsequent dates: under every tick at hour 00
    xticks = axs[2].get_xticks()
    for tick in xticks:
        # skip ticks at or before the left border (so we don't duplicate the first date)
        if tick <= x_min:
            continue

        dt = mdates.num2date(tick)
        if dt.hour == 0:
            label = dt.strftime(date_fmt)
            axs[2].text(
                tick, date_y_pos, label,
                ha='center', va='top',
                fontsize=DATE_SIZE,
                transform=axs[2].get_xaxis_transform()
            )

    axs[2].set_xlabel('Time (Hours)', fontsize=LABEL_SIZE)

    plt.tight_layout()

    # Save figure
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")

    plt.close()

    return fig


def plot_all_pavements_comparison(observed_data_dict, model_results_dict, start_date, end_date, save_path=None):
    """
    Plot surface temperature comparison for all pavements in one figure.
    Adapted from water_temperature_evaluate.py:plot_all_pavement_surface_temps_period()

    This matches the style of the original pavement_comparison_plots/ directory.

    Args:
        observed_data_dict: Dictionary with pavement keys and DataFrames with observed data
        model_results_dict: Dictionary with pavement keys and model results DataFrames
        start_date: Start date for plot
        end_date: End date for plot
        save_path: Path to save figure
    """
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # List of pavements to plot
    pavements = ['CP', 'PICP', 'PGr', 'PA', 'PC']

    # ========== FONT SIZE SETTINGS (adjust here for all text elements) ==========
    SUBPLOT_TITLE_SIZE = 22       # Size of pavement name titles (e.g., "Conventional Pavement")
    AXIS_LABEL_SIZE = 20          # Size of axis labels (e.g., "Date", "Temperature (°C)")
    TICK_LABEL_SIZE = 18          # Size of tick labels (numbers on axes)
    LEGEND_SIZE = 18              # Size of legend text
    RMSE_ANNOTATION_SIZE = 18     # Size of RMSE text annotation
    # ===========================================================================

    # Create figure with subplots
    fig, axs = plt.subplots(len(pavements), 1, figsize=(16, 16), sharex=True)

    all_temps = []

    # Plot each pavement
    for i, pavement in enumerate(pavements):
        ax = axs[i]
        color = COLOR_MAPPING.get(pavement, (0, 0, 0))

        # Get observed data for this pavement
        if pavement in observed_data_dict:
            obs_df = observed_data_dict[pavement].copy()
            obs_df['date'] = pd.to_datetime(obs_df['date'])

            # Match timezone if needed
            start_date_comp = start_date
            end_date_comp = end_date
            if hasattr(obs_df['date'].dtype, 'tz') and obs_df['date'].dt.tz is not None:
                if start_date.tz is None:
                    start_date_comp = start_date.tz_localize(obs_df['date'].dt.tz)
                    end_date_comp = end_date.tz_localize(obs_df['date'].dt.tz)

            # Filter for period
            obs_period = obs_df[(obs_df['date'] >= start_date_comp) & (obs_df['date'] < end_date_comp)]

            if not obs_period.empty and 'PavementTemperature' in obs_period.columns:
                temp_data = obs_period[['date', 'PavementTemperature']].dropna()
                if not temp_data.empty:
                    ax.plot(temp_data['date'], temp_data['PavementTemperature'],
                            color=color, linestyle='-', linewidth=2, marker='o', markersize=3,
                            label='Observed Surface')
                    all_temps.extend(temp_data['PavementTemperature'].tolist())

        # Plot modeled data if available
        if pavement in model_results_dict:
            model_df = model_results_dict[pavement].copy()
            model_df['date'] = pd.to_datetime(model_df['date'])

            # Match timezone if needed
            start_date_comp = start_date
            end_date_comp = end_date
            if hasattr(model_df['date'].dtype, 'tz') and model_df['date'].dt.tz is not None:
                if start_date.tz is None:
                    start_date_comp = start_date.tz_localize(model_df['date'].dt.tz)
                    end_date_comp = end_date.tz_localize(model_df['date'].dt.tz)

            # Filter for period
            model_period = model_df[(model_df['date'] >= start_date_comp) & (model_df['date'] < end_date_comp)]

            if not model_period.empty:
                ax.plot(model_period['date'], model_period['surface_temp'],
                        color=color, linestyle='--', linewidth=2,
                        label='Modeled Surface')
                all_temps.extend(model_period['surface_temp'].tolist())

                # Calculate RMSE if both observed and modeled exist
                if pavement in observed_data_dict:
                    obs_df = observed_data_dict[pavement].copy()
                    obs_df['date'] = pd.to_datetime(obs_df['date'])

                    # Match timezone if needed
                    start_date_comp = start_date
                    end_date_comp = end_date
                    if hasattr(obs_df['date'].dtype, 'tz') and obs_df['date'].dt.tz is not None:
                        if start_date.tz is None:
                            start_date_comp = start_date.tz_localize(obs_df['date'].dt.tz)
                            end_date_comp = end_date.tz_localize(obs_df['date'].dt.tz)

                    obs_period = obs_df[(obs_df['date'] >= start_date_comp) & (obs_df['date'] < end_date_comp)]

                    if not obs_period.empty and 'PavementTemperature' in obs_period.columns:
                        temp_data = obs_period[['date', 'PavementTemperature']].dropna()

                        if not temp_data.empty:
                            merged = pd.merge(temp_data, model_period[['date', 'surface_temp']],
                                            on='date', how='inner')

                            if not merged.empty:
                                rmse = np.sqrt(((merged['PavementTemperature'] -
                                                merged['surface_temp']) ** 2).mean())

                                ax.annotate(f'RMSE: {rmse:.2f}°C',
                                          xy=(0.95, 0.05), xycoords='axes fraction',
                                          ha='right', va='bottom', fontsize=RMSE_ANNOTATION_SIZE, fontweight='bold',
                                          bbox=dict(boxstyle="round,pad=0.3", fc="white",
                                                   ec="black", alpha=0.7))

        # Format subplot
        ax.set_ylabel('Temperature (°C)', fontsize=AXIS_LABEL_SIZE)
        ax.set_title(f'{pavement} Surface Temperature', fontsize=SUBPLOT_TITLE_SIZE)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='upper right', fontsize=LEGEND_SIZE)
        ax.tick_params(axis='both', which='major', labelsize=TICK_LABEL_SIZE)

    # Set consistent y-axis limits
    if all_temps:
        y_min = min(all_temps) - 2
        y_max = max(all_temps) + 2
        for ax_i in axs:
            ax_i.set_ylim(y_min, y_max)

    # Format x-axis
    axs[-1].set_xlabel('Date', fontsize=AXIS_LABEL_SIZE)
    axs[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    # Apply tick label size to x-axis
    axs[-1].tick_params(axis='x', which='major', labelsize=TICK_LABEL_SIZE)

    # Adjust x-axis ticks based on period
    days_in_period = (end_date - start_date).days
    if days_in_period <= 2:
        axs[-1].xaxis.set_major_locator(mdates.HourLocator(interval=6))
    elif days_in_period <= 7:
        axs[-1].xaxis.set_major_locator(mdates.DayLocator())
        axs[-1].xaxis.set_minor_locator(mdates.HourLocator(interval=6))
    else:
        axs[-1].xaxis.set_major_locator(mdates.DayLocator(interval=2))

    plt.xticks(rotation=30)

    # Overall title
    plt.suptitle(
        f'Surface Temperature Comparison: {start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")}',
        fontsize=16, y=0.995)

    plt.tight_layout()

    # Save figure
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")

    plt.close()

    return fig


def main(pavements=['PICP', 'PGr', 'PA', 'PC', 'CP']):
    """
    Main function to create all publication figures.

    Args:
        pavements: List of pavement types to process
    """
    print("=" * 80)
    print("CREATE PUBLICATION FIGURES - INTEGRATED MODEL")
    print("=" * 80)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(base_dir, "input_data")
    output_base = os.path.join(base_dir, "output")

    # Create output directories
    events_dir = os.path.join(output_base, "event_plots")
    comparison_dir = os.path.join(output_base, "pavement_comparison_plots")
    model_cache_dir = os.path.join(output_base, "model_results")
    os.makedirs(events_dir, exist_ok=True)
    os.makedirs(comparison_dir, exist_ok=True)
    os.makedirs(model_cache_dir, exist_ok=True)

    # Dictionaries to store results
    observed_data_dict = {}
    model_results_dict = {}

    print("\n" + "-" * 80)
    print("STEP 1: Loading/Running models for all pavements")
    print("-" * 80)

    # Run models for all pavements
    for pavement in pavements:
        print(f"\nProcessing {pavement}...")

        pavement_input = os.path.join(input_dir, pavement)

        # Check if directory exists
        if not os.path.exists(pavement_input):
            print(f"  Skipping {pavement} - directory not found: {pavement_input}")
            continue

        data_file = os.path.join(pavement_input, f"input_data_{pavement}.csv")
        temp_params_file = os.path.join(pavement_input, f"temperature_parameters_{pavement}.ini")
        wf_params_file = os.path.join(pavement_input, f"water_flow_calibrated_parameters_{pavement}.ini")

        # Check files exist
        if not all(os.path.exists(f) for f in [data_file, temp_params_file, wf_params_file]):
            print(f"  Skipping {pavement} - missing required files")
            continue

        # Load observed data
        obs_df = pd.read_csv(data_file, parse_dates=['date'])
        observed_data_dict[pavement] = obs_df
        print(f"  Loaded {len(obs_df)} timesteps of observed data")

        # Check for cached model results
        cache_file = os.path.join(model_cache_dir, f"{pavement}_model_results.csv")

        if os.path.exists(cache_file):
            print(f"  Found cached model results at: {cache_file}")
            try:
                results_df = pd.read_csv(cache_file, parse_dates=['date'])
                model_results_dict[pavement] = results_df
                print(f"  Loaded {len(results_df)} timesteps from cache")
                continue
            except Exception as e:
                print(f"  Warning: Could not load cached results ({e}), will re-run model")

        # Run integrated model if no cache exists or cache failed to load
        print(f"  No cache found, running model...")
        try:
            results_df, water_flow_df = itm.run_integrated_model(
                sim_df=obs_df,
                temp_params_file=temp_params_file,
                wf_params_file=wf_params_file,
                rainfall_column='Rainfall',
                rainfall_unit='mm',
                climate_change=False
            )
            model_results_dict[pavement] = results_df
            print(f"  Model run completed successfully")

            # Save results to cache
            try:
                results_df.to_csv(cache_file, index=False)
                print(f"  Saved results to cache: {cache_file}")
            except Exception as e:
                print(f"  Warning: Could not save cache ({e}), continuing anyway")

        except Exception as e:
            print(f"  ERROR running model for {pavement}: {e}")
            continue

    print(f"\nSuccessfully processed {len(model_results_dict)} pavements")

    # Create event-specific plots for permeable pavements
    print("\n" + "-" * 80)
    print("STEP 2: Creating event-specific plots")
    print("-" * 80)

    permeable_pavements = ['PICP', 'PGr', 'PA', 'PC']

    for pavement in permeable_pavements:
        if pavement not in model_results_dict:
            print(f"\nSkipping {pavement} - no model results")
            continue

        print(f"\nCreating event plots for {pavement}...")

        # Combine observed and modeled data
        obs_df = observed_data_dict[pavement]
        model_df = model_results_dict[pavement]

        # Determine which columns to merge from observed data
        obs_columns = ['date', 'PavementTemperature']
        if 'Rainfall' in obs_df.columns:
            obs_columns.append('Rainfall')
        if 'WaterTemperature' in obs_df.columns:
            obs_columns.append('WaterTemperature')

        # Determine which columns to merge from model results
        model_columns = ['date', 'surface_temp']
        if 'water_temp_infil' in model_df.columns:
            model_columns.append('water_temp_infil')

        combined = pd.merge(
            obs_df[obs_columns],
            model_df[model_columns],
            on='date',
            how='inner'
        )

        # Create plots for each event
        for event_name, (start, end) in EVENT_DATES.items():
            event_label = start.split()[0]  # Get date part (YYYY-MM-DD)
            save_path = os.path.join(events_dir, f"{pavement}-{event_label}.png")

            fig = plot_event_for_pavement(
                combined,
                start_date=start,
                end_date=end,
                pavement_type=pavement,
                save_path=save_path
            )

    # Create multi-pavement comparison plots (7-day intervals)
    print("\n" + "-" * 80)
    print("STEP 3: Creating multi-pavement comparison plots (7-day intervals)")
    print("-" * 80)

    # Find overall date range from all pavements
    all_dates = []
    for pavement in model_results_dict:
        all_dates.extend(model_results_dict[pavement]['date'].tolist())

    if len(all_dates) > 0:
        start_overall = pd.to_datetime(min(all_dates))
        end_overall = pd.to_datetime(max(all_dates))

        print(f"\nData range: {start_overall} to {end_overall}")
        print("Creating 7-day interval plots...")

        # Loop through 7-day periods
        current_start = start_overall
        plot_count = 0

        while current_start < end_overall:
            current_end = current_start + pd.Timedelta(days=7)

            if current_end > end_overall:
                current_end = end_overall

            # Create filename
            period_str = f"{current_start.strftime('%Y%m%d')}_to_{current_end.strftime('%Y%m%d')}"
            save_path = os.path.join(comparison_dir, f"pavement_comparison_{period_str}.png")

            # Generate plot
            fig = plot_all_pavements_comparison(
                observed_data_dict,
                model_results_dict,
                current_start,
                current_end,
                save_path=save_path
            )

            plot_count += 1

            # Move to next period
            current_start = current_end

        print(f"\nCreated {plot_count} comparison plots")

    # Summary
    print("\n" + "=" * 80)
    print("FIGURE GENERATION COMPLETE")
    print("=" * 80)
    print(f"\nEvent plots saved to: {events_dir}")
    print(f"Comparison plots saved to: {comparison_dir}")
    print(f"\nTotal files created:")
    print(f"  Event plots: {len(os.listdir(events_dir))} files")
    print(f"  Comparison plots: {len(os.listdir(comparison_dir))} files")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    # You can specify which pavements to process
    # Default: all pavements
    main(pavements=['PICP', 'PGr', 'PA', 'PC', 'CP'])
