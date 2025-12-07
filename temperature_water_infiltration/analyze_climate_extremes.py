"""
Climate Change Extreme Temperature Analysis

Functions to analyze and visualize extreme temperature metrics from climate scenarios.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os


def calculate_hours_over_threshold(df, column_name, threshold=40.0):
    """
    Calculate average hours per year where temperature exceeds threshold.

    Args:
        df: DataFrame with temperature data (must have 'date' column)
        column_name: Name of temperature column
        threshold: Temperature threshold in °C

    Returns:
        Average hours per year above threshold
    """
    if column_name not in df.columns:
        return 0

    if 'date' not in df.columns:
        print("Warning: 'date' column not found, cannot calculate per-year average")
        return 0

    # Drop NaN values
    valid_data = df[[column_name, 'date']].dropna()

    if len(valid_data) == 0:
        return 0

    # Count hours over threshold
    hours_over_total = (valid_data[column_name] > threshold).sum()

    # Calculate number of years in dataset
    date_range = valid_data['date'].max() - valid_data['date'].min()
    years = date_range.days / 365.25  # Account for leap years

    if years == 0:
        return 0

    # Return average hours per year
    hours_per_year = hours_over_total / years

    return hours_per_year


def load_climate_scenario_data(pavement, scenario, base_dir=None):
    """
    Load climate scenario data for a specific pavement.

    Args:
        pavement: Pavement type (e.g., 'PA', 'PC', etc.)
        scenario: Climate scenario ('historical', 'ssp126', 'ssp585')
        base_dir: Base directory (default: script location)

    Returns:
        DataFrame with climate scenario results
    """
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))

    file_path = os.path.join(base_dir, 'output', 'climate_change',
                             f'{pavement}_scenario{scenario}.csv')

    try:
        df = pd.read_csv(file_path, parse_dates=['date'])
        return df
    except FileNotFoundError:
        print(f"Warning: File not found: {file_path}")
        return None


def analyze_hours_over_threshold(pavements, scenarios, threshold=40.0, base_dir=None):
    """
    Analyze hours over temperature threshold for all pavements and scenarios.

    Args:
        pavements: List of pavement types
        scenarios: List of scenarios
        threshold: Temperature threshold in °C
        base_dir: Base directory for data files

    Returns:
        Dictionary with results organized by scenario and temperature type
    """
    # Temperature columns to analyze
    temp_columns = {
        'Surface': 'surface_temp',
        'Surface Water': 'water_temp_surface',
        'Infiltrated Water': 'water_temp_infil'
    }

    results = {}

    for scenario in scenarios:
        results[scenario] = {}

        for temp_type, col_name in temp_columns.items():
            results[scenario][temp_type] = {}

            for pavement in pavements:
                df = load_climate_scenario_data(pavement, scenario, base_dir)

                if df is not None:
                    hours = calculate_hours_over_threshold(df, col_name, threshold)
                    results[scenario][temp_type][pavement] = hours
                else:
                    results[scenario][temp_type][pavement] = 0

    return results


def plot_hours_over_threshold(results, threshold=40.0, figsize=(12, 10), save_path=None):
    """
    Create multi-panel plot showing hours over threshold for each scenario and temperature type.

    Args:
        results: Dictionary from analyze_hours_over_threshold()
        threshold: Temperature threshold in °C
        figsize: Figure size tuple
        save_path: Path to save figure (optional)

    Returns:
        matplotlib figure object
    """
    # Define scenarios and their labels
    scenario_labels = {
        'historical': 'Historical',
        'ssp126': 'SSP1-2.6',
        'ssp585': 'SSP5-8.5'
    }

    scenarios = ['historical', 'ssp126', 'ssp585']
    temp_types = ['Surface', 'Surface Water', 'Infiltrated Water']

    # Create figure with subplots
    fig, axes = plt.subplots(3, 3, figsize=figsize)
    fig.suptitle(f'Average Hours over {threshold}°C per Year', fontsize=16, fontweight='bold')

    # Color for bars
    bar_color = '#1f77b4'

    for row, scenario in enumerate(scenarios):
        for col, temp_type in enumerate(temp_types):
            ax = axes[row, col]

            # Get data for this scenario and temperature type
            data = results[scenario][temp_type]

            # Sort pavements for consistent ordering
            if temp_type == 'Infiltrated Water':
                # Only include PA, PC, PGr, PICP
                desired_pavements = ['PA', 'PC', 'PGr', 'PICP']
                pavements = [p for p in desired_pavements if p in data]
            else:
                pavements = sorted(data.keys())

            hours = [data[p] for p in pavements]

            # Create bar plot
            x_pos = np.arange(len(pavements))
            bars = ax.bar(x_pos, hours, color=bar_color, edgecolor='black', linewidth=0.5)

            # Customize plot
            ax.set_xticks(x_pos)
            ax.set_xticklabels(pavements, fontsize=10)
            ax.set_ylabel('Hours / year', fontsize=10)
            ax.grid(axis='y', alpha=0.3, linestyle='--')

            # Add title for first row
            if row == 0:
                ax.set_title(temp_type, fontsize=12, fontweight='bold')

            # Add scenario label on leftmost column
            if col == 0:
                ax.text(-0.25, 0.5, scenario_labels[scenario],
                       transform=ax.transAxes, fontsize=12, fontweight='bold',
                       rotation=90, va='center', ha='right')

            # Set y-axis limits with some padding
            if max(hours) > 0:
                ax.set_ylim(0, max(hours) * 1.1)

            # Add value labels on bars for significant values
            for i, (bar, h) in enumerate(zip(bars, hours)):
                if h > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, h,
                           f'{h:.0f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")

    return fig


def generate_hours_over_threshold_report(results, threshold=40.0, output_file=None):
    """
    Generate text report of hours over threshold analysis.

    Args:
        results: Dictionary from analyze_hours_over_threshold()
        threshold: Temperature threshold in °C
        output_file: Optional file path to save report
    """
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append(f"HOURS OVER {threshold}°C ANALYSIS")
    report_lines.append("=" * 80)

    scenario_labels = {
        'historical': 'Historical',
        'ssp126': 'SSP1-2.6',
        'ssp585': 'SSP5-8.5'
    }

    for scenario in ['historical', 'ssp126', 'ssp585']:
        report_lines.append(f"\n{scenario_labels[scenario]}:")
        report_lines.append("-" * 80)

        for temp_type in ['Surface', 'Surface Water', 'Infiltrated Water']:
            data = results[scenario][temp_type]

            report_lines.append(f"\n  {temp_type}:")
            for pavement in sorted(data.keys()):
                hours = data[pavement]
                report_lines.append(f"    {pavement:<6}: {hours:>6.0f} hours/year")

    report_lines.append("\n" + "=" * 80)

    # Print to console
    report_text = "\n".join(report_lines)
    print(report_text)

    # Save to file if requested
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report_text)
        print(f"\nReport saved to: {output_file}")


def main():
    """Main function to run hours over threshold analysis."""

    print("=" * 80)
    print("CLIMATE CHANGE EXTREME TEMPERATURE ANALYSIS")
    print("Hours over 40°C per Year")
    print("=" * 80)

    # Define parameters
    pavements = ['CP', 'PA', 'PC', 'PICP', 'PGr']
    scenarios = ['historical', 'ssp126', 'ssp585']
    threshold = 40.0

    # Base directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, 'output', 'climate_analysis')
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nAnalyzing hours over {threshold}°C...")
    print(f"Pavements: {', '.join(pavements)}")
    print(f"Scenarios: {', '.join(scenarios)}")

    # Analyze hours over threshold
    results = analyze_hours_over_threshold(pavements, scenarios, threshold, base_dir)

    # Generate report
    print("\n" + "-" * 80)
    print("ANALYSIS RESULTS")
    print("-" * 80)

    report_file = os.path.join(output_dir, f'hours_over_{threshold}C_report.txt')
    generate_hours_over_threshold_report(results, threshold, report_file)

    # Create plot
    print("\n" + "-" * 80)
    print("CREATING PLOT")
    print("-" * 80)

    plot_file = os.path.join(output_dir, f'hours_over_{threshold}C_plot.png')
    fig = plot_hours_over_threshold(results, threshold, figsize=(12, 10), save_path=plot_file)

    plt.show()

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}")


def plot_precipitation_climate_change(
        base_dir=r"C:\Users\Artur\PycharmProjects\pavement_temperature_model\temperature_water_infiltration\input_data"
):
    """
    Creates a two-panel figure:
        (1) Cumulative precipitation (mm)
        (2) Monthly average precipitation (mm/day)

    Reads:
        climate_change_historical.csv
        climate_change_ssp126.csv
        climate_change_ssp585.csv

    Each CSV must include:
        date (tz-aware string)  e.g., '1980-12-31 18:00:00-06:00'
        pr   (precipitation, mm per hour)
    """

    scenarios = {
        "historical": "Historical (1980–present)",
        "ssp126": "SSP1-2.6 (2015–2050)",
        "ssp585": "SSP5-8.5 (2015–2050)"
    }

    colors = {
        "historical": "#1f77b4",
        "ssp126": "#ff7f0e",
        "ssp585": "#2ca02c"
    }

    cumulative_results = {}    # scenario -> (years, cumulative_mm)
    monthly_avg_results = {}   # scenario -> 12-element array (Jan–Dec, mm/day)

    for scen_key, scen_label in scenarios.items():
        file_path = os.path.join(base_dir, f"climate_change_{scen_key}.csv")

        # --- Read and parse date column (tz-aware) ---
        df = pd.read_csv(file_path)

        # Force proper datetime parsing
        df["date"] = pd.to_datetime(df["date"], utc=True, errors="raise")

        # Set as DateTimeIndex and sort
        df = df.set_index("date").sort_index()

        # Precip column in mm per hour
        precip = df["pr"]

        # ======================================================
        # 1) CUMULATIVE PRECIPITATION (mm) – top panel
        # ======================================================
        # Sum hourly precip to annual totals
        # 'YE' = calendar year end
        yearly_precip = precip.resample("YE").sum()

        # -------- Trim SSP datasets to match historical length -------
        if scen_key == "historical":
            hist_length = len(yearly_precip)

        elif scen_key in ["ssp126", "ssp585"]:
            try:
                yearly_precip = yearly_precip.iloc[:hist_length]
            except NameError:
                pass  # historical not yet processed

        # ---- Use "year index" starting at 0 ----
        years = np.arange(len(yearly_precip))

        # cumulative mm
        cumulative_mm = yearly_precip.cumsum().values
        cumulative_results[scen_key] = (years, cumulative_mm)

        # ======================================================
        # 2) MONTHLY AVERAGE PRECIPITATION (mm/day) – bottom
        # ======================================================
        # First: daily totals (sum of hourly mm)
        daily_precip = precip.resample("D").sum()   # mm/day

        # Then: average of daily totals for each month (mm/day)
        monthly_avg = daily_precip.groupby(daily_precip.index.month).mean()

        # Ensure we have 12 months (Jan–Dec); fill missing with 0
        monthly_avg = monthly_avg.reindex(range(1, 13), fill_value=0.0)

        monthly_avg_results[scen_key] = monthly_avg.values

    # ==========================================================
    #                    PLOTTING
    # ==========================================================
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 10))

    # ---------- Top panel: cumulative precip ----------
    for scen_key, (years, cumvals) in cumulative_results.items():
        ax1.plot(
            years, cumvals,
            label=scenarios[scen_key],
            color=colors[scen_key],
            linewidth=2,
            marker="o",
            markersize=4
        )

    ax1.set_title("Cumulative Precipitation", fontsize=14)
    ax1.set_xlabel("Years", fontsize=12)
    ax1.set_ylabel("Cumulative precipitation (mm)", fontsize=12)
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=10)

    # ---------- Bottom panel: monthly average precip ----------
    months = np.arange(1, 13)

    for scen_key, monthly_vals in monthly_avg_results.items():
        ax2.plot(
            months, monthly_vals,
            label=scenarios[scen_key],
            color=colors[scen_key],
            linewidth=2,
            marker="o",
            markersize=5
        )

    ax2.set_title("Monthly Average Precipitation", fontsize=14)
    ax2.set_xlabel("Month", fontsize=12)
    ax2.set_ylabel("Monthly average precipitation (mm/day)", fontsize=12)
    ax2.set_xticks(months)
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    return fig

def calculate_monthly_temperature_differences(pavements, base_dir=None):
    """
    Calculate monthly temperature differences between future scenarios and historical baseline.

    Args:
        pavements: List of pavement types (e.g., ['CP', 'PA', 'PC', 'PGr', 'PICP'])
        base_dir: Base directory (default: script location)

    Returns:
        Dictionary with structure:
        {
            'ssp126': {
                'surface_temp': {pavement: [12 monthly values]},
                'water_temp_surface': {pavement: [12 monthly values]},
                'water_temp_infil': {pavement: [12 monthly values]}
            },
            'ssp585': {...}
        }
    """
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))

    scenarios = ['historical', 'ssp126', 'ssp585']
    temp_columns = ['surface_temp', 'water_temp_surface', 'water_temp_infil']

    # Store monthly averages for each scenario, temperature type, and pavement
    monthly_averages = {}

    for scenario in scenarios:
        monthly_averages[scenario] = {}

        for temp_col in temp_columns:
            monthly_averages[scenario][temp_col] = {}

            for pavement in pavements:
                # Load data
                df = load_climate_scenario_data(pavement, scenario, base_dir)

                if df is not None and temp_col in df.columns:
                    # Extract month from date (date is already datetime from load_climate_scenario_data)
                    # Ensure it's datetime type if not already
                    if not pd.api.types.is_datetime64_any_dtype(df['date']):
                        df['date'] = pd.to_datetime(df['date'], utc=True)

                    df['month'] = df['date'].dt.month

                    # Calculate monthly average (group by month and take mean)
                    monthly_avg = df.groupby('month')[temp_col].mean()

                    # Ensure we have all 12 months
                    monthly_avg = monthly_avg.reindex(range(1, 13), fill_value=np.nan)

                    monthly_averages[scenario][temp_col][pavement] = monthly_avg.values
                else:
                    # If data not available, fill with NaN
                    monthly_averages[scenario][temp_col][pavement] = np.full(12, np.nan)

    # Calculate differences: future scenarios - historical
    differences = {}

    for future_scenario in ['ssp126', 'ssp585']:
        differences[future_scenario] = {}

        for temp_col in temp_columns:
            differences[future_scenario][temp_col] = {}

            for pavement in pavements:
                historical_vals = monthly_averages['historical'][temp_col][pavement]
                future_vals = monthly_averages[future_scenario][temp_col][pavement]

                # Calculate difference
                diff = future_vals - historical_vals
                differences[future_scenario][temp_col][pavement] = diff

    return differences


def save_temperature_differences_to_csv(pavements=None, base_dir=r"C:\Users\Artur\PycharmProjects\pavement_temperature_model\temperature_water_infiltration", save_path=None):
    """
    Compute monthly temperature differences once and save to CSV.

    Output CSV columns:
        scenario, temp_type, pavement, month, delta_T
    """

    if pavements is None:
        pavements = ['CP', 'PA', 'PC', 'PGr', 'PICP']

    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))

    # Compute the differences ONCE
    differences = calculate_monthly_temperature_differences(pavements, base_dir)

    rows = []

    for scenario in ['ssp126', 'ssp585']:
        for temp_type in ['surface_temp', 'water_temp_surface', 'water_temp_infil']:
            for pavement in pavements:

                deltas = differences[scenario][temp_type][pavement]

                for m in range(1, 13):
                    rows.append({
                        "scenario": scenario,
                        "temp_type": temp_type,
                        "pavement": pavement,
                        "month": m,
                        "delta_T": deltas[m-1]
                    })

    df_out = pd.DataFrame(rows)

    # Default save location
    if save_path is None:
        save_path = os.path.join(base_dir, "temperature_differences_monthly.csv")

    df_out.to_csv(save_path, index=False)
    print(f"\nSaved monthly temperature differences to:\n{save_path}\n")
    return save_path


def plot_climate_change_temperature_differences_from_csv(
        csv_path,
        pavements=None,
        figsize=(12, 10),
        save_path=None):

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    # ----------------------------
    # FONT SETTINGS (edit here)
    # ----------------------------
    TITLE_FONTSIZE = 14
    LABEL_FONTSIZE = 11
    TICK_FONTSIZE = 11
    LEGEND_FONTSIZE = 16

    # Pavement list
    if pavements is None:
        pavements = ['CP', 'PA', 'PC', 'PGr', 'PICP']

    df = pd.read_csv(csv_path)
    df = df[df["pavement"].isin(pavements)]

    # Colors and markers
    pavement_styles = {
        'CP':   {'color': '#1f77b4', 'marker': 'o', 'label': 'CP'},
        'PA':   {'color': '#ff7f0e', 'marker': 's', 'label': 'PA'},
        'PC':   {'color': '#2ca02c', 'marker': '^', 'label': 'PC'},
        'PGr':  {'color': '#d62728', 'marker': 'v', 'label': 'PGR'},
        'PICP': {'color': '#9467bd', 'marker': 'D', 'label': 'PICP'},
    }

    # Line styles for visibility
    line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]

    # Variable labels
    temp_types = [
        ('surface_temp', 'Surface temperature (Δ°C)'),
        ('water_temp_surface', 'Surface water temperature (Δ°C)'),
        ('water_temp_infil', 'Infiltrated water temperature (Δ°C)')
    ]

    scenario_labels = {'ssp126': 'SSP1-2.6', 'ssp585': 'SSP5-8.5'}

    fig, axes = plt.subplots(3, 2, figsize=figsize, sharex=True)
    months = np.arange(1, 13)

    # To collect legend items globally
    legend_handles = []

    for row, (temp_col, ylabel) in enumerate(temp_types):
        for col, scenario in enumerate(['ssp126', 'ssp585']):
            ax = axes[row, col]

            df_sub = df[(df["scenario"] == scenario) & (df["temp_type"] == temp_col)]

            for pi, pavement in enumerate(pavements):
                df_p = df_sub[df_sub["pavement"] == pavement]
                diff_values = df_p["delta_T"].values
                style = pavement_styles[pavement]

                handle = ax.plot(
                    months, diff_values,
                    color=style['color'],
                    marker=style['marker'],
                    markersize=6,
                    markerfacecolor='white',
                    markeredgecolor=style['color'],
                    linewidth=2.2,
                    linestyle=line_styles[pi],
                    alpha=0.95,
                    label=pavement
                )[0]

                # Save only once for legend
                if row == 0 and col == 0:
                    legend_handles.append(handle)

            # Zero line
            ax.axhline(0, color='black', linewidth=0.8, alpha=0.6)

            # ----------------------------
            # MATCH ORIGINAL GRID STYLE
            # ----------------------------
            ax.grid(True, alpha=0.35, linestyle=':', linewidth=0.8)

            # Background
            ax.set_facecolor('#f9f9f9')

            # Y-axis label
            if col == 0:
                ax.set_ylabel(ylabel, fontsize=LABEL_FONTSIZE)

            # X-axis label
            if row == 2:
                ax.set_xlabel("Month", fontsize=LABEL_FONTSIZE)

            # Formatting
            ax.set_xlim(0.5, 12.5)
            ax.set_xticks(months)
            ax.tick_params(labelsize=TICK_FONTSIZE)

            # Titles
            if row == 0:
                ax.set_title(scenario_labels[scenario],
                             fontsize=TITLE_FONTSIZE,
                             fontweight='bold')

    # ----------------------------
    # GLOBAL CENTERED BOTTOM LEGEND
    # ----------------------------
    fig.legend(
        handles=legend_handles,
        labels=[p for p in pavements],
        loc='lower center',
        fontsize=LEGEND_FONTSIZE,
        ncol=len(pavements),
        frameon=False,
        bbox_to_anchor=(0.5, 0.00)
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.96])

    if save_path:
        plt.savefig(save_path, dpi=350, bbox_inches='tight')
        print(f"Saved figure at: {save_path}")

    return fig


if __name__ == "__main__":
    # main()
    # plot_precipitation_climate_change()

    # Generate the climate change temperature difference plot
    output_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'output', 'climate_analysis'
    )
    os.makedirs(output_dir, exist_ok=True)

    save_path = os.path.join(output_dir, 'climate_change_temperature_differences.png')

    csv_path = r"C:\Users\Artur\PycharmProjects\pavement_temperature_model\temperature_water_infiltration\output\climate_analysis\temperature_differences_monthly.csv"
    print(f"Generating plot and saving to: {save_path}")
    plot_climate_change_temperature_differences_from_csv(csv_path)
    plt.show()
