"""
Analyze Evaporation Contribution to Permeable Pavement Cooling

This script analyzes evaporation data from the integrated model to:
1. Calculate cumulative evaporation over the monitoring period
2. Quantify evaporative cooling contribution during rainfall events

IMPORTANT: The model only calculates evaporation during rainfall events,
so comparisons are made ONLY during wet periods for fairness.

Author: Evaporation Analysis
Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Configuration
PAVEMENTS = ['CP', 'PICP', 'PGr', 'PA', 'PC']

# Physical constants for unit conversions
RHO_WATER = 1000  # kg/m³
L_LATENT = 2.45e6  # J/kg (latent heat of vaporization)

# Styling
COLOR_MAPPING = {
    'CP': (0.121, 0.467, 0.706),
    'PICP': (1.0, 0.498, 0.055),
    'PGr': (0.173, 0.627, 0.173),
    'PA': (0.580, 0.404, 0.741),
    'PC': (0.839, 0.153, 0.157),
}

PAVEMENT_NAMES = {
    "CP": "Conventional Pavement",
    "PC": "Permeable Concrete",
    "PICP": "Permeable Interlocking Concrete Pavers",
    "PA": "Porous Asphalt",
    "PGr": "Permeable Grid"
}


def convert_heat_flux_to_evaporation_rate(h_evap, dt_seconds):
    """
    Convert evaporation heat flux (W/m²) to evaporation depth (mm).

    Args:
        h_evap: Evaporation heat flux (W/m²)
        dt_seconds: Time step duration (seconds)

    Returns:
        Evaporation depth (mm) over the time step
    """
    # Energy removed by evaporation (J/m²) = h_evap (W/m²) × dt (s)
    energy_per_area = h_evap * dt_seconds

    # Mass of water evaporated (kg/m²) = energy / latent_heat
    mass_evaporated = energy_per_area / L_LATENT

    # Depth of water (m) = mass / density
    depth_m = mass_evaporated / RHO_WATER

    # Convert to mm
    depth_mm = depth_m * 1000

    return depth_mm


def analyze_evaporation(results_df, pavement_name, dt_seconds=300):
    """
    Analyze evaporation data from model results.

    Args:
        results_df: DataFrame with model results (must have h_evap, h_rad, h_conv, h_net columns)
        pavement_name: Name of pavement type
        dt_seconds: Time step in seconds (default 300 = 5 min)

    Returns:
        Dictionary with evaporation statistics
    """
    # Identify periods with active evaporation (h_evap > 0)
    wet_periods = results_df['h_evap'] > 0.01  # Small threshold to filter numerical noise

    # Calculate evaporation depth for each timestep (mm)
    evap_depth = convert_heat_flux_to_evaporation_rate(results_df['h_evap'], dt_seconds)

    # Statistics during ALL periods
    total_evaporation_mm = evap_depth.sum()

    # Statistics during WET periods only (when evaporation is active)
    n_wet_timesteps = wet_periods.sum()

    if n_wet_timesteps > 0:
        # Energy balance during wet periods only
        h_evap_wet = results_df.loc[wet_periods, 'h_evap']
        h_rad_wet = results_df.loc[wet_periods, 'h_rad']
        h_conv_wet = results_df.loc[wet_periods, 'h_conv']
        h_net_wet = results_df.loc[wet_periods, 'h_net']

        # Calculate mean fluxes during wet periods
        mean_h_evap = h_evap_wet.mean()
        mean_h_rad = h_rad_wet.mean()
        mean_h_conv = h_conv_wet.mean()
        mean_h_net = h_net_wet.mean()

        # Percentage contribution during wet periods
        # Total heat dissipation = h_evap + h_conv (both remove heat from surface)
        # h_rad is NET radiation (can be positive or negative)
        # Fair comparison: evaporation as % of total heat removal processes
        mean_heat_removal = mean_h_evap + mean_h_conv  # W/m²

        if mean_heat_removal > 0:
            evap_contribution_pct = (mean_h_evap / mean_heat_removal) * 100
        else:
            evap_contribution_pct = 0

        # Alternative calculation: evaporation as % of net heat flux balance
        # This shows what fraction of incoming energy is dissipated via evaporation
        if abs(mean_h_net) > 0.01:
            evap_of_net_pct = (mean_h_evap / abs(mean_h_rad)) * 100  # % of radiation
        else:
            evap_of_net_pct = 0

        # Peak values
        peak_h_evap = h_evap_wet.max()

        # Evaporation rate during wet periods (mm/day)
        # Only calculate for periods where evaporation is active
        evap_rate_wet_mm_day = (evap_depth[wet_periods].sum() / n_wet_timesteps) * (86400 / dt_seconds)

    else:
        # No wet periods (shouldn't happen for permeable pavements, but handle it)
        mean_h_evap = 0
        mean_h_rad = 0
        mean_h_conv = 0
        mean_h_net = 0
        evap_contribution_pct = 0
        evap_of_net_pct = 0
        peak_h_evap = 0
        evap_rate_wet_mm_day = 0

    # Compile statistics
    stats = {
        'pavement': pavement_name,
        'total_evaporation_mm': total_evaporation_mm,
        'n_wet_timesteps': n_wet_timesteps,
        'wet_hours': n_wet_timesteps * dt_seconds / 3600,
        'mean_h_evap_wet': mean_h_evap,
        'mean_h_rad_wet': mean_h_rad,
        'mean_h_conv_wet': mean_h_conv,
        'mean_h_net_wet': mean_h_net,
        'peak_h_evap': peak_h_evap,
        'evap_contribution_pct': evap_contribution_pct,
        'evap_of_radiation_pct': evap_of_net_pct,
        'evap_rate_wet_mm_day': evap_rate_wet_mm_day,
    }

    return stats


def create_summary_table(all_stats):
    """
    Create a formatted summary table of evaporation statistics.

    Args:
        all_stats: List of dictionaries with statistics for each pavement

    Returns:
        pandas DataFrame with formatted statistics
    """
    df = pd.DataFrame(all_stats)

    # Reorder columns for presentation
    column_order = [
        'pavement',
        'total_evaporation_mm',
        'wet_hours',
        'evap_rate_wet_mm_day',
        'mean_h_evap_wet',
        'peak_h_evap',
        'evap_contribution_pct',
        'evap_of_radiation_pct',
    ]

    df = df[column_order]

    # Rename columns for clarity
    df.columns = [
        'Pavement',
        'Total Evap (mm)',
        'Active Evap Hours',
        'Mean Rate During Rain (mm/day)',
        'Mean h_evap (W/m²)',
        'Peak h_evap (W/m²)',
        'Evap % of Heat Removal',
        'Evap % of Radiation',
    ]

    return df


def plot_cumulative_evaporation(all_stats, save_path=None):
    """
    Create bar chart of cumulative evaporation for all pavements.

    Args:
        all_stats: List of dictionaries with statistics
        save_path: Path to save figure (optional)
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    pavements = [s['pavement'] for s in all_stats]
    evap_totals = [s['total_evaporation_mm'] for s in all_stats]
    colors = [COLOR_MAPPING.get(p, (0.5, 0.5, 0.5)) for p in pavements]

    bars = ax.bar(pavements, evap_totals, color=colors, edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for bar, value in zip(bars, evap_totals):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.1f}',
                ha='center', va='bottom', fontsize=14, fontweight='bold')

    ax.set_ylabel('Cumulative Evaporation (mm)', fontsize=16, fontweight='bold')
    ax.set_xlabel('Pavement Type', fontsize=16, fontweight='bold')
    ax.set_title('Total Evaporation During Rainfall Events\n(Monitoring Period: May 2023 - July 2024)',
                 fontsize=18, fontweight='bold')
    ax.tick_params(axis='both', labelsize=14)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved cumulative evaporation plot: {save_path}")

    plt.close()

    return fig


def plot_energy_partitioning(all_stats, save_path=None):
    """
    Create bar chart showing evaporation contribution to heat removal during wet periods.

    Args:
        all_stats: List of dictionaries with statistics
        save_path: Path to save figure (optional)
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    pavements = [s['pavement'] for s in all_stats]
    evap_pct = [s['evap_contribution_pct'] for s in all_stats]
    conv_pct = [100 - s['evap_contribution_pct'] for s in all_stats]

    x = np.arange(len(pavements))
    width = 0.6

    # Stacked bars
    colors_evap = [COLOR_MAPPING.get(p, (0.5, 0.5, 0.5)) for p in pavements]
    colors_conv = [(c[0]*0.6, c[1]*0.6, c[2]*0.6) for c in colors_evap]  # Darker shade

    bars1 = ax.bar(x, evap_pct, width, label='Latent Heat (Evaporation)',
                   color=colors_evap, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x, conv_pct, width, bottom=evap_pct, label='Sensible Heat (Convection)',
                   color=colors_conv, edgecolor='black', linewidth=1.5)

    # Add percentage labels
    for i, (bar1, bar2, e_pct, c_pct) in enumerate(zip(bars1, bars2, evap_pct, conv_pct)):
        # Evaporation label
        if e_pct > 5:  # Only show if large enough
            ax.text(bar1.get_x() + bar1.get_width()/2., e_pct/2,
                    f'{e_pct:.1f}%',
                    ha='center', va='center', fontsize=12, fontweight='bold', color='white')

        # Convection label
        if c_pct > 5:  # Only show if large enough
            ax.text(bar2.get_x() + bar2.get_width()/2., e_pct + c_pct/2,
                    f'{c_pct:.1f}%',
                    ha='center', va='center', fontsize=12, fontweight='bold', color='white')

    ax.set_ylabel('Percentage of Heat Removal (%)', fontsize=16, fontweight='bold')
    ax.set_xlabel('Pavement Type', fontsize=16, fontweight='bold')
    ax.set_title('Energy Partitioning During Rainfall Events\n(Latent vs. Sensible Heat Dissipation)',
                 fontsize=18, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(pavements)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_ylim(0, 100)
    ax.legend(loc='upper right', fontsize=14)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved energy partitioning plot: {save_path}")

    plt.close()

    return fig


def main():
    """
    Main function to run evaporation analysis for all pavements.
    """
    print("=" * 80)
    print("EVAPORATION CONTRIBUTION ANALYSIS")
    print("=" * 80)
    print("\nIMPORTANT: Model calculates evaporation ONLY during rainfall events.")
    print("All comparisons are made during wet periods for fairness.\n")

    # Setup paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, "output", "model_results")
    output_dir = os.path.join(base_dir, "output", "evaporation_analysis")
    os.makedirs(output_dir, exist_ok=True)

    # Check if model results exist
    if not os.path.exists(results_dir):
        print(f"ERROR: Model results directory not found: {results_dir}")
        print("\nPlease run create_publication_figures.py first to generate model results.")
        return

    # Analyze all pavements
    all_stats = []

    print("Analyzing evaporation for each pavement type:")
    print("-" * 80)

    for pavement in PAVEMENTS:
        results_file = os.path.join(results_dir, f"{pavement}_model_results.csv")

        if not os.path.exists(results_file):
            print(f"  {pavement}: SKIPPED (no results file)")
            continue

        print(f"  {pavement}: Loading data...")

        try:
            # Load results
            df = pd.read_csv(results_file, parse_dates=['date'])

            # Analyze evaporation
            stats = analyze_evaporation(df, pavement, dt_seconds=300)
            all_stats.append(stats)

            # Print summary
            print(f"    - Total evaporation: {stats['total_evaporation_mm']:.2f} mm")
            print(f"    - Active hours: {stats['wet_hours']:.1f} h")
            print(f"    - Evap contribution: {stats['evap_contribution_pct']:.1f}% of heat removal")

        except Exception as e:
            print(f"    ERROR: {e}")
            continue

    if len(all_stats) == 0:
        print("\nNo results to analyze. Exiting.")
        return

    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    # Create summary table
    summary_df = create_summary_table(all_stats)
    print("\n" + summary_df.to_string(index=False))

    # Save table to CSV
    table_path = os.path.join(output_dir, "evaporation_summary.csv")
    summary_df.to_csv(table_path, index=False)
    print(f"\nSaved summary table: {table_path}")

    # Save detailed stats to CSV
    detailed_path = os.path.join(output_dir, "evaporation_detailed_stats.csv")
    pd.DataFrame(all_stats).to_csv(detailed_path, index=False)
    print(f"Saved detailed statistics: {detailed_path}")

    print("\n" + "=" * 80)
    print("CREATING FIGURES")
    print("=" * 80)

    # Create figures
    fig1_path = os.path.join(output_dir, "cumulative_evaporation.png")
    plot_cumulative_evaporation(all_stats, save_path=fig1_path)

    fig2_path = os.path.join(output_dir, "energy_partitioning_wet_periods.png")
    plot_energy_partitioning(all_stats, save_path=fig2_path)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nOutput directory: {output_dir}")
    print("\nKey Findings:")
    print("-" * 80)

    # Sort by evaporation total
    sorted_stats = sorted(all_stats, key=lambda x: x['total_evaporation_mm'], reverse=True)

    print(f"1. Highest cumulative evaporation: {sorted_stats[0]['pavement']} "
          f"({sorted_stats[0]['total_evaporation_mm']:.1f} mm)")
    print(f"2. Lowest cumulative evaporation: {sorted_stats[-1]['pavement']} "
          f"({sorted_stats[-1]['total_evaporation_mm']:.1f} mm)")

    # Sort by contribution percentage
    sorted_by_contrib = sorted(all_stats, key=lambda x: x['evap_contribution_pct'], reverse=True)

    print(f"3. Highest evaporative cooling contribution: {sorted_by_contrib[0]['pavement']} "
          f"({sorted_by_contrib[0]['evap_contribution_pct']:.1f}% of heat removal during rain)")
    print(f"4. Lowest evaporative cooling contribution: {sorted_by_contrib[-1]['pavement']} "
          f"({sorted_by_contrib[-1]['evap_contribution_pct']:.1f}% of heat removal during rain)")

    # Average for permeable vs conventional
    permeable_stats = [s for s in all_stats if s['pavement'] != 'CP']
    cp_stats = [s for s in all_stats if s['pavement'] == 'CP']

    if permeable_stats and cp_stats:
        avg_perm_evap = np.mean([s['total_evaporation_mm'] for s in permeable_stats])
        avg_perm_contrib = np.mean([s['evap_contribution_pct'] for s in permeable_stats])

        print(f"\n5. Average permeable pavement evaporation: {avg_perm_evap:.1f} mm")
        print(f"   Average CP evaporation: {cp_stats[0]['total_evaporation_mm']:.1f} mm")
        print(f"   Difference: {avg_perm_evap - cp_stats[0]['total_evaporation_mm']:.1f} mm "
              f"({((avg_perm_evap / cp_stats[0]['total_evaporation_mm']) - 1) * 100:.1f}% higher)")

        print(f"\n6. Average permeable pavement evap contribution: {avg_perm_contrib:.1f}%")
        print(f"   CP evap contribution: {cp_stats[0]['evap_contribution_pct']:.1f}%")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()