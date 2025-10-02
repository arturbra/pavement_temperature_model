"""Utilities for climate change applications of the permeable pavement model.

This module operationalises the analysis plan outlined for the manuscript by
providing a high-level workflow that can:

* Load historical or scenario-based meteorological forcings.
* Execute the surface energy balance model for multiple pavement types.
* Compute temperature, flux, evaporation, and storm-event metrics that support
  manuscript-ready discussion points (e.g., heat mitigation, thermal pollution).
* Aggregate the diagnostics across user-defined planning windows and optional
  climate scenarios (e.g., SSP126 vs SSP585).

The module is intentionally modular—each function can be imported in notebooks
or scripts for bespoke plots—while the command line interface enables quick
generation of summary tables for entire scenarios.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import temperature_model


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class TimeWindow:
    """Simple container for analysis windows."""

    label: str
    start: pd.Timestamp
    end: pd.Timestamp


@dataclass
class EventSummary:
    """Storm-event level diagnostics."""

    start: pd.Timestamp
    end: pd.Timestamp
    depth_mm: float
    peak_intensity_mm_per_hr: float
    infiltration_exceedance: bool
    mean_surface_temp_c: Optional[float]
    mean_water_temp_c: Optional[float]
    max_water_temp_c: Optional[float]
    thermal_energy_j: Optional[float]
    thermal_energy_kwh: Optional[float]

    def to_dict(self) -> Dict[str, Optional[float]]:
        """Return a JSON-serialisable representation."""

        return {
            "start": None if pd.isna(self.start) else self.start.isoformat(),
            "end": None if pd.isna(self.end) else self.end.isoformat(),
            "depth_mm": float(self.depth_mm) if self.depth_mm is not None else None,
            "peak_intensity_mm_per_hr": float(self.peak_intensity_mm_per_hr)
            if self.peak_intensity_mm_per_hr is not None
            else None,
            "infiltration_exceedance": bool(self.infiltration_exceedance),
            "mean_surface_temp_c": float(self.mean_surface_temp_c)
            if self.mean_surface_temp_c is not None
            else None,
            "mean_water_temp_c": float(self.mean_water_temp_c)
            if self.mean_water_temp_c is not None
            else None,
            "max_water_temp_c": float(self.max_water_temp_c)
            if self.max_water_temp_c is not None
            else None,
            "thermal_energy_j": float(self.thermal_energy_j)
            if self.thermal_energy_j is not None
            else None,
            "thermal_energy_kwh": float(self.thermal_energy_kwh)
            if self.thermal_energy_kwh is not None
            else None,
        }


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


REQUIRED_COLUMNS = {
    "date",
    "AirTemperature",
    "RelativeHumidity",
    "DewPoint",
    "CloudCoverage",
    "WindSpeed",
    "SolarRadiation",
}


COLUMN_ALIASES: Mapping[str, Sequence[str]] = {
    "date": ("time", "datetime", "timestamp"),
    "AirTemperature": (
        "tas",
        "air_temperature",
        "temperature",
        "temp",
        "tair",
        "t2m",
    ),
    "RelativeHumidity": ("hurs", "relativehumidity", "relative_humidity", "rh", "hur"),
    "DewPoint": ("dewpoint", "dew_point", "td", "td2m", "dewpointtemperature"),
    "CloudCoverage": (
        "clt",
        "cloudcover",
        "cloud_coverage",
        "cloud_fraction",
        "cloudiness",
        "tcc",
    ),
    "WindSpeed": (
        "sfcwind",
        "wind_speed",
        "windspeed",
        "wind10m",
        "windspeed_10m",
        "wspd",
    ),
    "SolarRadiation": (
        "rsds",
        "shortwave_radiation",
        "solar_radiation",
        "swdown",
        "swd",
        "solar",
    ),
    "Rainfall": (
        "pr",
        "precipitation",
        "precip",
        "rainfall",
        "rain",
        "rainrate",
        "tp",
    ),
}


def _harmonise_forcing_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename common CMIP/ISIMIP fields and convert units when necessary."""

    df = df.copy()
    # Map lowercase column names for robust matching
    lower_to_original = {col.lower(): col for col in df.columns}
    rename_map: Dict[str, str] = {}
    alias_source: Dict[str, str] = {}

    for target, aliases in COLUMN_ALIASES.items():
        if target in df.columns:
            alias_source[target] = target
            continue
        for alias in aliases:
            original = lower_to_original.get(alias.lower())
            if original is not None:
                rename_map[original] = target
                alias_source[target] = original
                break

    if rename_map:
        df = df.rename(columns=rename_map)

    # Convert temperature units where necessary
    if "AirTemperature" in df.columns:
        temps = df["AirTemperature"].astype(float)
        source = alias_source.get("AirTemperature", "")
        if source.lower() == "tas" or temps.mean() > 200:
            df["AirTemperature"] = temps - 273.15
        elif temps.max() > 130:  # likely Fahrenheit
            df["AirTemperature"] = (temps - 32.0) / 1.8

    # Relative humidity expressed as percentage in most datasets
    if "RelativeHumidity" in df.columns:
        rh = df["RelativeHumidity"].astype(float)
        source = alias_source.get("RelativeHumidity", "")
        if source.lower() == "hurs" or rh.max() > 1.5:
            df["RelativeHumidity"] = rh / 100.0

    # Cloud coverage also frequently reported as percentage
    if "CloudCoverage" in df.columns:
        cc = df["CloudCoverage"].astype(float)
        source = alias_source.get("CloudCoverage", "")
        if source.lower() in {"clt", "tcc"} or cc.max() > 1.0:
            df["CloudCoverage"] = (cc / 100.0).clip(0.0, 1.0)

    # Wind speed is typically already in m/s; include a defensive conversion for km/h
    if "WindSpeed" in df.columns:
        wind = df["WindSpeed"].astype(float)
        if wind.max() > 60.0:
            df["WindSpeed"] = wind / 3.6

    # Rainfall conversions: if sourced from CMIP "pr" (kg m-2 s-1), convert to mm/hr
    if "Rainfall" in df.columns:
        rain = df["Rainfall"].astype(float)
        source = alias_source.get("Rainfall", "")
        if source.lower() == "pr":
            df["Rainfall"] = rain * 3600.0

    return df


def _parse_date_column(df: pd.DataFrame, column: str = "date") -> pd.Series:
    """Parse the datetime column and ensure timezone-naive UTC offsets."""

    if np.issubdtype(df[column].dtype, np.datetime64):
        if getattr(df[column].dt, "tz", None) is None:
            df[column] = df[column].dt.tz_localize("UTC")
        else:
            df[column] = df[column].dt.tz_convert("UTC")
    else:
        df[column] = pd.to_datetime(df[column], utc=True, errors="coerce")
    if df[column].isna().any():
        raise ValueError("Invalid timestamps detected in forcing data")
    # Convert to timezone-naive in UTC to avoid downstream tz-math issues
    return df[column].dt.tz_localize(None)


def _ensure_meteorological_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Fill or derive missing meteorological fields required by the model."""

    df = df.copy()
    df["date"] = _parse_date_column(df)

    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(
            "Missing required columns for the pavement temperature model: "
            f"{sorted(missing)}"
        )

    # Convert relative humidity from percent if necessary
    if df["RelativeHumidity"].max() > 1.5:
        df["RelativeHumidity"] = df["RelativeHumidity"] / 100.0

    # Cloud coverage defaults to 0.5 when unavailable or NaN
    df["CloudCoverage"] = df["CloudCoverage"].fillna(0.5)

    # Dew point can be derived from RH + temperature if required
    if df["DewPoint"].isna().any():
        temp = df["AirTemperature"].astype(float)
        rh = df["RelativeHumidity"].clip(lower=1e-6, upper=0.999999)
        a = 17.625
        b = 243.04
        alpha = np.log(rh) + (a * temp) / (b + temp)
        df["DewPoint"] = (b * alpha) / (a - alpha)

    # Wind speed occasionally arrives as knots; assume m/s unless high values
    if df["WindSpeed"].max() > 60:  # heuristically identify km/h inputs
        df["WindSpeed"] = df["WindSpeed"] / 3.6

    if "Rainfall" not in df.columns:
        df["Rainfall"] = 0.0
    else:
        df["Rainfall"] = df["Rainfall"].fillna(0.0)

    return df.sort_values("date").reset_index(drop=True)


def _resample_to_frequency(df: pd.DataFrame, frequency: Optional[str]) -> pd.DataFrame:
    """Resample forcing data to the requested frequency if provided."""

    if not frequency:
        return df

    df = df.set_index("date")
    df = df.resample(frequency).interpolate("time")
    df = df.reset_index()
    return df


def load_forcing_data(
    path: Path,
    frequency: Optional[str] = None,
    standardised_output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Load a forcing file (CSV/Parquet) and ensure required columns exist."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    df = _harmonise_forcing_columns(df)
    df = _ensure_meteorological_columns(df)
    df = _resample_to_frequency(df, frequency)

    if standardised_output_path is not None:
        standardised_output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(standardised_output_path, index=False)

    return df


def estimate_time_step_seconds(date_series: pd.Series) -> float:
    """Estimate the representative timestep in seconds from a datetime series."""

    diffs = date_series.sort_values().diff().dt.total_seconds().dropna()
    if diffs.empty:
        return 0.0
    return float(diffs.median())


def compute_temperature_metrics(
    sim_df: pd.DataFrame,
    thresholds: Sequence[float],
    column: str = "surface_temp",
) -> Dict[str, float]:
    """Summarise surface temperature statistics and degree-hours."""

    if column not in sim_df:
        raise KeyError(f"Column '{column}' not available in simulation output")

    df = sim_df.dropna(subset=["date", column]).sort_values("date")
    if df.empty:
        return {}

    temps = df[column].astype(float)
    timestep_hours = estimate_time_step_seconds(df["date"]) / 3600.0
    timestep_hours = timestep_hours if timestep_hours > 0 else 0.0

    metrics: Dict[str, float] = {
        "mean": float(temps.mean()),
        "median": float(temps.median()),
        "std": float(temps.std(ddof=0)),
        "p95": float(temps.quantile(0.95)),
        "p99": float(temps.quantile(0.99)),
        "max": float(temps.max()),
        "min": float(temps.min()),
    }

    if timestep_hours > 0:
        above = {}
        for thr in thresholds:
            exceedance = (temps - thr).clip(lower=0)
            degree_hours = float(exceedance.sum() * timestep_hours)
            hours_above = float((exceedance > 0).sum() * timestep_hours)
            above[f"deg_hours_above_{thr}"] = degree_hours
            above[f"hours_above_{thr}"] = hours_above
        metrics.update(above)

    return metrics


def compute_flux_statistics(sim_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Compute descriptive statistics for each energy flux term."""

    flux_columns = [
        "h_s",
        "h_li",
        "h_l0",
        "h_rad",
        "h_evap",
        "h_conv",
        "h_r0",
        "h_net",
    ]

    df = sim_df.dropna(subset=["date"]).sort_values("date")
    timestep_seconds = estimate_time_step_seconds(df["date"])

    stats: Dict[str, Dict[str, float]] = {}
    for col in flux_columns:
        if col not in df:
            continue
        series = df[col].dropna().astype(float)
        if series.empty:
            continue
        stat = {
            "mean": float(series.mean()),
            "median": float(series.median()),
            "p95": float(series.quantile(0.95)),
            "min": float(series.min()),
            "max": float(series.max()),
        }
        if timestep_seconds > 0:
            energy = float((series * timestep_seconds).sum())
            stat["cumulative_energy_j_per_m2"] = energy
        stats[col] = stat

    return stats


def compute_evaporation_metrics(
    sim_df: pd.DataFrame,
    latent_heat_j_kg: float = 2.45e6,
) -> Dict[str, float]:
    """Convert latent heat flux into equivalent evaporation depth."""

    if "h_evap" not in sim_df:
        return {}

    df = sim_df.dropna(subset=["date", "h_evap"]).sort_values("date")
    if df.empty:
        return {}

    timestep_seconds = estimate_time_step_seconds(df["date"])
    if timestep_seconds <= 0:
        return {}

    latent_flux = df["h_evap"].clip(lower=0)
    total_latent_energy = float((latent_flux * timestep_seconds).sum())
    evaporation_mm = total_latent_energy / latent_heat_j_kg

    return {
        "total_latent_energy_j_per_m2": total_latent_energy,
        "evaporation_equivalent_mm": evaporation_mm,
        "mean_latent_flux_w_m2": float(latent_flux.mean()),
        "max_latent_flux_w_m2": float(latent_flux.max()),
    }


def run_pavement_simulation(
    forcing_df: pd.DataFrame,
    parameter_file: Path,
) -> pd.DataFrame:
    """Wrapper around the full surface energy balance model."""

    parameter_file = Path(parameter_file)
    if not parameter_file.exists():
        raise FileNotFoundError(parameter_file)

    required_for_model = [
        "date",
        "AirTemperature",
        "RelativeHumidity",
        "DewPoint",
        "CloudCoverage",
        "WindSpeed",
        "SolarRadiation",
        "Rainfall",
    ]

    missing = [col for col in required_for_model if col not in forcing_df.columns]
    if missing:
        raise ValueError(f"Missing forcing columns required by the model: {missing}")

    sim_df = forcing_df[required_for_model].copy()
    return temperature_model.model_pavement_temperature(sim_df, str(parameter_file))


def summarise_precipitation_events(
    sim_df: pd.DataFrame,
    forcing_df: pd.DataFrame,
    area_m2: float,
    infiltration_capacity_mm_per_hr: Optional[float] = None,
    reference_water_temp_c: float = 20.0,
    min_event_depth_mm: float = 0.25,
    interevent_hours: float = 2.0,
) -> List[EventSummary]:
    """Identify rainfall events and estimate associated thermal loads."""

    df = forcing_df[["date", "Rainfall"]].copy()
    df["date"] = _parse_date_column(df, "date")
    df["Rainfall"] = df["Rainfall"].fillna(0.0).astype(float)
    df = df.sort_values("date").reset_index(drop=True)

    if df.empty:
        return []

    dt_hours = df["date"].diff().dt.total_seconds() / 3600.0
    dt_hours.iloc[0] = dt_hours.iloc[1] if len(dt_hours) > 1 else 1.0

    events: List[EventSummary] = []
    start_idx: Optional[int] = None
    end_idx: Optional[int] = None
    depth_mm = 0.0
    peak_intensity = 0.0
    hours_since_rain = np.inf

    for idx, rainfall in enumerate(df["Rainfall"].values):
        step_hours = float(dt_hours.iloc[idx]) if not np.isnan(dt_hours.iloc[idx]) else 0.0

        if rainfall > 0:
            if start_idx is None:
                start_idx = idx
                depth_mm = 0.0
                peak_intensity = 0.0
            end_idx = idx
            depth_mm += float(rainfall)
            if step_hours > 0:
                peak_intensity = max(peak_intensity, rainfall / step_hours)
            hours_since_rain = 0.0
        else:
            if start_idx is not None:
                hours_since_rain += step_hours
                if hours_since_rain >= interevent_hours:
                    _finalise_event(
                        events,
                        df,
                        start_idx,
                        end_idx,
                        depth_mm,
                        peak_intensity,
                        sim_df,
                        area_m2,
                        infiltration_capacity_mm_per_hr,
                        reference_water_temp_c,
                        min_event_depth_mm,
                    )
                    start_idx = None
                    end_idx = None
                    depth_mm = 0.0
                    peak_intensity = 0.0
                    hours_since_rain = np.inf

    # Final event if the loop ended during rainfall
    if start_idx is not None:
        _finalise_event(
            events,
            df,
            start_idx,
            end_idx,
            depth_mm,
            peak_intensity,
            sim_df,
            area_m2,
            infiltration_capacity_mm_per_hr,
            reference_water_temp_c,
            min_event_depth_mm,
        )

    return events


def _finalise_event(
    events: List[EventSummary],
    rainfall_df: pd.DataFrame,
    start_idx: Optional[int],
    end_idx: Optional[int],
    depth_mm: float,
    peak_intensity: float,
    sim_df: pd.DataFrame,
    area_m2: float,
    infiltration_capacity_mm_per_hr: Optional[float],
    reference_water_temp_c: float,
    min_event_depth_mm: float,
) -> None:
    """Append an :class:`EventSummary` if the rainfall event is valid."""

    if start_idx is None or end_idx is None:
        return
    if depth_mm < min_event_depth_mm:
        return

    start_time = rainfall_df.loc[start_idx, "date"]
    end_time = rainfall_df.loc[end_idx, "date"]

    mask = (sim_df["date"] >= start_time) & (sim_df["date"] <= end_time)
    event_sim = sim_df.loc[mask]

    surface_temp = event_sim.get("surface_temp", pd.Series(dtype=float)).dropna()
    water_temp = event_sim.get("water_temp", pd.Series(dtype=float)).dropna()

    mean_surface = float(surface_temp.mean()) if not surface_temp.empty else None
    mean_water = float(water_temp.mean()) if not water_temp.empty else None
    max_water = float(water_temp.max()) if not water_temp.empty else None

    rho_water = 1000.0  # kg/m3
    cp_water = 4186.0  # J/(kg·K)
    volume_m3 = (depth_mm / 1000.0) * area_m2

    energy_j: Optional[float]
    if mean_water is None:
        energy_j = None
    else:
        delta_t = mean_water - reference_water_temp_c
        energy_j = rho_water * cp_water * volume_m3 * delta_t

    event = EventSummary(
        start=start_time,
        end=end_time,
        depth_mm=depth_mm,
        peak_intensity_mm_per_hr=peak_intensity,
        infiltration_exceedance=
        bool(peak_intensity > infiltration_capacity_mm_per_hr)
        if infiltration_capacity_mm_per_hr is not None
        else False,
        mean_surface_temp_c=mean_surface,
        mean_water_temp_c=mean_water,
        max_water_temp_c=max_water,
        thermal_energy_j=energy_j,
        thermal_energy_kwh=energy_j / 3.6e6 if energy_j is not None else None,
    )

    events.append(event)


def split_time_windows(
    df: pd.DataFrame,
    windows: Sequence[TimeWindow],
) -> Dict[str, pd.DataFrame]:
    """Return a dictionary of dataframe slices for each analysis window."""

    slices: Dict[str, pd.DataFrame] = {}
    for window in windows:
        mask = (df["date"] >= window.start) & (df["date"] < window.end)
        slices[window.label] = df.loc[mask].copy()
    return slices


def analyse_scenario(
    forcing_df: pd.DataFrame,
    scenario_name: str,
    pavement_parameters: Mapping[str, Path],
    windows: Sequence[TimeWindow],
    thresholds: Sequence[float],
    area_m2: float,
    infiltration_capacities: Optional[Mapping[str, float]] = None,
    reference_water_temp_c: float = 20.0,
) -> Dict[str, Dict[str, object]]:
    """Run the climate scenario for all pavements and summarise diagnostics."""

    results: Dict[str, Dict[str, object]] = {}

    for pavement, param_file in pavement_parameters.items():
        simulation = run_pavement_simulation(forcing_df, param_file)
        simulation["date"] = _parse_date_column(simulation, "date")

        metrics = {
            "temperature_metrics": compute_temperature_metrics(simulation, thresholds),
            "flux_metrics": compute_flux_statistics(simulation),
            "evaporation_metrics": compute_evaporation_metrics(simulation),
            "event_metrics": [
                event.to_dict()
                for event in summarise_precipitation_events(
                    simulation,
                    forcing_df,
                    area_m2=area_m2,
                    infiltration_capacity_mm_per_hr=
                    infiltration_capacities.get(pavement)
                    if infiltration_capacities and pavement in infiltration_capacities
                    else None,
                    reference_water_temp_c=reference_water_temp_c,
                )
            ],
        }

        if windows:
            window_metrics: Dict[str, Dict[str, object]] = {}
            sim_windows = split_time_windows(simulation, windows)
            forcing_windows = split_time_windows(forcing_df, windows)
            for label in sim_windows:
                sim_slice = sim_windows[label]
                forcing_slice = forcing_windows[label]
                window_metrics[label] = {
                    "temperature_metrics": compute_temperature_metrics(sim_slice, thresholds),
                    "flux_metrics": compute_flux_statistics(sim_slice),
                    "evaporation_metrics": compute_evaporation_metrics(sim_slice),
                    "event_metrics": [
                        event.to_dict()
                        for event in summarise_precipitation_events(
                            sim_slice,
                            forcing_slice,
                            area_m2=area_m2,
                            infiltration_capacity_mm_per_hr=
                            infiltration_capacities.get(pavement)
                            if infiltration_capacities and pavement in infiltration_capacities
                            else None,
                            reference_water_temp_c=reference_water_temp_c,
                        )
                    ],
                }
            metrics["windows"] = window_metrics

        results[pavement] = metrics

    return {scenario_name: results}


# ---------------------------------------------------------------------------
# Command line interface
# ---------------------------------------------------------------------------


def _default_parameter_mapping(
    pavements: Iterable[str],
    parameters_dir: Path,
) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    for pavement in pavements:
        filename = parameters_dir / f"parameters_{pavement}.ini"
        if not filename.exists():
            raise FileNotFoundError(
                f"Parameter file not found for pavement '{pavement}': {filename}"
            )
        mapping[pavement] = filename
    return mapping


def _parse_windows(definitions: Sequence[str]) -> List[TimeWindow]:
    windows: List[TimeWindow] = []
    for item in definitions:
        try:
            label, start, end = item.split(":", 2)
            start_ts = pd.to_datetime(start, utc=True).tz_convert("UTC").tz_localize(None)
            end_ts = pd.to_datetime(end, utc=True).tz_convert("UTC").tz_localize(None)
            windows.append(TimeWindow(label=label, start=start_ts, end=end_ts))
        except ValueError as exc:  # pragma: no cover - defensive path
            raise ValueError(
                "Windows must be provided as 'label:start:end' in ISO format"
            ) from exc
    return windows


def _parse_infiltration_capacities(values: Sequence[str]) -> Dict[str, float]:
    capacities: Dict[str, float] = {}
    for item in values:
        try:
            pavement, capacity = item.split(":", 1)
            capacities[pavement] = float(capacity)
        except ValueError as exc:  # pragma: no cover - defensive path
            raise ValueError(
                "Infiltration capacities must be provided as 'Pavement:value'"
            ) from exc
    return capacities


def _resolve_forcing_sources(forcing_path: Path, scenario_name: Optional[str]) -> Dict[str, Path]:
    forcing_path = Path(forcing_path)
    if forcing_path.is_dir():
        files = sorted(p for p in forcing_path.iterdir() if p.suffix.lower() in {".csv", ".parquet"})
        if not files:
            raise FileNotFoundError(
                f"No forcing files found in directory: {forcing_path}"
            )
        return {file.stem: file for file in files}

    if not forcing_path.exists():
        raise FileNotFoundError(forcing_path)

    name = scenario_name or forcing_path.stem
    return {name: forcing_path}


def run_cli() -> None:
    parser = argparse.ArgumentParser(
        description="Run climate change applications for the permeable pavement model",
    )
    parser.add_argument(
        "--forcing",
        required=True,
        help="Path to a scenario forcing file or directory of files",
    )
    parser.add_argument(
        "--scenario-name",
        help="Name to assign to the scenario when a single file is provided",
    )
    parser.add_argument(
        "--frequency",
        help="Optional resampling frequency for the forcing data (e.g., '1H', '5min')",
    )
    parser.add_argument(
        "--parameters-dir",
        default="input_data",
        help="Directory containing pavement parameter INI files",
    )
    parser.add_argument(
        "--pavements",
        nargs="*",
        default=["CP", "PICP", "PGr", "PA", "PC"],
        help="List of pavement identifiers to simulate",
    )
    parser.add_argument(
        "--windows",
        nargs="*",
        help="Optional analysis windows formatted as label:start:end (ISO timestamps)",
    )
    parser.add_argument(
        "--thresholds",
        nargs="*",
        type=float,
        default=[45.0, 50.0],
        help="Surface temperature thresholds (°C) for degree-hour calculations",
    )
    parser.add_argument(
        "--area-m2",
        type=float,
        default=200.0,
        help="Drainage area represented by each pavement installation (m²)",
    )
    parser.add_argument(
        "--infiltration-capacity",
        nargs="*",
        help="Optional infiltration capacities (mm/hr) as Pavement:value entries",
    )
    parser.add_argument(
        "--reference-water-temp",
        type=float,
        default=20.0,
        help="Reference downstream water temperature (°C) for thermal load calculations",
    )
    parser.add_argument(
        "--output",
        help="Optional path to write scenario diagnostics as JSON",
    )
    parser.add_argument(
        "--standardised-output-dir",
        help=(
            "Directory to write harmonised forcing CSVs used by the model. "
            "Files are saved after unit conversions and optional resampling."
        ),
    )

    args = parser.parse_args()

    forcing_sources = _resolve_forcing_sources(Path(args.forcing), args.scenario_name)
    parameter_mapping = _default_parameter_mapping(args.pavements, Path(args.parameters_dir))
    windows = _parse_windows(args.windows) if args.windows else []
    infiltration_capacities = (
        _parse_infiltration_capacities(args.infiltration_capacity)
        if args.infiltration_capacity
        else None
    )

    scenario_results: Dict[str, Dict[str, object]] = {}

    standardised_dir = (
        Path(args.standardised_output_dir)
        if args.standardised_output_dir is not None
        else None
    )
    if standardised_dir is not None:
        standardised_dir.mkdir(parents=True, exist_ok=True)

    for scenario_name, filepath in forcing_sources.items():
        export_path = (
            standardised_dir / f"{scenario_name}.csv"
            if standardised_dir is not None
            else None
        )
        forcing_df = load_forcing_data(
            filepath,
            frequency=args.frequency,
            standardised_output_path=export_path,
        )
        scenario_results.update(
            analyse_scenario(
                forcing_df=forcing_df,
                scenario_name=scenario_name,
                pavement_parameters=parameter_mapping,
                windows=windows,
                thresholds=args.thresholds,
                area_m2=args.area_m2,
                infiltration_capacities=infiltration_capacities,
                reference_water_temp_c=args.reference_water_temp,
            )
        )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as fh:
            json.dump(scenario_results, fh, indent=2)

    # Pretty-print a condensed summary for quick inspection
    for scenario_name, pavements in scenario_results.items():
        print(f"Scenario: {scenario_name}")
        for pavement, metrics in pavements.items():
            temp_metrics = metrics.get("temperature_metrics", {})
            mean_temp = temp_metrics.get("mean")
            p95_temp = temp_metrics.get("p95")
            print(
                f"  {pavement}: mean={mean_temp:.2f} °C, p95={p95_temp:.2f} °C"
                if mean_temp is not None and p95_temp is not None
                else f"  {pavement}: insufficient data"
            )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    run_cli()

