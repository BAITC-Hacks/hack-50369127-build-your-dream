from __future__ import annotations

import tomllib
from dataclasses import dataclass
from datetime import date
from pathlib import Path


@dataclass(frozen=True)
class DataSettings:
    turbine_1_csv: Path
    timestamp_column: str
    wind_speed_column: str
    target_column: str
    temperature_column: str
    expected_step_minutes: int
    hourly_output_csv: Path


@dataclass(frozen=True)
class WeatherSettings:
    endpoint: str
    model: str
    forecast_hours: int
    timezone: str
    daily_calculation_hour_utc: int
    model_latency_hours: int
    wind_speed_variable: str
    temperature_variable: str
    hourly_variables: tuple[str, ...]
    retry_attempts: int = 3
    retry_backoff_seconds: float = 1.0
    fallback_runs: int = 4
    request_timeout_seconds: int = 30
    provider: str = "open-meteo-single-runs"
    require_as_issued: bool = False


@dataclass(frozen=True)
class TurbineSettings:
    id: str
    latitude: float
    longitude: float
    data_csv: Path | None
    capacity_mw: float | None
    model_source: str
    weight: float = 1.0


@dataclass(frozen=True)
class OutputSettings:
    forecast_dir: Path
    weather_dir: Path
    run_log_dir: Path


@dataclass(frozen=True)
class Settings:
    project_timezone: str
    data: DataSettings
    weather: WeatherSettings
    outputs: OutputSettings
    turbines: tuple[TurbineSettings, ...]
    daily_calculation_hour_local: int = 23
    horizon_hours: int = 48
    training_end_date: date = date(2026, 1, 31)
    test_start_date: date = date(2026, 2, 1)
    test_end_date: date = date(2026, 2, 28)


def _path(value: str, base_dir: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = base_dir / path
    return path


def load_settings(config_path: Path) -> Settings:
    config_path = config_path.resolve()
    root_dir = config_path.parent.parent
    raw = tomllib.loads(config_path.read_text(encoding="utf-8-sig"))

    data = raw["data"]
    weather = raw["weather"]
    outputs = raw["outputs"]
    turbines = raw.get("turbines", [])

    settings = Settings(
        project_timezone=raw["project"]["timezone"],
        data=DataSettings(
            turbine_1_csv=_path(data["turbine_1_csv"], root_dir),
            timestamp_column=data["timestamp_column"],
            wind_speed_column=data["wind_speed_column"],
            target_column=data["target_column"],
            temperature_column=data["temperature_column"],
            expected_step_minutes=int(data["expected_step_minutes"]),
            hourly_output_csv=_path(data["hourly_output_csv"], root_dir),
        ),
        weather=WeatherSettings(
            endpoint=weather["endpoint"],
            model=weather["model"],
            forecast_hours=int(weather["forecast_hours"]),
            timezone=weather["timezone"],
            daily_calculation_hour_utc=int(weather["daily_calculation_hour_utc"]),
            model_latency_hours=int(weather["model_latency_hours"]),
            wind_speed_variable=weather["wind_speed_variable"],
            temperature_variable=weather["temperature_variable"],
            hourly_variables=tuple(weather["hourly_variables"]),
            retry_attempts=int(weather.get("retry_attempts", 3)),
            retry_backoff_seconds=float(weather.get("retry_backoff_seconds", 1)),
            fallback_runs=int(weather.get("fallback_runs", 4)),
            request_timeout_seconds=int(weather.get("request_timeout_seconds", 30)),
            provider=weather.get("provider", "open-meteo-single-runs"),
            require_as_issued=bool(weather.get("require_as_issued", False)),
        ),
        outputs=OutputSettings(
            forecast_dir=_path(outputs["forecast_dir"], root_dir),
            weather_dir=_path(outputs["weather_dir"], root_dir),
            run_log_dir=_path(outputs["run_log_dir"], root_dir),
        ),
        turbines=tuple(
            TurbineSettings(
                id=item["id"],
                latitude=float(item["latitude"]),
                longitude=float(item["longitude"]),
                data_csv=_path(item["data_csv"], root_dir) if item.get("data_csv") else None,
                capacity_mw=float(item["capacity_mw"]) if "capacity_mw" in item else None,
                model_source=item.get("model_source", "self"),
                weight=float(item.get("weight", 1.0)),
            )
            for item in turbines
        ),
        daily_calculation_hour_local=int(raw["project"].get("daily_calculation_hour_local", 23)),
        horizon_hours=int(raw["project"].get("horizon_hours", 48)),
        training_end_date=date.fromisoformat(
            str(raw["project"].get("training_end_date", "2026-01-31"))
        ),
        test_start_date=date.fromisoformat(
            str(raw["project"].get("test_start_date", "2026-02-01"))
        ),
        test_end_date=date.fromisoformat(str(raw["project"].get("test_end_date", "2026-02-28"))),
    )
    if settings.horizon_hours not in (24, 48):
        raise ValueError("project.horizon_hours must be 24 or 48")
    if not 0 <= settings.daily_calculation_hour_local <= 23:
        raise ValueError("daily_calculation_hour_local must be in 0..23")
    if settings.training_end_date >= settings.test_start_date:
        raise ValueError("Training must end before the test period")
    if settings.test_end_date < settings.test_start_date:
        raise ValueError("test_end_date must follow test_start_date")
    if settings.weather.timezone != "UTC":
        raise ValueError("Weather responses must use UTC")
    if settings.weather.model_latency_hours < 0:
        raise ValueError("Weather publication latency cannot be negative")
    if not settings.turbines or len({t.id for t in settings.turbines}) != len(settings.turbines):
        raise ValueError("Configure at least one turbine with unique IDs")
    import math

    for turbine in settings.turbines:
        if not (-90 <= turbine.latitude <= 90 and -180 <= turbine.longitude <= 180):
            raise ValueError(f"Invalid coordinates: {turbine.id}")
        if not math.isfinite(turbine.weight) or turbine.weight <= 0:
            raise ValueError("Turbine weights must be positive and finite")
        if turbine.capacity_mw is not None and (
            not math.isfinite(turbine.capacity_mw) or turbine.capacity_mw <= 0
        ):
            raise ValueError("Turbine capacity must be positive and finite when supplied")
    if any(t.capacity_mw is not None for t in settings.turbines) and not all(
        t.capacity_mw is not None for t in settings.turbines
    ):
        raise ValueError("Provide capacity_mw for all turbines or leave all unspecified")
    return settings
