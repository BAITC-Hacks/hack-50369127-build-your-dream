from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from datetime import date, datetime, time, timedelta
from importlib.metadata import version
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .data import make_hourly_measurements
from .model import predict_power, train_power_model
from .settings import Settings
from .weather import get_weather_for_forecast


@dataclass(frozen=True)
class ForecastRunReport:
    calculation_date: str
    issued_at: str
    weather_run: str
    forecast_csv: str
    run_log_json: str
    rows: int
    checks: dict
    model: dict
    inputs: dict
    fingerprint: str
    status: str = "calculated"

    def to_dict(self) -> dict:
        return asdict(self)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def forecast_window(settings: Settings, calculation_date: date):
    local = ZoneInfo(settings.project_timezone)
    issued_at = pd.Timestamp(
        datetime.combine(
            calculation_date, time(settings.daily_calculation_hour_local), tzinfo=local
        )
    )
    start = pd.Timestamp(calculation_date + timedelta(days=1), tz=local)
    targets = pd.date_range(start, periods=settings.horizon_hours, freq="h")
    if not (targets > issued_at).all():
        raise ValueError("All target hours must start after the calculation instant")
    return issued_at, targets


class ForecastAgent:
    """Policy agent: observe inputs, choose eligible runs/model, validate, publish or reuse.

    Historical issue time is fixed during refreshes: a later retrieved payload
    must still describe a run eligible at that original issue time.
    """

    def __init__(self, settings: Settings, *, offline: bool = False, refresh: bool = False):
        self.settings = settings
        self.offline = offline
        self.refresh = refresh
        self._hourly = {}
        self._models = {}
        self.runtime_versions = {
            name: version(name) for name in ("numpy", "pandas", "scikit-learn", "tzdata")
        }
        self.code_hash = _json_hash(
            {path.name: _sha256(path) for path in Path(__file__).parent.glob("*.py")}
        )

    def _event(self, day: date, action: str, **details):
        path = self.settings.outputs.run_log_dir / "agent_events.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "recorded_at": datetime.now().astimezone().isoformat(),
            "calculation_date": str(day),
            "action": action,
            **details,
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def run(self, calculation_date: date) -> ForecastRunReport:
        try:
            return self._run(calculation_date)
        except Exception as exc:
            self._event(calculation_date, "failed", error_type=type(exc).__name__, error=str(exc))
            failure = {
                "status": "failed",
                "calculation_date": str(calculation_date),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            _atomic_text(
                self.settings.outputs.run_log_dir / f"failed_{calculation_date}.json",
                json.dumps(failure, ensure_ascii=False, indent=2),
            )
            raise

    def _run(self, day: date) -> ForecastRunReport:
        settings = self.settings
        issued_at, targets = forecast_window(settings, day)
        cutoff = min(
            issued_at,
            pd.Timestamp(
                settings.training_end_date + timedelta(days=1), tz=settings.project_timezone
            ),
        )
        self._event(
            day, "observe", issued_at=issued_at.isoformat(), training_cutoff=cutoff.isoformat()
        )
        histories, weather, provenance, inputs, history_keys = {}, {}, {}, {}, {}
        for turbine in settings.turbines:
            if turbine.model_source != "self" or turbine.data_csv is None:
                raise ValueError(
                    f"Independent history required for {turbine.id}; set model_source=self"
                )
            source_hash = _sha256(turbine.data_csv)
            cache_key = (turbine.id, source_hash)
            if cache_key not in self._hourly:
                data_settings = replace(settings.data, turbine_1_csv=turbine.data_csv)
                self._hourly[cache_key] = make_hourly_measurements(
                    data_settings, settings.project_timezone, write_output=False
                )
            hourly = self._hourly[cache_key]
            training = (
                hourly[hourly["available_at"] <= cutoff]
                .dropna(subset=["wind_speed", "temperature", "normalized_power"])
                .copy()
            )
            if training.empty:
                raise ValueError(f"No complete training hours before cutoff for {turbine.id}")
            history_key = hashlib.sha256(
                pd.util.hash_pandas_object(
                    training[["timestamp", "wind_speed", "temperature", "normalized_power"]],
                    index=False,
                ).values.tobytes()
            ).hexdigest()
            histories[turbine.id] = training
            history_keys[turbine.id] = (turbine.id, history_key)
            inputs[turbine.id] = {
                "source_csv": str(turbine.data_csv),
                "source_sha256": source_hash,
                "training_sha256": history_key,
                "last_training_available_at": training["available_at"].max().isoformat(),
            }
            provider = get_weather_for_forecast
            if settings.weather.provider == "noaa-gfs":
                from .gfs import get_weather_for_forecast as gfs_provider

                provider = gfs_provider
            elif settings.weather.provider != "open-meteo-single-runs":
                raise ValueError(f"Unknown weather provider: {settings.weather.provider}")
            frame, metadata = provider(
                settings.weather,
                turbine,
                issued_at.to_pydatetime(),
                targets,
                settings.outputs.weather_dir,
                offline=self.offline,
                refresh=self.refresh,
            )
            if settings.weather.require_as_issued and not metadata.get(
                "as_issued_authenticity_verified", False
            ):
                raise ValueError(
                    "Strict replay requires verified original operational forecasts; "
                    "this weather archive is not verified as-issued"
                )
            frame = frame.copy()
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True).dt.tz_convert(
                settings.project_timezone
            )
            if list(frame["timestamp"]) != list(targets):
                raise ValueError(f"Weather hours are not the exact forecast grid for {turbine.id}")
            weather[turbine.id] = frame
            provenance[turbine.id] = metadata
        fingerprint = _json_hash(
            {
                "settings": asdict(settings),
                "runtime_versions": self.runtime_versions,
                "code": self.code_hash,
                "day": day,
                "training": history_keys,
                "weather": {
                    key: {
                        field: value
                        for field, value in meta.items()
                        if field in ("payload_sha256", "run", "request_url", "assumed_available_at")
                    }
                    for key, meta in provenance.items()
                },
            }
        )
        path = settings.outputs.forecast_dir / f"forecast_{day}.csv"
        log_path = settings.outputs.run_log_dir / f"run_{day}.json"
        if path.exists() and log_path.exists():
            previous = json.loads(log_path.read_text(encoding="utf-8"))
            if previous.get("fingerprint") == fingerprint and previous.get("checks", {}).get(
                "forecast_sha256"
            ) == _sha256(path):
                self._event(day, "reuse_unchanged", fingerprint=fingerprint)
                previous["status"] = "unchanged"
                return ForecastRunReport(**previous)
        self._event(day, "plan", decision="recalculate", fingerprint=fingerprint)
        forecast = pd.DataFrame({"target_time": targets})
        model_reports = {}
        for turbine in settings.turbines:
            key = history_keys[turbine.id]
            if key not in self._models:
                self._models[key] = train_power_model(histories[turbine.id])
                self._event(day, "train", turbine=turbine.id)
            model, report = self._models[key]
            model_reports[turbine.id] = report.to_dict()
            forecast[f"{turbine.id}_normalized_power"] = predict_power(
                model, weather[turbine.id]
            ).to_numpy()
            forecast[f"{turbine.id}_mean_baseline"] = report.training_mean_power
            forecast[f"{turbine.id}_persistence_baseline"] = float(
                histories[turbine.id]["normalized_power"].iloc[-1]
            )
            forecast[f"{turbine.id}_weather_run"] = provenance[turbine.id]["run"]
        columns = [f"{t.id}_normalized_power" for t in settings.turbines]
        known_capacity = all(t.capacity_mw is not None for t in settings.turbines)
        weights = np.array(
            [t.capacity_mw if known_capacity else t.weight for t in settings.turbines]
        )
        forecast["wind_farm_normalized_power"] = (
            forecast[columns].to_numpy() @ weights / weights.sum()
        )
        if known_capacity:
            forecast["wind_farm_power_mw"] = forecast[columns].to_numpy() @ weights
            forecast["wind_farm_energy_mwh"] = forecast["wind_farm_power_mw"]  # one-hour bins
        values = forecast[columns + ["wind_farm_normalized_power"]].to_numpy()
        if not np.isfinite(values).all() or not ((values >= 0) & (values <= 1)).all():
            raise ValueError("Forecast quality gate failed: non-finite or out-of-range power")
        forecast.insert(0, "calculation_date", day.isoformat())
        forecast.insert(1, "issued_at", issued_at.tz_convert("UTC").isoformat())
        forecast.insert(2, "forecast_hour", np.arange(1, len(forecast) + 1))
        forecast.insert(3, "lead_hours", (targets - issued_at).total_seconds() / 3600)
        self._event(day, "analyze", result="passed", rows=len(forecast))
        csv = forecast.to_csv(index=False)
        revision_path = (
            settings.outputs.forecast_dir / "revisions" / f"forecast_{day}_{fingerprint[:12]}.csv"
        )
        _atomic_text(revision_path, csv)
        _atomic_text(path, csv)
        checks = {
            "expected_rows": settings.horizon_hours,
            "actual_rows": len(forecast),
            "missing_predictions": 0,
            "predictions_within_0_1": True,
            "all_targets_after_issue": True,
            "training_cutoff": cutoff.isoformat(),
            "test_labels_used_in_training": False,
            "aggregation": "capacity_weighted" if known_capacity else "configured_weights_assumed",
            "telemetry_timezone": settings.project_timezone,
            "timezone_confirmed": False,
            "weather_availability": {
                key: meta.get("availability_basis", "initialization_plus_assumed_delay")
                for key, meta in provenance.items()
            },
            "as_issued_authenticity_verified": all(
                meta.get("as_issued_authenticity_verified", False) for meta in provenance.values()
            ),
            "forecast_sha256": _sha256(path),
            "revision_csv": str(revision_path),
        }
        inputs["weather"] = provenance
        inputs["code_sha256"] = self.code_hash
        inputs["runtime_versions"] = self.runtime_versions
        weather_runs = sorted({meta["run"] for meta in provenance.values()})
        report = ForecastRunReport(
            calculation_date=str(day),
            issued_at=issued_at.isoformat(),
            weather_run=",".join(weather_runs),
            forecast_csv=str(path),
            run_log_json=str(log_path),
            rows=len(forecast),
            checks=checks,
            model=model_reports,
            inputs=inputs,
            fingerprint=fingerprint,
        )
        content = json.dumps(report.to_dict(), ensure_ascii=False, indent=2, default=str)
        _atomic_text(
            settings.outputs.run_log_dir / "revisions" / f"run_{day}_{fingerprint[:12]}.json",
            content,
        )
        _atomic_text(log_path, content)
        self._event(day, "publish", forecast_csv=str(path), fingerprint=fingerprint)
        return report


def run_daily_forecast(
    settings: Settings, calculation_date: date, *, offline: bool = False, refresh: bool = False
) -> ForecastRunReport:
    return ForecastAgent(settings, offline=offline, refresh=refresh).run(calculation_date)
