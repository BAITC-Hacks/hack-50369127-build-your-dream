from __future__ import annotations

import argparse
import logging
from pathlib import Path
import platform
from typing import Any

import numpy as np
import pandas as pd

from src.data.fetch_weather import WeatherClient, VARIABLES
from src.data.preprocess_telemetry import prepare_all
from src.utils.common import load_config, resolve_path, setup_logging, utc, write_csv, write_json

LOG = logging.getLogger(__name__)
FEATURE_COLUMNS = ["turbine_id", *VARIABLES, "air_density", "lead_hours", "weather_lead_hours",
                   "hour", "month", "dayofweek", "hour_sin", "hour_cos",
                   "month_sin", "month_cos", "dayofweek_sin", "dayofweek_cos",
                   "wind_direction_10m_sin", "wind_direction_10m_cos",
                   "wind_direction_100m_sin", "wind_direction_100m_cos"]


def add_features(weather: pd.DataFrame) -> pd.DataFrame:
    frame = weather.copy()
    frame["hour"] = frame["valid_time"].dt.hour
    frame["month"] = frame["valid_time"].dt.month
    frame["dayofweek"] = frame["valid_time"].dt.dayofweek
    for name, period, offset in [("hour", 24, 0), ("month", 12, 1), ("dayofweek", 7, 0),
                                  ("wind_direction_10m", 360, 0),
                                  ("wind_direction_100m", 360, 0)]:
        angle = 2 * np.pi * (frame[name] - offset) / period
        frame[f"{name}_sin"] = np.sin(angle)
        frame[f"{name}_cos"] = np.cos(angle)
    return frame


def join_labels(weather: pd.DataFrame, telemetry: pd.DataFrame,
                cfg: dict[str, Any]) -> pd.DataFrame:
    frame = add_features(weather)
    if not (frame["run_time"] <= frame["available_at"]).all():
        raise ValueError("Run availability precedes initialization")
    if not (frame["available_at"] <= frame["issue_time"]).all():
        raise ValueError("Weather release was unavailable at forecast issue time")
    if not (frame["valid_time"] > frame["issue_time"]).all():
        raise ValueError("Forecast target must be after issue time")
    cutoff = utc(cfg["dataset"]["label_cutoff_exclusive"])
    # A mean over [t,t+1h) is fully observable only at t+1h.
    labels = telemetry[telemetry["valid_time"] + pd.Timedelta(hours=1) <= cutoff]
    labels = labels[["turbine_id", "valid_time", "power_norm", "coverage_power"]]
    labels = labels.rename(columns={"power_norm": "target_power_norm"})
    result = frame.merge(labels, on=["turbine_id", "valid_time"], how="left", validate="many_to_one")
    result["label_available"] = result["target_power_norm"].notna()
    result["in_evaluation_window"] = (
        result["valid_time"].ge(utc(cfg["dataset"]["evaluation_start"])) &
        result["valid_time"].lt(utc(cfg["dataset"]["evaluation_end_exclusive"])))
    # No contemporaneous measured wind/temperature enters the predictor allowlist.
    return result


def build(cfg: dict[str, Any], offline: bool = False, refresh: bool = False,
          telemetry_only: bool = False, allow_missing: bool = False) -> dict[str, Any]:
    telemetry, reports = prepare_all(cfg)
    if telemetry_only:
        return {"mode": "telemetry_only", "quality": reports}
    options = cfg["dataset"]
    horizon = int(options["horizon_hours"])
    if horizon not in {24, 48} or not 0 <= int(options["issue_hour_utc"]) <= 23:
        raise ValueError("horizon_hours must be 24 or 48; issue_hour_utc must be in 0..23")
    start = pd.Timestamp(options["issue_start_date"])
    end = pd.Timestamp(options["issue_end_date"])
    if start.tzinfo is not None or end.tzinfo is not None or start != start.normalize() or end != end.normalize():
        raise ValueError("Issue range requires plain dates YYYY-MM-DD")
    if end < start:
        raise ValueError("Issue range end is before start")
    days = pd.date_range(start, end, freq="D", tz="UTC")
    folder = resolve_path(cfg, cfg["paths"]["features"])
    frames: list[pd.DataFrame] = []
    failures: list[dict[str, str]] = []
    client = WeatherClient(cfg, offline=offline, refresh=refresh)
    try:
        for day in days:
            issue = day + pd.Timedelta(hours=int(options["issue_hour_utc"]))
            times = pd.date_range(issue + pd.Timedelta(hours=1), periods=horizon, freq="h")
            daily = []
            try:
                for turbine in cfg["turbines"]:
                    weather = client.fetch(turbine, issue, times)
                    daily.append(join_labels(weather, telemetry, cfg))
            except (ValueError, RuntimeError, FileNotFoundError) as exc:
                if not allow_missing:
                    raise
                LOG.warning("Excluding unavailable training issue %s: %s", issue, exc)
                failures.append({"issue_time": issue.isoformat(), "reason": str(exc)})
                continue
            frame = pd.concat(daily, ignore_index=True)
            if len(frame) != len(cfg["turbines"]) * horizon:
                raise ValueError("Incomplete issue batch")
            write_csv(folder / f"features_{issue.strftime('%Y%m%dT%H%M%SZ')}.csv", frame)
            frames.append(frame)
            LOG.info("Built issue %s: %d turbine-hours", issue, len(frame))
    finally:
        client.close()
    if not frames:
        raise ValueError("No complete weather issues were available")
    combined = pd.concat(frames, ignore_index=True)
    keys = ["turbine_id", "issue_time", "run_time", "valid_time"]
    if combined.duplicated(keys).any():
        raise ValueError("Nonunique dataset primary key")
    tag = f"{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}"
    destination = folder / f"dataset_{tag}.csv"
    write_csv(destination, combined)
    manifest = {
        "dataset": str(destination.relative_to(cfg["_root"])),
        "rows": len(combined), "issues": len(frames), "requested_issues": len(days),
        "excluded_issues": failures, "turbines": len(cfg["turbines"]),
        "horizon_hours": horizon, "feature_columns": FEATURE_COLUMNS,
        "target_column": "target_power_norm", "primary_key": keys,
        "label_cutoff_exclusive": options["label_cutoff_exclusive"],
        "labels_present": int(combined["label_available"].sum()),
        "availability_verified": bool(combined["availability_verified"].all()),
        "availability_policy": cfg["weather"]["availability_policy"],
        "weather_cache_keys": sorted(combined["cache_key"].unique().tolist()),
        "quality": reports,
        "versions": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__},
        "configuration": {k: v for k, v in cfg.items() if not k.startswith("_")},
    }
    write_json(folder / f"manifest_{tag}.json", manifest)
    LOG.info("Dataset saved: %s; labels=%d", destination, manifest["labels_present"])
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Build hourly features from historical forecast runs")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--start", help="First issue date, inclusive YYYY-MM-DD")
    parser.add_argument("--end", help="Last issue date, inclusive YYYY-MM-DD")
    parser.add_argument("--telemetry-only", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--allow-missing", action="store_true", help="Record and skip unavailable training issues")
    args = parser.parse_args()
    try:
        cfg = load_config(args.config)
        setup_logging(cfg)
        if args.start:
            cfg["dataset"]["issue_start_date"] = args.start
        if args.end:
            cfg["dataset"]["issue_end_date"] = args.end
        build(cfg, args.offline, args.refresh, args.telemetry_only, args.allow_missing)
        return 0
    except Exception:
        LOG.exception("Dataset build failed; no fabricated weather or silent fallback used")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
