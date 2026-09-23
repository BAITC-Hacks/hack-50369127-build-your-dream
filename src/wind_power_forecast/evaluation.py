"""Evaluate frozen forecast artifacts. Targets are never passed back to training."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from .data import make_hourly_measurements
from .settings import Settings


def _scores(actual: pd.Series, predicted: pd.Series) -> dict[str, float | int]:
    delta = np.asarray(actual, dtype=float) - np.asarray(predicted, dtype=float)
    return {
        "hours": len(delta),
        "mae": float(np.abs(delta).mean()),
        "rmse": float(np.sqrt(np.square(delta).mean())),
        "bias": float(-delta.mean()),
    }


def compile_submission(settings: Settings) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Day-ahead: exactly one forecast issued on the preceding local date per target."""
    paths = sorted(settings.outputs.forecast_dir.glob("forecast_????-??-??.csv"))
    if not paths:
        raise ValueError("No forecasts found; run forecast-range first")
    frames = []
    for path in paths:
        day = path.stem.removeprefix("forecast_")
        log = settings.outputs.run_log_dir / f"run_{day}.json"
        if log.exists():
            recorded = json.loads(log.read_text(encoding="utf-8"))
            expected_hash = recorded.get("checks", {}).get("forecast_sha256")
            if expected_hash and hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
                raise ValueError(f"Forecast checksum mismatch: {path.name}")
        frames.append(pd.read_csv(path))
    forecasts = pd.concat(frames, ignore_index=True)
    for column in ("target_time", "issued_at"):
        forecasts[column] = pd.to_datetime(forecasts[column], utc=True)
    start = pd.Timestamp(settings.test_start_date, tz=settings.project_timezone)
    end = pd.Timestamp(settings.test_end_date, tz=settings.project_timezone) + pd.Timedelta(days=1)
    forecasts = forecasts[
        (forecasts["target_time"] >= start) & (forecasts["target_time"] < end)
    ].copy()
    if (forecasts["issued_at"] >= forecasts["target_time"]).any():
        raise ValueError("A forecast was issued after its target hour started")
    forecasts["target_day"] = (
        forecasts["target_time"].dt.tz_convert(settings.project_timezone).dt.date
    )
    issue_dates = pd.to_datetime(forecasts["calculation_date"]).dt.date
    local_issue_dates = forecasts["issued_at"].dt.tz_convert(settings.project_timezone).dt.date
    if not local_issue_dates.equals(issue_dates):
        raise ValueError("calculation_date must match the issue date in the project timezone")
    offsets = pd.Series(
        [(target - issue).days for target, issue in zip(forecasts["target_day"], issue_dates)],
        index=forecasts.index,
    )
    forecasts["horizon_day"] = offsets
    if forecasts.duplicated(["issued_at", "target_time"]).any():
        raise ValueError("Duplicate issue/target forecast rows")
    if not forecasts["horizon_day"].isin([1, 2]).all():
        raise ValueError("Forecast targets must fall in the first or second following local day")
    submission = forecasts[forecasts["horizon_day"] == 1].copy()
    expected = pd.date_range(start, end, freq="h", inclusive="left").tz_convert("UTC")
    if submission["target_time"].duplicated().any():
        raise ValueError("Duplicate day-ahead forecast hours")
    if set(submission["target_time"]) != set(expected):
        raise ValueError(
            f"Incomplete test period: expected {len(expected)} day-ahead hours, "
            f"found {len(submission)}. Run all dates from the day before test_start."
        )
    power_columns = [f"{t.id}_normalized_power" for t in settings.turbines]
    power_columns.append("wind_farm_normalized_power")
    values = forecasts[power_columns].to_numpy(dtype=float)
    if not np.isfinite(values).all() or not ((values >= 0) & (values <= 1)).all():
        raise ValueError("Forecasts have missing, non-finite or out-of-range predictions")
    forecasts = forecasts.drop(columns="target_day").sort_values(["issued_at", "target_time"])
    submission = submission.drop(columns="target_day").sort_values("target_time")
    output_dir = settings.outputs.forecast_dir
    forecasts.to_csv(output_dir / "all_forecasts.csv", index=False)
    submission.to_csv(output_dir / "submission.csv", index=False)
    return submission, forecasts


def evaluate_forecasts(settings: Settings, actual_dir: Path | None = None) -> dict:
    submission, forecasts = compile_submission(settings)
    results: dict = {
        "status": "evaluated",
        "weather_provider": settings.weather.provider,
        "require_as_issued": settings.weather.require_as_issued,
        "target": "normalized_hourly_mean_active_power",
        "test_start": settings.test_start_date.isoformat(),
        "test_end": settings.test_end_date.isoformat(),
        "expected_hours": len(submission),
        "forecast_hours_by_horizon_day": {
            str(day): int((forecasts["horizon_day"] == day).sum()) for day in (1, 2)
        },
        "day_ahead": {},
        "second_day": {},
        "note": "Only observed hours are scored; missing targets are not filled or set to zero.",
    }
    matched = 0
    actual_by_turbine: dict[str, pd.Series] = {}
    for turbine in settings.turbines:
        actual_path = (actual_dir / f"{turbine.id}.csv") if actual_dir else turbine.data_csv
        if actual_path is None or not actual_path.exists():
            results["day_ahead"][turbine.id] = {"hours": 0, "status": "no_actuals"}
            results["second_day"][turbine.id] = {"hours": 0, "status": "no_actuals"}
            continue
        data_settings = replace(settings.data, turbine_1_csv=actual_path)
        actual = make_hourly_measurements(
            data_settings, settings.project_timezone, write_output=False
        )[["timestamp", "normalized_power"]].dropna()
        actual["timestamp"] = pd.to_datetime(actual["timestamp"], utc=True)
        actual_by_turbine[turbine.id] = actual.set_index("timestamp")["normalized_power"]
        for day, label in ((1, "day_ahead"), (2, "second_day")):
            selected = forecasts[forecasts["horizon_day"] == day]
            pairs = selected.merge(actual, left_on="target_time", right_on="timestamp")
            if pairs.empty:
                results[label][turbine.id] = {"hours": 0, "status": "no_actuals"}
                continue
            matched += len(pairs)
            scores = _scores(pairs["normalized_power"], pairs[f"{turbine.id}_normalized_power"])
            for name in ("mean_baseline", "persistence_baseline"):
                column = f"{turbine.id}_{name}"
                if column in pairs:
                    scores[name] = _scores(pairs["normalized_power"], pairs[column])
            results[label][turbine.id] = scores
    known_capacity = all(turbine.capacity_mw is not None for turbine in settings.turbines)
    weights = np.asarray([
        turbine.capacity_mw if known_capacity else turbine.weight
        for turbine in settings.turbines
    ], dtype=float)
    weights /= weights.sum()
    results["farm_aggregation"] = {
        "basis": "capacity_weighted" if known_capacity else "configured_weights_assumed",
        "normalized_weights": {
            turbine.id: float(weight) for turbine, weight in zip(settings.turbines, weights)
        },
        "requires_complete_actuals_for_all_turbines": True,
    }
    farm_actual = None
    if len(actual_by_turbine) == len(settings.turbines):
        # Inner alignment retains only hours observed for every turbine. An
        # absent component must never become zero or change the station weights.
        aligned = pd.concat(
            [actual_by_turbine[turbine.id] for turbine in settings.turbines],
            axis=1, join="inner",
        ).dropna()
        farm_actual = pd.Series(
            aligned.to_numpy(dtype=float) @ weights,
            index=aligned.index,
            name="actual_farm_normalized_power",
        )
    for day, label in ((1, "day_ahead"), (2, "second_day")):
        key = "wind_farm_normalized_power"
        if farm_actual is None or farm_actual.empty:
            results[label][key] = {"hours": 0, "status": "no_actuals"}
            continue
        pairs = forecasts[forecasts["horizon_day"] == day].merge(
            farm_actual, left_on="target_time", right_index=True, validate="many_to_one"
        )
        if pairs.empty:
            results[label][key] = {"hours": 0, "status": "no_actuals"}
            continue
        scores = _scores(pairs["actual_farm_normalized_power"], pairs[key])
        for name in ("mean_baseline", "persistence_baseline"):
            columns = [f"{turbine.id}_{name}" for turbine in settings.turbines]
            if not set(columns).issubset(pairs.columns):
                scores[name] = {"hours": 0, "status": "incomplete_baseline"}
                continue
            components = pairs[columns].to_numpy(dtype=float)
            if not np.isfinite(components).all():
                scores[name] = {"hours": 0, "status": "incomplete_baseline"}
                continue
            scores[name] = _scores(
                pairs["actual_farm_normalized_power"], components @ weights
            )
        results[label][key] = scores
    if matched == 0:
        results["status"] = "unavailable_no_test_actuals"
    report_path = settings.outputs.run_log_dir / "evaluation.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return results
