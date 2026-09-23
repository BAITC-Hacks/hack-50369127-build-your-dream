from __future__ import annotations

import argparse
from contextlib import closing
from copy import deepcopy
import hashlib
import json
import logging
from pathlib import Path
import sqlite3
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.agent.planner import decide
from src.data.build_features import add_features
from src.data.fetch_weather import WeatherClient
from src.models.features import enrich
from src.models.train import apply_intervals
from src.utils.common import load_config, setup_logging, sha256_file, utc, write_csv, write_json

LOG = logging.getLogger(__name__)


class ForecastAgent:
    def __init__(self, cfg: dict[str, Any], offline: bool = False,
                 use_llm: bool = False, refresh: bool = False) -> None:
        self.cfg = deepcopy(cfg)
        if self.cfg["agent"]["strict_provenance"]:
            self.cfg["weather"]["require_verified_availability"] = True
        self.offline, self.use_llm = offline, use_llm
        path = cfg["_root"] / "artifacts/forecast_bundle.joblib"
        self.model_hash = sha256_file(path)
        # Load only a trusted local model file produced by this project's train CLI.
        self.bundle = joblib.load(path)
        self.client = WeatherClient(self.cfg, offline=offline, refresh=refresh)
        self.output = cfg["_root"] / "outputs"
        self.output.mkdir(exist_ok=True)
        self.database = self.output / "agent.sqlite"
        with closing(sqlite3.connect(self.database)) as db, db:
            db.execute("""CREATE TABLE IF NOT EXISTS forecasts (
              fingerprint TEXT PRIMARY KEY, issue_time TEXT, output_path TEXT, output_sha256 TEXT,
              model_sha256 TEXT, created_at TEXT)""")
            db.execute("""CREATE TABLE IF NOT EXISTS events (
              id INTEGER PRIMARY KEY, issue_time TEXT, state TEXT, details TEXT, recorded_at TEXT)""")

    def close(self) -> None:
        self.client.close()

    def event(self, issue: pd.Timestamp, state: str, details: dict[str, Any]) -> None:
        with closing(sqlite3.connect(self.database)) as db, db:
            db.execute("INSERT INTO events(issue_time,state,details,recorded_at) VALUES(?,?,?,?)",
                       (issue.isoformat(), state, json.dumps(details, default=str),
                        pd.Timestamp.now(tz="UTC").isoformat()))
        LOG.info("%s %s %s", issue, state, details)

    def run(self, issue_time: pd.Timestamp) -> pd.DataFrame:
        issue = utc(issue_time)
        if utc(self.bundle["training_asof"]) > issue:
            raise ValueError("Model contains labels that were not available at issue_time")
        if self.bundle["weather_model"] != self.cfg["weather"]["model"]:
            raise ValueError("Weather provider/model differs from the trained model")
        self.event(issue, "PLAN", {"horizon_hours": self.cfg["dataset"]["horizon_hours"],
                                    "planner": "openai" if self.use_llm else "deterministic"})
        times = pd.date_range(issue.floor("h") + pd.Timedelta(hours=1),
                              periods=self.cfg["dataset"]["horizon_hours"], freq="h")
        frames = []
        max_age = int(self.cfg["agent"]["fallback_max_age_hours"])
        if not self.cfg["agent"]["allow_fallback"]:
            max_age = 0
        try:
            for turbine in self.cfg["turbines"]:
                successful = False
                for age in range(0, max_age + 1, 6):
                    try:
                        frame = self.client.fetch(turbine, issue - pd.Timedelta(hours=age), times)
                        frame["issue_time"] = issue
                        frame["lead_hours"] = (frame.valid_time - issue).dt.total_seconds() / 3600
                        frame["fallback_age_hours"] = age
                        if (frame.available_at > issue).any():
                            raise ValueError("Weather unavailable at issue")
                        self.event(issue, "WEATHER_OK", {"turbine": turbine["id"], "fallback_age_hours": age,
                                                        "run_time": str(frame.run_time.iloc[0])})
                        frames.append(frame)
                        successful = True
                        break
                    except (ValueError, RuntimeError, FileNotFoundError) as exc:
                        self.event(issue, "WEATHER_FAILED", {"turbine": turbine["id"], "age": age, "error": str(exc)})
                        if age >= max_age:
                            raise
                        decision = decide({"weather_failed": True, "turbine": turbine["id"],
                                           "older_runs_remaining": (max_age - age) // 6, "error": str(exc)}, self.use_llm)
                        self.event(issue, "REPLAN", decision)
                        if decision["action"] == "halt":
                            raise RuntimeError("Planner halted after weather failure") from exc
                if not successful:
                    raise RuntimeError("No weather release available")
            features = enrich(add_features(pd.concat(frames, ignore_index=True)))
            token = {"issue_time": str(issue), "model_hash": self.model_hash,
                     "weather_hashes": features.weather_sha256.tolist(),
                     "releases": features[["turbine_id", "run_time", "available_at", "fallback_age_hours"]].drop_duplicates().to_dict("records"),
                     "config": {k: v for k, v in self.cfg.items() if not k.startswith("_")},
                     "llm": self.use_llm}
            fingerprint = hashlib.sha256(json.dumps(token, sort_keys=True, default=str).encode()).hexdigest()
            with closing(sqlite3.connect(self.database)) as db:
                previous = db.execute("SELECT output_path,output_sha256 FROM forecasts WHERE fingerprint=?",
                                       (fingerprint,)).fetchone()
            if previous:
                path = self.cfg["_root"] / previous[0]
                if path.is_file() and sha256_file(path) == previous[1]:
                    self.event(issue, "UNCHANGED", {"fingerprint": fingerprint})
                    return pd.read_csv(path, float_precision="round_trip")
            prediction = self.bundle["model"].predict(features)
            lower, upper = apply_intervals(features, prediction, self.bundle["calibration"])
            radius = np.maximum(prediction - lower, upper - prediction)
            radius *= 1 + features.fallback_age_hours.to_numpy() / 48
            lower, upper = np.maximum(0, prediction - radius), np.minimum(1, prediction + radius)
            out = features[["turbine_id", "issue_time", "run_time", "available_at", "valid_time", "lead_hours",
                            "weather_lead_hours", "availability_verified", "availability_basis", "weather_sha256",
                            "wind_speed_100m", "fallback_age_hours"]].copy()
            out["power_norm"] = prediction
            out["lower_80"] = lower
            out["upper_80"] = upper
            out["wide_interval"] = upper - lower > self.cfg["agent"]["wide_interval_threshold"]
            out["ramp_alert"] = out.groupby("turbine_id").power_norm.diff().abs().fillna(0) > self.cfg["agent"]["ramp_threshold"]
            out["model_sha256"] = self.model_hash
            out["forecast_id"] = fingerprint
            expected = len(times) * len(self.cfg["turbines"])
            if len(out) != expected or out.duplicated(["turbine_id", "valid_time"]).any():
                raise ValueError("Incomplete or duplicate forecast rows")
            if not np.isfinite(out[["power_norm", "lower_80", "upper_80"]]).all().all():
                raise ValueError("Non-finite forecast")
            if not ((out.lower_80 <= out.power_norm) & (out.power_norm <= out.upper_80) &
                    (out.lower_80 >= 0) & (out.upper_80 <= 1)).all():
                raise ValueError("Invalid prediction intervals")
            summary = {"weather_failed": False, "rows": len(out), "ramp_alerts": int(out.ramp_alert.sum()),
                       "wide_intervals": int(out.wide_interval.sum()),
                       "verified_availability": bool(out.availability_verified.all()),
                       "max_fallback_age_hours": int(out.fallback_age_hours.max())}
            decision = decide(summary, self.use_llm)
            self.event(issue, "ANALYZE", {**summary, **decision})
            if decision["action"] != "publish":
                raise RuntimeError("Planner halted publication")
            path = self.output / f"forecast_{issue.strftime('%Y%m%dT%H%M%SZ')}_{fingerprint[:10]}.csv"
            write_csv(path, out)
            relative = path.relative_to(self.cfg["_root"]).as_posix()
            with closing(sqlite3.connect(self.database)) as db, db:
                db.execute("INSERT OR REPLACE INTO forecasts VALUES(?,?,?,?,?,?)",
                           (fingerprint, issue.isoformat(), relative, sha256_file(path), self.model_hash,
                            pd.Timestamp.now(tz="UTC").isoformat()))
            self.event(issue, "PUBLISHED", {"path": relative, "forecast_id": fingerprint})
            return out
        except Exception as exc:
            self.event(issue, "FAILED", {"error": str(exc)})
            raise


def replay(cfg: dict[str, Any], offline: bool = False, use_llm: bool = False,
           refresh: bool = False) -> dict[str, Any]:
    agent = ForecastAgent(cfg, offline, use_llm, refresh)
    frames = []
    try:
        for day in pd.date_range(cfg["dataset"]["issue_start_date"], cfg["dataset"]["issue_end_date"], tz="UTC"):
            frames.append(agent.run(day + pd.Timedelta(hours=cfg["dataset"]["issue_hour_utc"])))
    finally:
        agent.close()
    combined = pd.concat(frames, ignore_index=True)
    for col in ["issue_time", "valid_time"]:
        combined[col] = pd.to_datetime(combined[col], utc=True)
    combined = combined.sort_values(["issue_time", "turbine_id", "valid_time"])
    write_csv(cfg["_root"] / "outputs/forecast_all_issues.csv", combined)
    feb = combined[(combined.valid_time >= utc(cfg["dataset"]["evaluation_start"])) &
                   (combined.valid_time < utc(cfg["dataset"]["evaluation_end_exclusive"]))]
    # One latest day-ahead issue for each turbine and February hour.
    day_ahead = feb[feb.lead_hours <= 24].sort_values("issue_time").drop_duplicates(["turbine_id", "valid_time"], keep="last")
    write_csv(cfg["_root"] / "outputs/submission_february.csv", day_ahead)
    # This index assumes equal rated capacities. It is not MW or a verified farm total.
    farm = day_ahead.groupby("valid_time", as_index=False).agg(
        equal_weight_power_index=("power_norm", "mean"), turbines=("turbine_id", "nunique"))
    farm["capacity_assumption"] = "equal_weights_unverified_rated_capacities"
    write_csv(cfg["_root"] / "outputs/farm_equal_weight_index.csv", farm)
    report = {"issue_count": len(frames), "forecast_rows": len(combined), "february_day_ahead_rows": len(day_ahead),
              "february_hours": len(farm), "max_fallback_age_hours": int(combined.fallback_age_hours.max()),
              "availability_verified": bool(combined.availability_verified.all()), "planner": "openai" if use_llm else "deterministic",
              "submission_sha256": sha256_file(cfg["_root"] / "outputs/submission_february.csv")}
    write_json(cfg["_root"] / "reports/replay_report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Autonomous daily forecast replay or one issue")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--issue-time")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--llm", action="store_true")
    args = parser.parse_args()
    try:
        cfg = load_config(args.config)
        setup_logging(cfg)
        if args.issue_time:
            agent = ForecastAgent(cfg, args.offline, args.llm, args.refresh)
            try:
                agent.run(utc(args.issue_time))
            finally:
                agent.close()
        else:
            replay(cfg, args.offline, args.llm, args.refresh)
        return 0
    except Exception:
        LOG.exception("Agent failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
