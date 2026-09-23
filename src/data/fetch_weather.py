from __future__ import annotations

import argparse
from contextlib import closing
import hashlib
import json
import logging
from pathlib import Path
import sqlite3
from typing import Any

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.utils.common import load_config, resolve_path, setup_logging, utc, write_csv

LOG = logging.getLogger(__name__)
VARIABLES = ["wind_speed_10m", "wind_speed_100m", "wind_direction_10m",
             "wind_direction_100m", "temperature_2m", "surface_pressure"]
UNITS = {"wind_speed_10m": "m/s", "wind_speed_100m": "m/s",
         "wind_direction_10m": "°", "wind_direction_100m": "°",
         "temperature_2m": "°C", "surface_pressure": "hPa"}


class WeatherUnavailableError(RuntimeError):
    """Transport/HTTP failure for which an older available run may be tried."""


def select_run(issue_time: pd.Timestamp, cfg: dict[str, Any]) -> tuple[pd.Timestamp, pd.Timestamp, str]:
    """Return run time, available-at estimate/evidence, provenance basis."""
    issue = utc(issue_time)
    w = cfg["weather"]
    if w["availability_policy"] == "manifest":
        if not w.get("availability_csv"):
            raise ValueError("manifest policy requires weather.availability_csv")
        records = pd.read_csv(resolve_path(cfg, w["availability_csv"]))
        required = {"run_time", "available_at", "source", "model"}
        if not required.issubset(records.columns):
            raise ValueError(f"Availability manifest requires {required}")
        records["run_time"] = records["run_time"].map(utc)
        records["available_at"] = records["available_at"].map(utc)
        records = records[records["model"].eq(w["model"])]
        if (records["available_at"] < records["run_time"]).any():
            raise ValueError("available_at precedes run_time")
        records = records[(records["available_at"] <= issue) & (records["run_time"] <= issue)]
        records = records.dropna(subset=["source"])
        records = records[records["source"].astype(str).str.strip().ne("")]
        if records.empty:
            raise ValueError(f"No evidenced weather run available at {issue}")
        row = records.sort_values("run_time").iloc[-1]
        return utc(row["run_time"]), utc(row["available_at"]), "manifest:" + str(row["source"])
    if w["availability_policy"] != "assumed_delay":
        raise ValueError("availability_policy must be manifest or assumed_delay")
    if w["require_verified_availability"]:
        raise ValueError("Verified availability required: provide an evidence manifest")
    delay = float(w["availability_delay_hours"])
    if delay < 0 or w["run_cycle_hours"] != 6:
        raise ValueError("Invalid run delay or ECMWF cycle")
    run = (issue - pd.Timedelta(hours=delay)).floor("6h")
    return run, run + pd.Timedelta(hours=delay), "assumed_delay"


def parse_response(payload: dict[str, Any], expected: pd.DatetimeIndex,
                   variables: list[str]) -> pd.DataFrame:
    if payload.get("error"):
        raise ValueError(f"Open-Meteo error: {payload.get('reason')}")
    if payload.get("utc_offset_seconds") != 0:
        raise ValueError("API returned non-UTC data")
    hourly = payload.get("hourly")
    if not isinstance(hourly, dict) or not set(["time", *variables]).issubset(hourly):
        raise ValueError("API response is missing requested hourly variables")
    for name in variables:
        if payload.get("hourly_units", {}).get(name) != UNITS[name]:
            raise ValueError(f"Unexpected unit for {name}")
    frame = pd.DataFrame(hourly)
    frame["valid_time"] = pd.to_datetime(frame.pop("time"), utc=True, errors="raise")
    if frame["valid_time"].duplicated().any():
        raise ValueError("Duplicate weather timestamps")
    if not frame["valid_time"].eq(frame["valid_time"].dt.floor("h")).all():
        raise ValueError("Weather timestamps must be aligned to whole hours")
    frame = frame.set_index("valid_time").reindex(expected)
    numeric = frame[variables].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("Weather response has missing hours, nulls or non-finite values")
    bounds = {"wind_speed_10m": (0, 150), "wind_speed_100m": (0, 150),
              "wind_direction_10m": (0, 360), "wind_direction_100m": (0, 360),
              "temperature_2m": (-100, 70), "surface_pressure": (100, 1200)}
    for name, limits in bounds.items():
        if not numeric[name].between(*limits).all():
            raise ValueError(f"Weather values outside physical sanity bounds: {name}")
    numeric["air_density"] = (numeric["surface_pressure"] * 100.0 /
                               (287.05 * (numeric["temperature_2m"] + 273.15)))
    numeric.index.name = "valid_time"
    return numeric.reset_index()


class WeatherClient:
    """SQLite cache stores immutable responses, parameters, hash and fetch time."""

    def __init__(self, cfg: dict[str, Any], offline: bool = False,
                 refresh: bool = False, session: requests.Session | None = None) -> None:
        self.cfg = cfg
        self.w = cfg["weather"]
        self.offline = offline
        self.refresh = refresh
        if offline and refresh:
            raise ValueError("--offline and --refresh are mutually exclusive")
        if self.w["endpoint"] != "https://single-runs-api.open-meteo.com/v1/forecast":
            raise ValueError("Replay requires Single Runs API, not a stitched historical series")
        if self.w["model"] != "ecmwf_ifs" or set(self.w["variables"]) != set(VARIABLES):
            raise ValueError("This data contract requires ecmwf_ifs and all six weather variables")
        self.cache = resolve_path(cfg, cfg["paths"]["cache"])
        self.cache.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.cache, timeout=30)) as db, db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("""CREATE TABLE IF NOT EXISTS weather_cache (
                cache_key TEXT PRIMARY KEY, request_json TEXT NOT NULL,
                payload_json TEXT NOT NULL, payload_sha256 TEXT NOT NULL,
                fetched_at TEXT NOT NULL)""")
        self.session = session if session is not None else requests.Session()
        self.owns_session = session is None
        if session is None:
            retry = Retry(total=int(self.w["retries"]),
                          backoff_factor=float(self.w["backoff_seconds"]),
                          status_forcelist=[429, 500, 502, 503, 504],
                          allowed_methods=frozenset(["GET"]), respect_retry_after_header=True)
            self.session.mount("https://", HTTPAdapter(max_retries=retry))

    def close(self) -> None:
        if self.owns_session:
            self.session.close()

    def fetch(self, turbine: dict[str, Any], issue_time: pd.Timestamp,
              valid_times: pd.DatetimeIndex) -> pd.DataFrame:
        issue = utc(issue_time)
        if len(valid_times) == 0 or valid_times.tz is None:
            raise ValueError("Nonempty timezone-aware forecast window required")
        times = valid_times.tz_convert("UTC")
        if not times.equals(pd.date_range(times.min(), periods=len(times), freq="h")):
            raise ValueError("Forecast window must be ordered, unique and hourly")
        if times.min() < issue or not times.equals(times.floor("h")):
            raise ValueError("Forecast window is in the past or not hour-aligned")
        run, available, basis = select_run(issue, self.cfg)
        if run.hour not in {0, 6, 12, 18} or run != run.floor("h"):
            raise ValueError("Invalid ECMWF run cycle")
        if run < utc(self.w["archive_start"]):
            raise ValueError("Selected run precedes configured archive coverage")
        lat, lon = float(turbine["latitude"]), float(turbine["longitude"])
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            raise ValueError("Invalid turbine coordinates")
        params = {
            "latitude": lat, "longitude": lon, "models": self.w["model"],
            "run": run.strftime("%Y-%m-%dT%H:%M"),
            "hourly": ",".join(self.w["variables"]), "timezone": "UTC",
            "wind_speed_unit": "ms", "temperature_unit": "celsius",
            "forecast_days": 7,
        }
        request_text = json.dumps({"endpoint": self.w["endpoint"], "params": params}, sort_keys=True)
        key = hashlib.sha256(request_text.encode()).hexdigest()
        with closing(sqlite3.connect(self.cache, timeout=30)) as db:
            cached = db.execute("SELECT request_json, payload_json, payload_sha256, fetched_at FROM weather_cache "
                                "WHERE cache_key=?", (key,)).fetchone()
        # A refresh must not hide an existing integrity failure. The request
        # and payload are both part of the cache's provenance contract.
        if cached is not None:
            stored_request, stored_body, stored_digest, _ = cached
            if (hashlib.sha256(stored_request.encode()).hexdigest() != key or
                    stored_request != request_text or
                    hashlib.sha256(stored_body.encode()).hexdigest() != stored_digest):
                raise ValueError(f"Corrupt weather cache entry {key}")
        cache_hit = cached is not None and not self.refresh
        if cache_hit:
            _, body, digest, fetched_at = cached
            payload = json.loads(body)
            LOG.info("Weather cache hit turbine=%s run=%s", turbine["id"], run)
        else:
            if self.offline:
                raise FileNotFoundError(f"Weather cache miss in offline mode: {turbine['id']} {run}")
            LOG.info("Fetching weather turbine=%s run=%s", turbine["id"], run)
            try:
                response = self.session.get(self.w["endpoint"], params=params,
                    timeout=(self.w["connect_timeout_seconds"], self.w["read_timeout_seconds"]))
                response.raise_for_status()
            except requests.RequestException as exc:
                raise WeatherUnavailableError(f"Weather retrieval failed for {turbine['id']} {run}: {exc}") from exc
            # Malformed responses are validation failures, not permission to
            # conceal the problem by falling back to a different weather run.
            try:
                payload = response.json()
            except ValueError as exc:
                raise ValueError(f"Invalid weather JSON for {turbine['id']} {run}") from exc
            if not isinstance(payload, dict):
                raise ValueError("Expected single-location JSON object")
            body = json.dumps(payload, sort_keys=True, allow_nan=False)
            digest = hashlib.sha256(body.encode()).hexdigest()
            fetched_at = pd.Timestamp.now(tz="UTC").isoformat()
        frame = parse_response(payload, times, self.w["variables"])
        if not cache_hit:
            with closing(sqlite3.connect(self.cache, timeout=30)) as db, db:
                # Refresh that changes a historical response must be reviewed explicitly.
                old = db.execute("SELECT request_json, payload_json, payload_sha256 FROM weather_cache WHERE cache_key=?", (key,)).fetchone()
                if old and (old[0] != request_text or hashlib.sha256(old[0].encode()).hexdigest() != key or
                            hashlib.sha256(old[1].encode()).hexdigest() != old[2]):
                    raise ValueError(f"Corrupt weather cache entry {key}")
                if old and old[2] != digest:
                    previous = json.loads(old[1])
                    if previous.get("hourly") != payload.get("hourly"):
                        raise ValueError("Archived hourly data changed; preserve cache and investigate")
                db.execute("INSERT OR IGNORE INTO weather_cache VALUES (?, ?, ?, ?, ?)",
                           (key, request_text, body, digest, fetched_at))
                stored = db.execute("SELECT request_json, payload_json, payload_sha256, fetched_at FROM weather_cache WHERE cache_key=?", (key,)).fetchone()
                stored_request, stored_body, digest, fetched_at = stored
                if (stored_request != request_text or hashlib.sha256(stored_request.encode()).hexdigest() != key or
                        hashlib.sha256(stored_body.encode()).hexdigest() != digest):
                    raise ValueError(f"Corrupt weather cache entry {key}")
                payload = json.loads(stored_body)
                frame = parse_response(payload, times, self.w["variables"])
        frame["turbine_id"] = turbine["id"]
        frame["issue_time"] = issue
        frame["run_time"] = run
        frame["available_at"] = available
        frame["availability_basis"] = basis
        frame["availability_verified"] = basis.startswith("manifest:")
        frame["model"] = self.w["model"]
        frame["cache_key"] = key
        frame["weather_sha256"] = digest
        frame["fetched_at"] = fetched_at
        frame["requested_latitude"] = lat
        frame["requested_longitude"] = lon
        frame["grid_latitude"] = payload.get("latitude")
        frame["grid_longitude"] = payload.get("longitude")
        frame["lead_hours"] = (frame["valid_time"] - issue).dt.total_seconds() / 3600
        frame["weather_lead_hours"] = (frame["valid_time"] - run).dt.total_seconds() / 3600
        return frame


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch one archived weather run for all turbines")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--issue-time", required=True, help="ISO timestamp with timezone")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    client: WeatherClient | None = None
    try:
        cfg = load_config(args.config)
        setup_logging(cfg)
        issue = utc(args.issue_time)
        times = pd.date_range(issue.floor("h") + pd.Timedelta(hours=1),
                              periods=cfg["dataset"]["horizon_hours"], freq="h")
        client = WeatherClient(cfg, args.offline, args.refresh)
        for turbine in cfg["turbines"]:
            frame = client.fetch(turbine, issue, times)
            target = resolve_path(cfg, cfg["paths"]["processed"]) / (
                f"{turbine['id']}_weather_{issue.strftime('%Y%m%dT%H%M%SZ')}.csv")
            write_csv(target, frame)
        return 0
    except Exception:
        LOG.exception("Weather fetch failed")
        return 1
    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    raise SystemExit(main())
