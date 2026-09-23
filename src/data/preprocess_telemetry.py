from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.common import load_config, resolve_path, setup_logging, sha256_file, write_csv, write_json

LOG = logging.getLogger(__name__)


def parse_times(values: pd.Series, timezone: str) -> pd.Series:
    """Reject mixed naive/aware time rather than silently shifting some rows."""
    strings = values.astype(str).str.strip()
    aware = strings.str.contains(r"(?:Z|[+-]\d{2}:?\d{2})$", regex=True)
    if aware.any() and not aware.all():
        raise ValueError("Mixed timezone-aware and naive timestamps in CSV")
    parsed = pd.to_datetime(strings, format="mixed", errors="raise", utc=bool(aware.all()))
    if parsed.isna().any():
        raise ValueError("Missing timestamp")
    if not aware.all():
        parsed = parsed.dt.tz_localize(timezone, ambiguous="raise", nonexistent="raise")
    return parsed.dt.tz_convert("UTC")


def preprocess_telemetry(path: Path, turbine_id: str,
                         settings: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Average equal-duration slots; retain every hour and its coverage indicators."""
    raw = pd.read_csv(path, encoding="utf-8-sig")
    mapping = settings["columns"]
    missing = set(mapping.values()) - set(raw.columns)
    if missing or raw.empty:
        raise ValueError(f"{path}: empty data or missing columns: {sorted(missing)}")
    x = raw[list(mapping.values())].rename(columns={v: k for k, v in mapping.items()})
    x["time"] = parse_times(x["time"], settings["source_timezone"])
    if settings["timestamp_convention"] == "interval_end":
        x["time"] -= pd.Timedelta(minutes=settings["interval_minutes"])
    if not x["time"].eq(x["time"].dt.floor("10min")).all():
        raise ValueError(f"{turbine_id}: timestamps are not on the 10-minute grid")
    exact_duplicates = int(x.duplicated().sum())
    x = x.drop_duplicates()
    if x["time"].duplicated().any():
        raise ValueError(f"{turbine_id}: conflicting readings at the same timestamp")
    x = x.sort_values("time").set_index("time")
    invalid: dict[str, int] = {}
    for col, bounds in settings["bounds"].items():
        numeric = pd.to_numeric(x[col], errors="coerce")
        bad = ~np.isfinite(numeric) | ~numeric.between(*bounds)
        invalid[col] = int(bad.sum())
        x[col] = numeric.mask(bad)
    # Six equally weighted interval averages constitute one full hour.
    resampler = x.resample("h", label="left", closed="left")
    means = resampler.mean()
    counts = resampler.count()
    output = pd.DataFrame(index=means.index)
    output["samples_present"] = resampler.size()
    for col in settings["bounds"]:
        output[f"{col}_valid_samples"] = counts[col]
        output[col] = means[col].where(counts[col] >= settings["min_valid_samples"])
    output["coverage_power"] = counts["power_norm"] / 6.0
    output["target_usable"] = output["power_norm"].notna()
    output["hour_complete"] = counts["power_norm"].eq(6)
    output["turbine_id"] = turbine_id
    output = output.reset_index().rename(columns={"time": "valid_time"})
    expected_slots = int((x.index.max() - x.index.min()) / pd.Timedelta(minutes=10)) + 1
    report = {
        "turbine_id": turbine_id, "input_sha256": sha256_file(path),
        "input_rows": len(raw), "exact_duplicates_removed": exact_duplicates,
        "missing_10min_slots": expected_slots - len(x), "invalid_measurements": invalid,
        "source_timezone": settings["source_timezone"],
        "timestamp_convention": settings["timestamp_convention"],
        "start_utc": str(x.index.min()), "end_utc": str(x.index.max()),
        "hours_total": len(output), "hours_complete": int(output["hour_complete"].sum()),
        "hours_target_usable": int(output["target_usable"].sum()),
        "hours_without_readings": int(output["samples_present"].eq(0).sum()),
        "min_valid_samples": settings["min_valid_samples"],
    }
    LOG.info("%s: %d input rows -> %d hours; usable targets=%d", turbine_id,
             len(raw), len(output), report["hours_target_usable"])
    return output, report


def prepare_all(cfg: dict[str, Any]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    frames: list[pd.DataFrame] = []
    reports: list[dict[str, Any]] = []
    folder = resolve_path(cfg, cfg["paths"]["processed"])
    for turbine in cfg["turbines"]:
        frame, report = preprocess_telemetry(resolve_path(cfg, turbine["telemetry_file"]),
                                              turbine["id"], cfg["telemetry"])
        write_csv(folder / f"{turbine['id']}_hourly.csv", frame)
        write_json(folder / f"{turbine['id']}_quality.json", report)
        frames.append(frame)
        reports.append(report)
    return pd.concat(frames, ignore_index=True), reports


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate and aggregate turbine CSVs")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    try:
        cfg = load_config(args.config)
        setup_logging(cfg)
        prepare_all(cfg)
        return 0
    except Exception:
        LOG.exception("Telemetry preprocessing failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
