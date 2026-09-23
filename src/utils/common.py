from __future__ import annotations

import hashlib
import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import tempfile
from typing import Any

import pandas as pd
import yaml


def utc(value: Any) -> pd.Timestamp:
    """Require explicit timezone for machine-facing timestamps."""
    stamp = pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise ValueError(f"Timezone-aware timestamp required: {value!r}")
    return stamp.tz_convert("UTC")


def load_config(path: str | Path = "config/config.yaml") -> dict[str, Any]:
    file = Path(path).resolve()
    with file.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError("Configuration must be a YAML mapping")
    cfg["_root"] = file.parent.parent
    if cfg["project"]["timezone"] != "UTC":
        raise ValueError("All output timestamps must use UTC")
    t = cfg["telemetry"]
    if t["interval_minutes"] != 10 or not 1 <= t["min_valid_samples"] <= 6:
        raise ValueError("Expected 10-minute intervals and min_valid_samples in 1..6")
    if t["timestamp_convention"] not in {"interval_start", "interval_end"}:
        raise ValueError("Invalid timestamp_convention")
    if not cfg["turbines"] or len({x["id"] for x in cfg["turbines"]}) != len(cfg["turbines"]):
        raise ValueError("Turbine IDs must be nonempty and unique")
    for turbine in cfg["turbines"]:
        if not str(turbine["id"]).isalnum():
            raise ValueError("Turbine IDs must be alphanumeric")
    return cfg


def resolve_path(cfg: dict[str, Any], value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else cfg["_root"] / path


def setup_logging(cfg: dict[str, Any]) -> None:
    folder = resolve_path(cfg, cfg["paths"]["logs"])
    folder.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=cfg["project"]["log_level"],
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[logging.StreamHandler(), RotatingFileHandler(
            folder / "pipeline.log", maxBytes=5_000_000, backupCount=3, encoding="utf-8")],
        force=True,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="",
                                         dir=path.parent, delete=False) as handle:
            name = handle.name
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if name is not None and os.path.exists(name):
            os.unlink(name)


def write_json(path: Path, content: dict[str, Any]) -> None:
    atomic_write(path, json.dumps(content, ensure_ascii=False, indent=2, default=str))


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    atomic_write(path, frame.to_csv(index=False))
