"""Deterministic SYNTHETIC data, exclusively for testing and demonstrations."""
import csv
from datetime import datetime, timedelta
import hashlib
import io
import math
import random

from .common import UTC, digest, iso, now, utc

WARNING = "УЧЕБНЫЕ ДАННЫЕ: история и погода синтетические; результат не является прогнозом реальной ВЭС."


def signal(at, turbine_id):
    hour = (utc(at) - datetime(2025, 11, 1, tzinfo=UTC)).total_seconds() / 3600
    shift = int(turbine_id) * 0.12
    wind = max(0.1, 7.1 + 3.8 * math.sin(hour / 31) + 1.8 * math.cos(hour / 9 + shift))
    temperature = -4 + 7 * math.cos(hour / 150) + 2 * math.sin(hour / 24)
    return wind, temperature


def history_csv(config):
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(["timestamp", "turbine_id", "wind_speed", "temperature", "power"])
    rng = random.Random(202602)
    stop = utc(config["training_cutoff"])
    start = stop - timedelta(days=92) + timedelta(hours=1)
    for hour in range(92 * 24):
        at = start + timedelta(hours=hour)
        for target in config["turbines"]:
            wind, temperature = signal(at, target["id"])
            wind = max(0, wind + rng.gauss(0, 0.35))
            density_wind = wind * (288.15 / (273.15 + temperature)) ** (1 / 3)
            power = min(1, max(0, (density_wind ** 3 - 3 ** 3) / (12 ** 3 - 3 ** 3)))
            power = min(1, max(0, power + rng.gauss(0, 0.025)))
            writer.writerow([iso(at), target["id"], round(wind, 4), round(temperature, 4), round(power, 6)])
    return stream.getvalue()


def fetch_demo_weather(as_of, targets, hours, **_):
    as_of = utc(as_of)
    rows = []
    for target in targets:
        seed = int(hashlib.sha256((iso(as_of) + target["id"]).encode()).hexdigest()[:16], 16)
        rng = random.Random(seed)
        for lead in range(1, hours + 1):
            at = as_of + timedelta(hours=lead)
            wind, temperature = signal(at, target["id"])
            rows.append({"turbine_id": target["id"], "valid_time": iso(at),
                         "wind_speed": round(max(0, wind + rng.gauss(0, 0.5 + lead / 100)), 4),
                         "temperature": round(temperature + rng.gauss(0, 1), 4)})
    return {"rows": rows, "warnings": [WARNING], "provenance": [
        {"turbine_id": t["id"], "source": "synthetic_demo", "model": "deterministic_fixture_v1",
         "run_time": iso(as_of - timedelta(hours=12)), "available_at": iso(as_of),
         "availability_basis": "synthetic_fixture_not_real_weather", "url": None,
         "sha256": digest([r for r in rows if r["turbine_id"] == t["id"]]),
         "retrieved_at": now(), "competition_ready": False} for t in targets]}
