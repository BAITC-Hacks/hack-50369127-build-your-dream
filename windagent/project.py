"""Data-only import of the supplied project and native-model offline inference.

Archive Python scripts and joblib files are never imported or executed. This
adapter independently reconstructs the inspected forecast-only feature contract.
"""
from contextlib import closing
import csv
from datetime import timedelta
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import sqlite3
import subprocess
import sys
import tempfile
from urllib.parse import urlencode
import zipfile

from .common import ROOT, digest, encoded, iso, now, read_json, save_json, utc

VARIABLES = ["wind_speed_10m", "wind_speed_100m", "wind_direction_10m", "wind_direction_100m", "temperature_2m", "surface_pressure"]
FEATURE_NAMES = ["turbine_id", *VARIABLES, "air_density", "lead_hours", "weather_lead_hours",
    "hour", "month", "dayofweek", "hour_sin", "hour_cos", "month_sin", "month_cos", "dayofweek_sin", "dayofweek_cos",
    "wind_direction_10m_sin", "wind_direction_10m_cos", "wind_direction_100m_sin", "wind_direction_100m_cos",
    "wind100_cubed", "wind10_cubed", "density_adjusted_wind", "wind_shear_ratio", "wind_speed_difference",
    "wind100_u", "wind100_v", "forecast_wind_delta", "forecast_wind_mean3", "dayofyear_sin", "dayofyear_cos"]
ENDPOINT = "https://single-runs-api.open-meteo.com/v1/forecast"
WARNINGS = [
    "Часовой пояс исходной телеметрии не указан. UTC и начало 10-минутного интервала сохранены как допущения исходного проекта.",
    "Доступность погоды принята как время запуска + 8 ч; историческая публикация и оперативное происхождение выпусков не подтверждены.",
    "Фактических значений февраля в исходных CSV нет. Метрики относятся к январю 2026, а не к тестовому февралю.",
    "Интервалы — эмпирические полосы с целевым покрытием 80%, калиброванные на январе; покрытие будущих значений не гарантируется.",
]
ASSETS = ["data/raw/turbine_1.csv", "data/raw/turbine_2.csv", "data/cache/weather.sqlite",
    "artifacts/catboost.cbm", "reports/training_report.json", "reports/january_predictions.csv",
    "outputs/forecast_all_issues.csv", "outputs/submission_february.csv", "config/config.yaml"]


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def install_archive(path, destination, external_turbines=None):
    """Verify every manifest entry, then extract only whitelisted data assets."""
    path = Path(path)
    archive_sha = sha_file(path)
    folder = Path(destination) / archive_sha[:20]
    with zipfile.ZipFile(path) as source:
        entries = source.infolist()
        names = [item.filename for item in entries]
        if len(names) != len(set(names)) or sum(item.file_size for item in entries) > 300_000_000:
            raise ValueError("Некорректный архив: дубликаты имён или превышен допустимый размер 300 МБ.")
        prefix = "wind-forecast/"
        for item in entries:
            rel = PurePosixPath(item.filename)
            if rel.is_absolute() or ".." in rel.parts or "\\" in item.filename:
                raise ValueError("Архив содержит недопустимое имя файла.")
        manifest = source.read(prefix + "SHA256SUMS.txt").decode("utf-8-sig")
        hashes = {}
        for line in manifest.splitlines():
            if not line.strip():
                continue
            expected, rel = line.split(maxsplit=1)
            rel = rel.lstrip("* ")
            if rel.startswith("./"):
                rel = rel[2:]
            if len(expected) != 64 or rel in hashes:
                raise ValueError("Некорректный манифест SHA-256.")
            body = source.read(prefix + rel)
            if hashlib.sha256(body).hexdigest() != expected:
                raise ValueError("Нарушена контрольная сумма файла архива: " + rel)
            hashes[rel] = expected
        for asset in ASSETS:
            if asset not in hashes:
                raise ValueError("В проверенном манифесте нет обязательного файла: " + asset)
        if external_turbines:
            for turbine_id, original in external_turbines.items():
                if turbine_id not in {"1", "2"} or sha_file(original) != hashes[f"data/raw/turbine_{turbine_id}.csv"]:
                    raise ValueError("Отдельный CSV не совпадает с данными обученного проекта: турбина " + turbine_id)
        for asset in ASSETS:
            target = folder / asset
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and sha_file(target) != hashes[asset]:
                raise ValueError("Изменён ранее импортированный исходный файл: " + asset)
            if not target.exists():
                target.write_bytes(source.read(prefix + asset))
    info = {"archive_path": str(path), "archive_sha256": archive_sha, "folder": str(folder),
            "asset_hashes": {key: hashes[key] for key in ASSETS}, "manifest_files_verified": len(hashes),
            "imported_at": now(), "timezone_assumption": "UTC", "interval_convention": "start",
            "availability_lag_hours": 8, "training_cutoff": "2026-01-31T23:00:00+00:00"}
    info["weather_audit"] = audit_cache(folder / "data/cache/weather.sqlite")
    info["january_audit"] = audit_january(folder / "reports/january_predictions.csv",
                                          read_json(folder / "reports/training_report.json"))
    save_json(folder / "import_manifest.json", info)
    return info


def audit_cache(path):
    index = []
    with closing(sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)) as database:
        if database.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError("SQLite-кеш погоды повреждён.")
        for key, request, body, expected, retrieved in database.execute("SELECT cache_key,request_json,payload_json,payload_sha256,fetched_at FROM weather_cache"):
            if hashlib.sha256(body.encode()).hexdigest() != expected or hashlib.sha256(request.encode()).hexdigest() != key:
                raise ValueError("Контрольная сумма погодного ответа или запроса не совпадает.")
            meta = json.loads(request)
            if meta["endpoint"] != ENDPOINT or meta["params"]["models"] != "ecmwf_ifs":
                raise ValueError("Кеш содержит несовместимый источник погоды.")
            index.append({"key": key, "request": meta, "sha256": expected, "retrieved_at": retrieved})
    if not index:
        raise ValueError("Кеш погодных выпусков пуст.")
    return {"records": len(index), "integrity": "ok", "index": index, "availability_verified": False}


def audit_january(path, report):
    groups = {}
    seen = set()
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            stamp = utc(row["valid_time"])
            if not utc("2026-01-01T00:00:00Z") <= stamp < utc("2026-02-01T00:00:00Z"):
                raise ValueError("Январский отчёт содержит дату за пределами января.")
            key = (row["turbine_id"], utc(row["issue_time"]), stamp)
            if key in seen or not key[1] < stamp:
                raise ValueError("Повтор или некорректный горизонт в январском отчёте.")
            lead = float(row["lead_hours"])
            if row["turbine_id"] not in {"T1", "T2"} or not 1 <= lead <= 48 or lead != int(lead) or lead != (stamp - key[1]).total_seconds() / 3600:
                raise ValueError("Некорректная турбина или горизонт в январском отчёте.")
            seen.add(key)
            actual, prediction = float(row["target_power_norm"]), float(row["prediction"])
            if not all(math.isfinite(x) and 0 <= x <= 1 for x in (actual, prediction)):
                raise ValueError("Некорректная мощность в январском отчёте.")
            bucket = "01-24" if lead <= 24 else "25-48"
            for group in (("ALL", "ALL"), (row["turbine_id"], bucket)):
                groups.setdefault(group, []).append(prediction - actual)
    if not seen:
        raise ValueError("Январский отчёт пуст.")
    verified = []
    for (turbine, bucket), errors in groups.items():
        metrics = {"model": "selected_model", "turbine_id": turbine, "horizon": bucket, "n": len(errors),
            "mae": math.fsum(abs(x) for x in errors) / len(errors),
            "rmse": math.sqrt(math.fsum(x * x for x in errors) / len(errors)), "bias": math.fsum(errors) / len(errors)}
        saved = next(r for r in report["january_metrics"] if r["model"] == "selected_model" and r["turbine_id"] == turbine and r["horizon"] == bucket)
        if any(abs(metrics[k] - saved[k]) > 1e-10 for k in ("mae", "rmse", "bias")) or metrics["n"] != saved["n"]:
            raise ValueError("Январские метрики не совпадают с сохранёнными прогнозами.")
        verified.append(metrics)
    return {"metrics": verified, "rows": len(seen), "arithmetic_verified": True,
            "independent_retraining_verified": False, "note": "Метрики независимо пересчитаны по сохранённым январским прогнозам; повторное обучение январской модели ещё не выполнено."}


def model_python():
    configured = os.environ.get("WIND_AGENT_MODEL_PYTHON")
    if configured:
        if not Path(configured).is_file():
            raise ValueError("Не найден Python из WIND_AGENT_MODEL_PYTHON.")
        return configured
    local = ROOT / ".venv-model" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    task = Path(tempfile.gettempdir()) / "wind-agent/project-env-py312" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    for candidate in (local, task):
        environment = candidate.parent.parent
        installed = list(environment.glob("Lib/site-packages/catboost/__init__.py")) + list(environment.glob("lib/python*/site-packages/catboost/__init__.py"))
        if candidate.is_file() and installed:
            return str(candidate)
    return None


def make_features(weather_rows, as_of):
    """Reconstruct the archive's exact numerical feature allowlist, in UTC."""
    result, histories = [], {}
    for row in sorted(weather_rows, key=lambda r: (r["turbine_id"], utc(r["valid_time"]))):
        valid, run = utc(row["valid_time"]), utc(row["run_time"])
        values = {name: float(row[name]) for name in VARIABLES}
        values.update(turbine_id="T" + row["turbine_id"], air_density=values["surface_pressure"] * 100 / (287.05 * (values["temperature_2m"] + 273.15)),
            lead_hours=(valid - utc(as_of)).total_seconds() / 3600, weather_lead_hours=(valid - run).total_seconds() / 3600,
            hour=valid.hour, month=valid.month, dayofweek=valid.weekday())
        for name, period, offset in (("hour", 24, 0), ("month", 12, 1), ("dayofweek", 7, 0), ("wind_direction_10m", 360, 0), ("wind_direction_100m", 360, 0)):
            angle = 2 * math.pi * (values[name] - offset) / period
            values[name + "_sin"], values[name + "_cos"] = math.sin(angle), math.cos(angle)
        wind, low = values["wind_speed_100m"], values["wind_speed_10m"]
        history = histories.setdefault(row["turbine_id"], [])
        previous = history[-1] if history else wind
        history.append(wind)
        angle = math.radians(values["wind_direction_100m"])
        day = 2 * math.pi * (valid.timetuple().tm_yday - 1) / 365.25
        values.update(wind100_cubed=wind ** 3, wind10_cubed=low ** 3,
            density_adjusted_wind=wind * (values["air_density"] / 1.225) ** (1 / 3),
            wind_shear_ratio=wind / (low + .5), wind_speed_difference=wind - low,
            wind100_u=-wind * math.sin(angle), wind100_v=-wind * math.cos(angle),
            forecast_wind_delta=wind - previous, forecast_wind_mean3=sum(history[-3:]) / len(history[-3:]),
            dayofyear_sin=math.sin(day), dayofyear_cos=math.cos(day))
        if not all(math.isfinite(values[k]) for k in FEATURE_NAMES if k != "turbine_id"):
            raise ValueError("Признаки модели содержат нечисловое значение.")
        result.append([values[name] for name in FEATURE_NAMES])
    return result


class ProjectEngine:
    def __init__(self, info, targets):
        self.info, self.targets = info, targets
        self.folder = Path(info["folder"])
        self.report = read_json(self.folder / "reports/training_report.json")
        for asset in ("artifacts/catboost.cbm", "reports/training_report.json"):
            if sha_file(self.folder / asset) != info["asset_hashes"][asset]:
                raise ValueError("Изменён исходный файл модели/калибровки; требуется новая проверка: " + asset)

    def weather(self, as_of, hours):
        issue = utc(as_of)
        cutoff = issue - timedelta(hours=8)
        run = cutoff.replace(hour=cutoff.hour // 6 * 6, minute=0, second=0, microsecond=0)
        times = [issue + timedelta(hours=i) for i in range(1, hours + 1)]
        rows, provenance = [], []
        with closing(sqlite3.connect((self.folder / "data/cache/weather.sqlite").resolve().as_uri() + "?mode=ro", uri=True)) as db:
            for target in self.targets:
                candidates = [entry for entry in self.info["weather_audit"]["index"]
                    if utc(entry["request"]["params"]["run"] + ":00Z") == run]
                candidates = sorted(candidates, key=lambda e: (float(e["request"]["params"]["latitude"]) - target["latitude"]) ** 2 + (float(e["request"]["params"]["longitude"]) - target["longitude"]) ** 2)
                if not candidates:
                    raise ValueError("В кеше проекта нет выпуска " + iso(run) + ". Чужой выпуск или фактическая погода не подставляются.")
                entry = candidates[0]
                params = entry["request"]["params"]
                if abs(params["latitude"] - target["latitude"]) > .001 or abs(params["longitude"] - target["longitude"]) > .001:
                    raise ValueError("В кеше нет координат выбранной турбины.")
                stored = db.execute("SELECT request_json,payload_json,payload_sha256,fetched_at FROM weather_cache WHERE cache_key=?", (entry["key"],)).fetchone()
                if stored is None or hashlib.sha256(stored[0].encode()).hexdigest() != entry["key"] or hashlib.sha256(stored[1].encode()).hexdigest() != stored[2] or stored[2] != entry["sha256"]:
                    raise ValueError("Изменён или повреждён кеш проекта; расчёт остановлен.")
                if json.loads(stored[0]) != entry["request"]:
                    raise ValueError("Индекс выпуска погоды не совпадает с проверенным исходным запросом.")
                payload = json.loads(stored[1])
                if payload.get("utc_offset_seconds") != 0 or payload.get("error"):
                    raise ValueError("Некорректный UTC-ответ погоды.")
                hourly = payload["hourly"]
                for name, unit in zip(VARIABLES, ["m/s", "m/s", "°", "°", "°C", "hPa"]):
                    if payload["hourly_units"].get(name) != unit or len(hourly[name]) != len(hourly["time"]):
                        raise ValueError("Несовместимые единицы или длины погодных массивов.")
                stamps = [utc(stamp + ":00Z" if len(stamp) == 16 else stamp) for stamp in hourly["time"]]
                if len(set(stamps)) != len(stamps):
                    raise ValueError("Повтор погодных часов в кеше.")
                lookup = {at: index for index, at in enumerate(stamps)}
                for valid in times:
                    if valid not in lookup:
                        raise ValueError("В погодном кеше отсутствует требуемый час " + iso(valid))
                    row = {"turbine_id": target["id"], "valid_time": iso(valid), "run_time": iso(run)}
                    for name, bounds in zip(VARIABLES, [(0, 150), (0, 150), (0, 360), (0, 360), (-100, 70), (100, 1200)]):
                        number = hourly[name][lookup[valid]]
                        if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number) or not bounds[0] <= number <= bounds[1]:
                            raise ValueError("Некорректное значение прогнозной погоды: " + name)
                        row[name] = number
                    row.update(wind_speed=row["wind_speed_100m"], temperature=row["temperature_2m"])
                    rows.append(row)
                provenance.append({"turbine_id": target["id"], "source": "Open-Meteo Single Runs · кеш предоставленного проекта",
                    "model": "ecmwf_ifs", "run_time": iso(run), "available_at": iso(run + timedelta(hours=8)),
                    "availability_basis": "assumed_delay_8h_unverified", "availability_verified": False,
                    "url": ENDPOINT + "?" + urlencode(params), "sha256": entry["sha256"], "retrieved_at": stored[3],
                    "cache_key": entry["key"], "cache_hit": True, "grid_latitude": payload.get("latitude"),
                    "grid_longitude": payload.get("longitude"), "requested_latitude": params["latitude"], "requested_longitude": params["longitude"]})
        return {"rows": sorted(rows, key=lambda r: (r["turbine_id"], r["valid_time"])), "provenance": provenance, "warnings": WARNINGS.copy()}

    def saved(self, as_of, weather):
        file = self.folder / "outputs/forecast_all_issues.csv"
        if sha_file(file) != self.info["asset_hashes"]["outputs/forecast_all_issues.csv"]:
            raise ValueError("Изменён сохранённый прогноз архива.")
        matches = {}
        with file.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                if utc(row["issue_time"]) == utc(as_of):
                    key = (row["turbine_id"].removeprefix("T"), utc(row["valid_time"]))
                    if key in matches:
                        raise ValueError("Повтор целевого часа в сохранённых прогнозах.")
                    matches[key] = row
        result = []
        for row in weather["rows"]:
            stored = matches.get((row["turbine_id"], utc(row["valid_time"])))
            if stored is None:
                raise ValueError("Для этого выпуска нет сохранённого прогноза. Установите среду CatBoost для нового расчёта.")
            provenance = next(p for p in weather["provenance"] if p["turbine_id"] == row["turbine_id"])
            if stored["weather_sha256"] != provenance["sha256"] or utc(stored["run_time"]) != utc(row["run_time"]):
                raise ValueError("Сохранённый прогноз относится к другому выпуску погоды.")
            result.append({**row, "power_pred": float(stored["power_norm"]), "lower": float(stored["lower_80"]), "upper": float(stored["upper_80"])})
        return result

    def predict(self, as_of, weather, execute=True):
        if utc(as_of) < utc(self.report["final_training_asof"]):
            raise ValueError("Импортированная модель обучена позже выбранного выпуска. Ранние выпуски требуют отдельного обучения.")
        weather = {**weather, "rows": sorted(weather["rows"], key=lambda r: (r["turbine_id"], utc(r["valid_time"])))}
        runner = model_python() if execute else None
        if runner is None:
            return self.saved(as_of, weather), "saved_archive", None
        request = {"model_path": str(self.folder / "artifacts/catboost.cbm"), "feature_names": FEATURE_NAMES,
                   "features": make_features(weather["rows"], as_of)}
        process = subprocess.run([runner, "-B", str(ROOT / "windagent/catboost_worker.py")], input=encoded(request),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=90, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if process.returncode:
            raise ValueError("Расчёт CatBoost завершился ошибкой: " + process.stderr.decode("utf-8", errors="replace")[-1800:])
        predicted = json.loads(process.stdout)["predictions"]
        if len(predicted) != len(weather["rows"]):
            raise ValueError("CatBoost вернул неверное число прогнозов.")
        result = []
        for row, value in zip(weather["rows"], predicted):
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("CatBoost вернул нечисловую мощность.")
            point = min(1, max(0, value))
            lead = (utc(row["valid_time"]) - utc(as_of)).total_seconds() / 3600
            radius = self.report["calibration_january"]["T" + row["turbine_id"] + ("/01-24" if lead <= 24 else "/25-48")]
            result.append({**row, "power_pred": point, "lower": max(0, point - radius), "upper": min(1, point + radius)})
        comparison = None
        try:
            saved = self.saved(as_of, weather)
        except ValueError as exc:
            if "Для этого выпуска" not in str(exc):
                raise
        else:
            comparison = max(abs(a["power_pred"] - b["power_pred"]) for a, b in zip(result, saved))
            if comparison > 1e-9:
                raise ValueError(f"Повторный CatBoost-расчёт отличается от предоставленного прогноза на {comparison:.6g}; проверьте признаки и версию модели.")
        return result, "recomputed", comparison
