"""Post-hoc scoring; evaluation labels are never passed to a forecasting model."""
import csv
import math

from .common import utc
from .data import load_history


def evaluate(forecast_path, actual_path, timezone_offset_hours=5, power_scale=1):
    actuals, _ = load_history(actual_path, timezone_offset_hours, power_scale)
    lookup = {(r["turbine_id"], utc(r["timestamp"])): r["power"] for r in actuals}
    groups = {}
    matched, missing = 0, 0
    seen = set()
    with open(forecast_path, encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        required = {"forecast_origin", "valid_time", "turbine_id", "power_pred", "persistence_pred"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("CSV прогноза должен содержать столбцы: " + ", ".join(sorted(required)))
        for row in reader:
            if any(row.get(key) is None or not row[key].strip() for key in required):
                raise ValueError(f"CSV прогноза, строка {reader.line_num}: пропущено обязательное значение.")
            origin, valid = utc(row["forecast_origin"]), utc(row["valid_time"])
            key = (origin, valid, row["turbine_id"])
            if key in seen:
                raise ValueError("В прогнозе повторяется пара дата выпуска / прогнозный час / турбина.")
            seen.add(key)
            lead = (valid - origin).total_seconds() / 3600
            if lead not in range(1, 49):
                raise ValueError("Прогноз содержит горизонт вне 1–48 часов.")
            prediction = float(row["power_pred"])
            baseline = float(row["persistence_pred"])
            if not all(math.isfinite(v) and 0 <= v <= 1 for v in (prediction, baseline)):
                raise ValueError("Оценка требует конечных нормализованных значений мощности 0–1.")
            truth = lookup.get((row["turbine_id"], valid))
            if truth is None:
                missing += 1
                continue
            group = row["turbine_id"] + (" / 01–24 ч" if lead <= 24 else " / 25–48 ч")
            groups.setdefault(group, []).append((truth - prediction, truth - baseline))
            matched += 1
    if not matched:
        raise ValueError("Нет совпадающих фактических и прогнозных часов.")
    metrics = {}
    for key, errors in groups.items():
        n = len(errors)
        metrics[key] = {"count": n, "mae": sum(abs(a) for a, _ in errors) / n,
            "rmse": math.sqrt(sum(a * a for a, _ in errors) / n),
            "persistence_mae": sum(abs(b) for _, b in errors) / n,
            "persistence_rmse": math.sqrt(sum(b * b for _, b in errors) / n)}
    return {"matched": matched, "missing_actuals": missing, "metrics": metrics, "unit": "normalized_power",
            "complete": missing == 0, "note": "Оценка выполняется отдельно и не изменяет обученные модели."}
