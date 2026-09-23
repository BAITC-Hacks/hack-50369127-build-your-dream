"""Small empirical weather-to-power models with chronological validation.

No random splits, future SCADA features, or observations after the supplied
cutoff enter training. Validation uses *observed* SCADA wind and temperature;
it measures the conversion model, not end-to-end NWP forecast accuracy.
"""

from __future__ import annotations

import bisect
import math
from collections import defaultdict
from datetime import datetime, timezone

from .data import iso_utc


MIN_TRAINING_ROWS = 48
VALIDATION_DESCRIPTION = (
    "Последние 20% доступной истории каждой турбины, хронологически. "
    "На входе фактические ветер и температура SCADA: это качество модели «погода→мощность», "
    "не точность прогноза с архивным погодным прогнозом."
)
INTERVAL_DESCRIPTION = (
    "Эмпирическая полоса ±90-й процентиль абсолютной ошибки на отложенной истории "
    "с фактической погодой; не калиброванный доверительный интервал. "
    "Не учитывает дополнительную ошибку прогноза погоды."
)


def _aware(value: datetime | str, field: str) -> datetime:
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            raise ValueError(f"{field}: некорректная дата ISO 8601.") from None
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field}: необходима дата с часовым поясом.")
    return value.astimezone(timezone.utc)


def _finite_range(value: object, field: str, low: float, high: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field}: необходимо конечное число.") from None
    if not math.isfinite(number) or not low <= number <= high:
        raise ValueError(f"{field}: необходимо конечное число в диапазоне [{low}, {high}].")
    return number


def effective_wind(wind_speed: float, temperature: float) -> float:
    """Approximate constant-pressure density adjustment to 15 °C.

    rho/rho_ref ≈ 288.15 / T[K], and wind power is proportional to rho*v³.
    Actual pressure, hub height, wake losses, and operating state are unknown.
    The empirical model learns average operating behavior from the data.
    """
    wind = _finite_range(wind_speed, "wind_speed", 0, 100)
    temperature = _finite_range(temperature, "temperature", -100, 70)
    return wind * (288.15 / (temperature + 273.15)) ** (1 / 3)


def _mean(values: list[float]) -> float:
    return math.fsum(values) / len(values)


def _fit(rows: list[dict], kind: str, bin_width: float | None = None) -> dict:
    mean_power = _mean([row["power"] for row in rows])
    winds = [effective_wind(row["wind_speed"], row["temperature"]) for row in rows]
    result = {"kind": kind, "mean_power": mean_power,
              "wind_min": min(winds), "wind_max": max(winds)}
    if kind == "constant":
        return result
    grouped: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for wind, row in zip(winds, rows):
        grouped[math.floor(wind / bin_width)].append((wind, row["power"]))
    result["bin_width"] = bin_width
    result["points"] = [
        {"wind_speed": _mean([item[0] for item in group]),
         "power": _mean([item[1] for item in group]), "count": len(group)}
        for _, group in sorted(grouped.items())
    ]
    return result


def _predict_one(model: dict, wind_speed: float, temperature: float) -> float:
    wind = effective_wind(wind_speed, temperature)
    if model["kind"] == "constant":
        return model["mean_power"]
    points = model["points"]
    index = bisect.bisect_left([point["wind_speed"] for point in points], wind)
    if index == 0:
        power = points[0]["power"]
    elif index == len(points):
        power = points[-1]["power"]
    else:
        left, right = points[index - 1], points[index]
        weight = (wind - left["wind_speed"]) / (right["wind_speed"] - left["wind_speed"])
        power = left["power"] * (1 - weight) + right["power"] * weight
    return min(1.0, max(0.0, power))


def _metrics(actual: list[float], predicted: list[float]) -> dict:
    residuals = [truth - forecast for truth, forecast in zip(actual, predicted)]
    return {"mae": _mean([abs(value) for value in residuals]),
            "rmse": math.sqrt(_mean([value * value for value in residuals])),
            "bias": _mean([-value for value in residuals]), "n": len(actual)}


def _quantile(values: list[float], level: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * level
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def train_models(rows: list[dict], cutoff: datetime) -> dict:
    """Select curve resolution using a time holdout and refit up to cutoff.

    Curves use bins of 0.5, 1, or 2 m/s and linear interpolation. A constant
    mean is selected if it has the lowest holdout MAE (also on ties). The
    persistence baseline holds the last *training-prefix* power unchanged
    over the full holdout; it never consumes holdout labels during prediction.
    """
    cutoff = _aware(cutoff, "cutoff")
    grouped: dict[str, list[dict]] = defaultdict(list)
    future_count = 0
    seen = set()
    for supplied in rows:
        timestamp = _aware(supplied["timestamp"], "timestamp")
        available = _aware(supplied.get("available_at", timestamp), "available_at")
        if timestamp > cutoff or available > cutoff:
            future_count += 1
            continue
        if available < timestamp:
            raise ValueError("available_at не может предшествовать timestamp измерения.")
        turbine_id = str(supplied.get("turbine_id", "")).strip()
        if not turbine_id:
            raise ValueError("Отсутствует turbine_id.")
        key = (turbine_id, timestamp)
        if key in seen:
            raise ValueError(f"Повторный час {iso_utc(timestamp)} для турбины {turbine_id}; сначала загрузите CSV.")
        seen.add(key)
        row = {"timestamp": timestamp, "available_at": available, "turbine_id": turbine_id}
        row["wind_speed"] = _finite_range(supplied.get("wind_speed"), "wind_speed", 0, 100)
        row["temperature"] = _finite_range(supplied.get("temperature"), "temperature", -100, 70)
        row["power"] = _finite_range(supplied.get("power"), "power", 0, 1)
        grouped[turbine_id].append(row)
    if not grouped:
        raise ValueError("До момента прогноза нет доступной истории для обучения.")

    fitted, diagnostics, warnings = {}, {}, []
    training_ends, availability_ends = [], []
    for turbine_id, history in sorted(grouped.items()):
        history.sort(key=lambda row: row["timestamp"])
        if len(history) < MIN_TRAINING_ROWS:
            raise ValueError(
                f"Турбина {turbine_id}: до cutoff доступно {len(history)} часов; "
                f"для обучения и хронологической проверки требуется минимум {MIN_TRAINING_ROWS}."
            )
        holdout_count = max(10, math.ceil(len(history) * 0.2))
        prefix, holdout = history[:-holdout_count], history[-holdout_count:]
        # Even custom rows with delayed availability must respect the split.
        split = holdout[0]["timestamp"]
        prefix = [row for row in prefix if row["available_at"] < split]
        if len(prefix) < 24:
            raise ValueError(f"Турбина {turbine_id}: недостаточно доступных до holdout часов после учёта задержек.")
        actual = [row["power"] for row in holdout]
        candidates = [("constant", None)] + [("empirical_curve", width) for width in (0.5, 1.0, 2.0)]
        evaluations = []
        for kind, width in candidates:
            candidate = _fit(prefix, kind, width)
            predictions = [_predict_one(candidate, row["wind_speed"], row["temperature"]) for row in holdout]
            evaluations.append({"kind": kind, "bin_width": width, "metrics": _metrics(actual, predictions),
                                "predictions": predictions})
        winner = min(evaluations, key=lambda entry: entry["metrics"]["mae"])
        model = _fit(history, winner["kind"], winner["bin_width"])
        residuals = [abs(truth - forecast) for truth, forecast in zip(actual, winner["predictions"])]
        model.update({
            "error_band": _quantile(residuals, 0.9),
            "persistence_power": history[-1]["power"],
            "persistence_timestamp": iso_utc(history[-1]["timestamp"]),
            "training_rows": len(history), "training_end": iso_utc(history[-1]["timestamp"]),
            "training_available_end": iso_utc(max(row["available_at"] for row in history)),
        })
        fitted[turbine_id] = model
        diagnostics[turbine_id] = {
            "selected_kind": winner["kind"], "selected_bin_width": winner["bin_width"],
            "selected": winner["metrics"], "constant_baseline": evaluations[0]["metrics"],
            "persistence_baseline": _metrics(actual, [prefix[-1]["power"]] * len(holdout)),
            "persistence_baseline_value": prefix[-1]["power"],
            "fit_rows": len(prefix), "holdout_rows": len(holdout),
            "fit_end": iso_utc(prefix[-1]["timestamp"]),
            "fit_available_end": iso_utc(max(row["available_at"] for row in prefix)),
            "holdout_start": iso_utc(holdout[0]["timestamp"]),
            "holdout_end": iso_utc(holdout[-1]["timestamp"]),
            "candidates": [{key: value for key, value in entry.items() if key != "predictions"}
                           for entry in evaluations],
            "validation_description": VALIDATION_DESCRIPTION,
        }
        if len({row["power"] for row in history}) == 1:
            warnings.append(f"Турбина {turbine_id}: постоянная мощность; качество на этой истории не доказывает работоспособность.")
        if len(history) < 168:
            warnings.append(f"Турбина {turbine_id}: история меньше недели; модель и оценка ошибки нестабильны.")
        if winner["kind"] == "constant":
            warnings.append(f"Турбина {turbine_id}: выбрана константа — кривые мощности не улучшили MAE на holdout.")
        latest_age_hours = (cutoff - history[-1]["timestamp"]).total_seconds() / 3600
        if latest_age_hours > 48:
            warnings.append(f"Турбина {turbine_id}: последняя история старше cutoff на {latest_age_hours:.0f} ч; persistence устарел.")
        training_ends.append(history[-1]["timestamp"])
        availability_ends.extend(row["available_at"] for row in history)
    warnings.extend([
        VALIDATION_DESCRIPTION, INTERVAL_DESCRIPTION,
        "Температурная поправка плотности приближённая при постоянном давлении; "
        "высота измерения ветра, давление, простои и ограничения мощности отдельно не моделируются.",
        "За диапазоном обучающего ветра используется ближайшая оценка; неизвестную отсечку турбины модель не угадывает.",
    ])
    return {
        "schema_version": 1, "per_turbine": fitted, "diagnostics": diagnostics,
        "training_cutoff": iso_utc(cutoff), "training_end": iso_utc(max(training_ends)),
        "training_available_end": iso_utc(max(availability_ends)),
        "training_rows": sum(len(history) for history in grouped.values()),
        "excluded_future_rows": future_count, "warnings": warnings,
        "target": "normalized_active_power_0_to_1", "validation": VALIDATION_DESCRIPTION,
        "interval": {"kind": "empirical_absolute_error_band", "quantile": 0.9,
                     "calibrated": False, "weather_basis": "observed_scada",
                     "description": INTERVAL_DESCRIPTION},
        "density_adjustment": "v_eff = wind_speed * (288.15 / (temperature_C + 273.15)) ** (1/3)",
    }


def predict(models: dict, weather_rows: list[dict]) -> list[dict]:
    """Predict normalized power; preserve weather lineage and row ordering."""
    forecasts = []
    cutoff = _aware(models["training_cutoff"], "training_cutoff")
    for row in weather_rows:
        turbine_id = str(row.get("turbine_id", "")).strip()
        if turbine_id not in models["per_turbine"]:
            raise ValueError(f"Нет обученной модели для турбины {turbine_id!r}.")
        valid_time = _aware(row["valid_time"], "valid_time")
        if valid_time <= cutoff:
            raise ValueError("valid_time прогноза должен быть позже training_cutoff.")
        model = models["per_turbine"][turbine_id]
        wind = _finite_range(row.get("wind_speed"), "wind_speed", 0, 100)
        temperature = _finite_range(row.get("temperature"), "temperature", -100, 70)
        point = _predict_one(model, wind, temperature)
        corrected = effective_wind(wind, temperature)
        forecasts.append({
            **row, "turbine_id": turbine_id, "power_pred": point,
            "lower": max(0.0, point - model["error_band"]),
            "upper": min(1.0, point + model["error_band"]),
            "persistence_pred": model["persistence_power"],
            "wind_outside_training_range": not model["wind_min"] <= corrected <= model["wind_max"],
            "interval_kind": "empirical_error_band_not_calibrated",
        })
    return forecasts
