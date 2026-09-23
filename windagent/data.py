"""Strict, dependency-free loading of hourly or subhourly SCADA CSV data.

The power target is normalized to the interval [0, 1]. ``power_scale`` is
an explicitly supplied divisor, never a quantity inferred from future labels.
Naive timestamps use the configured *fixed* UTC offset; callers should prefer
timestamps containing the offset used by the original measurement system.
"""

from __future__ import annotations

import csv
import math
import re
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path


UTC = timezone.utc
REQUIRED_COLUMNS = ("timestamp", "turbine_id", "wind_speed", "temperature", "power")


def _column_key(value: str) -> str:
    return re.sub(r"[^\w]", "", value.strip().casefold().replace("ё", "е"))


_ALIASES = {
    "timestamp": (
        "timestamp", "time", "datetime", "date_time", "статистическое время",
        "время", "дата и время", "дата время", "статистическое_время",
    ),
    "turbine_id": (
        "turbine_id", "turbine", "turbine id", "турбина", "номер турбины",
        "id турбины", "идентификатор турбины", "вэу", "номер вэу",
    ),
    "wind_speed": (
        "wind_speed", "wind speed", "wind_speed_ms", "windspeed",
        "средняя скорость ветра, м/с", "средняя скорость ветра",
        "скорость ветра", "скорость ветра м/с",
    ),
    "temperature": (
        "temperature", "temperature_c", "ambient_temperature", "temp",
        "средняя температура окружающей среды, °C",
        "средняя температура окружающей среды, °С",
        "средняя температура окружающей среды", "температура", "температура °C",
    ),
    "power": (
        "power", "normalized_power", "active_power", "power_normalized",
        "нормализированная активная мощность на стороне линии",
        "нормализированная активная мощность", "нормализованная мощность",
        "активная мощность", "мощность",
    ),
}


def iso_utc(value: datetime) -> str:
    """Serialize an aware datetime without discarding its time zone."""
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Дата должна содержать часовой пояс.")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _timestamp(value: str, fixed_tz: timezone | None) -> tuple[datetime, bool]:
    value = value.strip()
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        parsed = None
        for fmt in ("%d.%m.%Y %H:%M:%S", "%d.%m.%Y %H:%M"):
            try:
                parsed = datetime.strptime(value, fmt)
                break
            except ValueError:
                continue
        if parsed is None:
            raise ValueError(f"не удалось разобрать время {value!r}; используйте ISO 8601")
    naive = parsed.tzinfo is None or parsed.utcoffset() is None
    if naive:
        if fixed_tz is None:
            raise ValueError("время без часового пояса; задайте timezone_offset_hours")
        parsed = parsed.replace(tzinfo=fixed_tz)
    return parsed.astimezone(UTC), naive


def _number(value: str | None, name: str) -> float:
    if value is None or not value.strip():
        raise ValueError(f"пропущено поле {name}")
    try:
        result = float(value.strip().replace(",", "."))
    except ValueError:
        raise ValueError(f"поле {name}: ожидалось число, получено {value!r}") from None
    if not math.isfinite(result):
        raise ValueError(f"поле {name}: NaN и бесконечность запрещены")
    return result


def _resolve_columns(headers: list[str], mapping: dict | None) -> dict[str, str]:
    if len(set(headers)) != len(headers):
        raise ValueError("CSV содержит повторяющиеся названия столбцов.")
    mapping = mapping or {}
    unknown = set(mapping) - set(REQUIRED_COLUMNS)
    if unknown:
        raise ValueError(f"Неизвестные поля в column_mapping: {', '.join(sorted(unknown))}")
    resolved = {}
    for canonical in REQUIRED_COLUMNS:
        if canonical in mapping:
            source = mapping[canonical]
            if source not in headers:
                raise ValueError(f"Столбец {source!r} для поля {canonical} отсутствует в CSV.")
            matches = [source]
        else:
            aliases = {_column_key(alias) for alias in _ALIASES[canonical]}
            matches = [header for header in headers if _column_key(header) in aliases]
        if not matches:
            raise ValueError(
                f"В CSV отсутствует столбец {canonical}. "
                "Укажите canonical→название CSV в column_mapping."
            )
        if len(matches) > 1:
            raise ValueError(f"Неоднозначные столбцы для {canonical}: {matches}; задайте column_mapping.")
        resolved[canonical] = matches[0]
    if len(set(resolved.values())) != len(resolved):
        raise ValueError("Каждому полю должен соответствовать отдельный столбец CSV.")
    return resolved


def load_history(
    path: str | Path,
    timezone_offset_hours: float | None = 5,
    power_scale: float = 1.0,
    column_mapping: dict[str, str] | None = None,
) -> tuple[list[dict], dict]:
    """Load CSV, reject bad records, and explicitly average subhourly samples.

    Accepted column mappings map canonical field names to exact CSV headers.
    Identical duplicates are removed; conflicting duplicates fail loudly.
    Returned ``timestamp`` is the UTC-hour bucket. ``available_at`` is the
    latest contributing sample timestamp, preventing a partial future hour
    from entering training before all its samples were observed. Existing
    exact hourly labels are assumed available at their label time. Measurement
    publishing delays cannot be inferred from this CSV.
    """
    try:
        power_scale = float(power_scale)
    except (ValueError, TypeError):
        raise ValueError("power_scale должен быть положительным конечным числом.") from None
    if not math.isfinite(power_scale) or power_scale <= 0:
        raise ValueError("power_scale должен быть положительным конечным числом.")
    fixed_tz = None
    if timezone_offset_hours is not None:
        try:
            offset = float(timezone_offset_hours)
        except (ValueError, TypeError):
            raise ValueError("timezone_offset_hours должен быть числом от -14 до 14.") from None
        if not math.isfinite(offset) or not -14 <= offset <= 14:
            raise ValueError("timezone_offset_hours должен быть числом от -14 до 14.")
        fixed_tz = timezone(timedelta(hours=offset))

    unique: dict[tuple[str, datetime], dict] = {}
    raw_count = naive_count = duplicate_count = subhourly_count = 0
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        sample = handle.read(16384)
        if not sample.strip():
            raise ValueError("CSV пуст.")
        handle.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        except csv.Error:
            # A malformed row must still produce a field-level error below.
            first_line = sample.splitlines()[0]
            delimiter = max((",", ";", "\t"), key=first_line.count)
            dialect = csv.excel
            reader = csv.DictReader(handle, dialect=dialect, delimiter=delimiter)
        else:
            reader = csv.DictReader(handle, dialect=dialect)
        if reader.fieldnames is None:
            raise ValueError("CSV не содержит заголовка.")
        columns = _resolve_columns(reader.fieldnames, column_mapping)
        delimiter = reader.reader.dialect.delimiter
        try:
            for record in reader:
                line = reader.line_num
                raw_count += 1
                try:
                    if None in record or any(record.get(source) is None for source in columns.values()):
                        raise ValueError("количество значений не соответствует заголовку CSV")
                    raw_time = record[columns["timestamp"]]
                    if not raw_time or not raw_time.strip():
                        raise ValueError("пропущено поле timestamp")
                    timestamp, naive = _timestamp(raw_time, fixed_tz)
                    naive_count += int(naive)
                    turbine_id = record[columns["turbine_id"]].strip()
                    if not turbine_id:
                        raise ValueError("пропущено поле turbine_id")
                    wind = _number(record[columns["wind_speed"]], "wind_speed")
                    temperature = _number(record[columns["temperature"]], "temperature")
                    power = _number(record[columns["power"]], "power") / power_scale
                    if not 0 <= wind <= 100:
                        raise ValueError("wind_speed вне диапазона 0–100 м/с")
                    if not -100 <= temperature <= 70:
                        raise ValueError("temperature вне диапазона −100…70 °C")
                    if not 0 <= power <= 1:
                        raise ValueError(
                            "нормализованная power вне диапазона [0, 1]; "
                            "проверьте power_scale (явный делитель исходной мощности)"
                        )
                    row = {"timestamp": timestamp, "turbine_id": turbine_id,
                           "wind_speed": wind, "temperature": temperature, "power": power}
                    key = (turbine_id, timestamp)
                    if key in unique:
                        previous = unique[key]
                        if any(previous[name] != row[name] for name in ("wind_speed", "temperature", "power")):
                            raise ValueError(f"противоречивый дубликат турбины {turbine_id}, {iso_utc(timestamp)}")
                        duplicate_count += 1
                        continue
                    unique[key] = row
                    subhourly_count += int(bool(timestamp.minute or timestamp.second or timestamp.microsecond))
                except (ValueError, OverflowError) as exc:
                    raise ValueError(f"CSV, строка {line}: {exc}") from None
        except csv.Error as exc:
            raise ValueError(f"CSV, строка {reader.line_num}: {exc}") from None
    if not unique:
        raise ValueError("CSV не содержит строк данных.")

    buckets: dict[tuple[str, datetime], list[dict]] = defaultdict(list)
    for row in unique.values():
        hour = row["timestamp"].replace(minute=0, second=0, microsecond=0)
        buckets[(row["turbine_id"], hour)].append(row)
    rows = []
    for (turbine_id, hour), members in buckets.items():
        row = {
            "timestamp": hour,
            "available_at": max(member["timestamp"] for member in members),
            "turbine_id": turbine_id,
            "sample_count": len(members),
        }
        for field in ("wind_speed", "temperature", "power"):
            row[field] = math.fsum(member[field] for member in members) / len(members)
        rows.append(row)
    rows.sort(key=lambda row: (row["timestamp"], row["turbine_id"]))

    by_turbine: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_turbine[row["turbine_id"]].append(row)
    turbine_report = {}
    warnings = []
    for turbine_id, group in sorted(by_turbine.items()):
        missing_hours = sum(
            max(0, int((right["timestamp"] - left["timestamp"]).total_seconds() // 3600) - 1)
            for left, right in zip(group, group[1:])
        )
        constant_power = len({row["power"] for row in group}) == 1
        turbine_report[turbine_id] = {
            "rows": len(group), "first_timestamp": iso_utc(group[0]["timestamp"]),
            "last_timestamp": iso_utc(group[-1]["timestamp"]),
            "missing_hours": missing_hours, "constant_power": constant_power,
        }
        if missing_hours:
            warnings.append(f"Турбина {turbine_id}: отсутствует {missing_hours} часов; пропуски не заполнены.")
        if constant_power:
            warnings.append(f"Турбина {turbine_id}: мощность постоянна во всей истории; проверьте работу и экспорт.")
    if naive_count:
        warnings.append(
            f"{naive_count} строк без часового пояса: применён фиксированный UTC{float(timezone_offset_hours):+g}; "
            "исторические изменения часового пояса не восстанавливаются автоматически."
        )
    if subhourly_count or len(rows) < len(unique):
        warnings.append(
            "Подчасовые измерения усреднены арифметически внутри часа UTC; "
            "при нерегулярной частоте это среднее измерений, а не взвешенное по времени."
        )
    warnings.append(
        "Часовая метка считается временем доступности измерения; задержка публикации не известна. "
        "Агрегированный час доступен только после последнего входящего измерения."
    )
    return rows, {
        "source": str(Path(path)), "input_rows": raw_count, "hourly_rows": len(rows),
        "unique_input_rows": len(unique), "identical_duplicates_removed": duplicate_count,
        "subhourly_rows": subhourly_count, "aggregation": "arithmetic_mean_in_UTC_hour",
        "aggregation_rows_collapsed": len(unique) - len(rows),
        "timestamp_assumption": "hourly_label_available_at_label; aggregate_available_at_latest_sample",
        "naive_timestamp_rows": naive_count, "timezone_offset_hours": timezone_offset_hours,
        "power_scale": power_scale, "delimiter": delimiter, "columns": columns,
        "turbines": turbine_report, "warnings": warnings,
    }
