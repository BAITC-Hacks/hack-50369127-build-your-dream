"""Read original turbine files without changing them or guessing their clock.

Every retained target represents six complete consecutive ten-minute slots.
Hourly labels denote interval starts; availability is at the interval end.
The supplied offset and interval convention are assumptions, recorded as such.
"""

from __future__ import annotations

import csv
import hashlib
import io
import math
from collections import Counter, defaultdict
from datetime import timedelta, timezone
from pathlib import Path

from .data import _ALIASES, _column_key, _timestamp, iso_utc


FIELDS = ("wind_speed", "temperature", "power")
TEN_MINUTES = timedelta(minutes=10)
ONE_HOUR = timedelta(hours=1)


def _columns(headers):
    if not headers or len(set(headers)) != len(headers):
        raise ValueError("Исходный CSV не содержит уникального заголовка.")
    result = {}
    for canonical in ("timestamp", *FIELDS):
        aliases = {_column_key(alias) for alias in _ALIASES[canonical]}
        matches = [header for header in headers if _column_key(header) in aliases]
        if len(matches) != 1:
            raise ValueError(f"В исходном CSV требуется ровно один столбец {canonical}; найдено: {matches}.")
        result[canonical] = matches[0]
    return result


def _measurement(raw, field):
    target = field == "power"
    if raw is None or not raw.strip():
        return None, "missing_target" if target else "missing_predictor"
    try:
        value = float(raw.strip().replace(",", "."))
    except ValueError:
        return None, "nonnumeric_target" if target else "invalid_predictor"
    if not math.isfinite(value):
        return None, "nonfinite_target" if target else "nonfinite_predictor"
    low, high = {"wind_speed": (0, 100), "temperature": (-100, 70), "power": (0, 1)}[field]
    if not low <= value <= high:
        return None, "target_out_of_range" if target else "predictor_out_of_range"
    return value, None


def load_turbine_files(
    paths: dict[str, Path],
    timezone_offset_hours: float = 0,
    interval_convention: str = "start",
    min_samples: int = 6,
) -> tuple[list[dict], dict]:
    """Convert separate source CSVs into complete hourly observations.

    ``paths`` explicitly maps turbine identifiers to files; source ``ID`` is
    never interpreted as a turbine identifier. ``start`` means a source label
    starts its ten-minute interval; ``end`` shifts it backwards ten minutes.
    Bad measurements are reported and exclude their entire hour. No target is
    clipped, filled, or extrapolated. Exactly six regular slots are required;
    relaxing this policy through ``min_samples`` is deliberately unsupported.
    """
    if not isinstance(paths, dict) or not paths:
        raise ValueError("Укажите соответствие идентификаторов турбин исходным CSV.")
    identifiers = [str(key).strip() for key in paths]
    if any(not key for key in identifiers) or len(set(identifiers)) != len(identifiers):
        raise ValueError("Идентификаторы турбин должны быть непустыми и уникальными.")
    if interval_convention not in ("start", "end"):
        raise ValueError("interval_convention должен быть start или end.")
    if type(min_samples) is not int or min_samples != 6:
        raise ValueError("Для полного часа требуется min_samples=6; частичные часы не допускаются.")
    try:
        offset = float(timezone_offset_hours)
    except (ValueError, TypeError):
        raise ValueError("Явно задайте числовой timezone_offset_hours от -14 до 14.") from None
    if not math.isfinite(offset) or not -14 <= offset <= 14:
        raise ValueError("timezone_offset_hours должен быть конечным числом от -14 до 14.")
    fixed_tz = timezone(timedelta(hours=offset))
    rows, reports = [], {}
    warnings = [
        f"Время источника не подтверждено: для меток без зоны явно применено UTC{offset:+g}. "
        "Это настройка импорта, а не выведенный из CSV часовой пояс; историческое смещение не угадывается.",
        f"Принята неподтверждённая конвенция 10-минутных меток: {interval_convention}; "
        "часовая timestamp обозначает начало интервала, available_at — его конец. "
        "Дополнительная задержка публикации не известна.",
        "Используются только полные часы с шестью допустимыми измерениями; пропуски не заполнены.",
    ]
    for turbine_id, source in sorted(((str(key).strip(), Path(value)) for key, value in paths.items())):
        sha = hashlib.sha256()
        with source.open("rb") as binary:
            for chunk in iter(lambda: binary.read(1024 * 1024), b""):
                sha.update(chunk)
        samples = {}
        observed = defaultdict(set)
        bad_hours = set()
        rejection_counts = Counter()
        rejected_rows = []
        raw_count = valid_count = naive_count = duplicate_count = conflicting_count = 0
        source_times = []
        original_first = original_last = None
        with source.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            columns = _columns(reader.fieldnames)
            for record in reader:
                raw_count += 1
                source_id = record.get("ID")
                raw_stamp = record.get(columns["timestamp"])
                reasons = []
                if None in record or any(record.get(column) is None for column in columns.values()):
                    reasons.append("malformed_row")
                try:
                    if not raw_stamp or not raw_stamp.strip():
                        raise ValueError("empty timestamp")
                    stamp, naive = _timestamp(raw_stamp, fixed_tz)
                except (ValueError, OverflowError):
                    reasons.append("invalid_timestamp")
                    stamp = None
                if stamp is not None:
                    source_times.append(stamp)
                    naive_count += int(naive)
                    original_first = raw_stamp if original_first is None else original_first
                    original_last = raw_stamp
                    slot = stamp - TEN_MINUTES if interval_convention == "end" else stamp
                    hour = slot.replace(minute=0, second=0, microsecond=0)
                    observed[hour].add(slot)
                    if slot.minute % 10 or slot.second or slot.microsecond:
                        reasons.append("unaligned_10minute_slot")
                parsed = {}
                for field in FIELDS:
                    value, reason = _measurement(record.get(columns[field]), field)
                    if reason:
                        reasons.append(reason + ":" + field)
                    else:
                        parsed[field] = value
                if reasons:
                    if stamp is not None:
                        bad_hours.add(hour)
                    rejection_counts.update(reason.split(":", 1)[0] for reason in set(reasons))
                    rejected_rows.append({"line": reader.line_num, "source_id": source_id,
                                          "timestamp": raw_stamp, "reasons": reasons})
                    continue
                valid_count += 1
                if slot in samples:
                    if samples[slot] == parsed:
                        duplicate_count += 1
                    else:
                        conflicting_count += 1
                        bad_hours.add(hour)
                        rejection_counts["conflicting_duplicate"] += 1
                        rejected_rows.append({"line": reader.line_num, "source_id": source_id,
                                              "timestamp": raw_stamp, "reasons": ["conflicting_duplicate"]})
                    continue
                samples[slot] = parsed
        if raw_count == 0:
            raise ValueError(f"Исходный CSV турбины {turbine_id} не содержит данных.")
        if not source_times:
            raise ValueError(f"В CSV турбины {turbine_id} нет корректных временных меток.")
        complete_count = partial_count = invalid_hour_count = 0
        rejected_hours = []
        sample_histogram = Counter()
        for hour, slots in sorted(observed.items()):
            expected = {hour + index * TEN_MINUTES for index in range(6)}
            sample_histogram[str(len(slots))] += 1
            reasons = []
            if slots != expected:
                partial_count += 1
                reasons.append("incomplete_hour")
            if hour in bad_hours:
                invalid_hour_count += 1
                reasons.append("invalid_measurement_or_conflicting_duplicate")
            if reasons:
                rejected_hours.append({"timestamp": iso_utc(hour), "observed_slots": len(slots), "reasons": reasons})
                continue
            complete_count += 1
            members = [samples[slot] for slot in sorted(slots)]
            rows.append({
                "timestamp": hour, "available_at": hour + ONE_HOUR,
                "sample_count": 6, "turbine_id": turbine_id,
                **{field: math.fsum(member[field] for member in members) / 6 for field in FIELDS},
            })
        hours = sorted(observed)
        normalized_times = sorted(slot for slots in observed.values() for slot in slots)
        missing_slots = sum(max(0, int((right - left).total_seconds() // 600) - 1)
                            for left, right in zip(normalized_times, normalized_times[1:]))
        missing_hours = (int((hours[-1] - hours[0]).total_seconds() // 3600) + 1 - len(hours)) if hours else 0
        report = {
            "source": str(source), "source_sha256": sha.hexdigest(), "sha256": sha.hexdigest(),
            "source_bytes": source.stat().st_size, "columns": columns,
            "raw_rows": raw_count, "valid_raw_rows": valid_count,
            "invalid_raw_rows": raw_count - valid_count,
            "naive_timestamp_rows": naive_count, "unique_timestamps": len(normalized_times),
            "identical_duplicates_removed": duplicate_count, "conflicting_duplicates": conflicting_count,
            "raw_first_timestamp": original_first, "raw_last_timestamp": original_last,
            "first_sample_timestamp": iso_utc(min(source_times)),
            "last_sample_timestamp": iso_utc(max(source_times)),
            "hourly_rows": complete_count, "partial_hours_excluded": partial_count,
            "invalid_hours_excluded": invalid_hour_count, "missing_hours": missing_hours,
            "missing_10minute_slots": missing_slots, "hourly_sample_histogram": dict(sample_histogram),
            "rejection_counts": dict(rejection_counts), "rejected_rows": rejected_rows,
            "rejected_hours": rejected_hours,
        }
        reports[turbine_id] = report
        if partial_count or invalid_hour_count or missing_slots:
            warnings.append(
                f"Турбина {turbine_id}: исключено частичных часов {partial_count}, "
                f"часов с ошибками {invalid_hour_count}; отсутствует 10-минутных слотов {missing_slots}."
            )
        if not complete_count:
            warnings.append(f"Турбина {turbine_id}: нет пригодных полных часов; обучение этой турбины невозможно.")
    rows.sort(key=lambda row: (row["timestamp"], row["turbine_id"]))
    return rows, {
        "schema_version": 1, "source_kind": "separate_turbine_10minute_csv",
        "timezone_offset_hours": offset, "timezone_confirmed": False,
        "interval_convention": interval_convention, "interval_convention_confirmed": False,
        "timestamp_convention": "hour_start", "availability_convention": "hour_end",
        "min_samples": 6, "expected_sample_minutes": [0, 10, 20, 30, 40, 50],
        "aggregation": "arithmetic_mean_of_six_complete_10minute_intervals",
        "hourly_rows": len(rows), "raw_rows": sum(report["raw_rows"] for report in reports.values()),
        "turbines": reports, "warnings": warnings,
    }


def history_to_csv(rows: list[dict]) -> str:
    """Serialize canonical history including availability and sample counts."""
    output = io.StringIO(newline="")
    fields = ("timestamp", "turbine_id", "wind_speed", "temperature", "power", "available_at", "sample_count")
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        count = row.get("sample_count", 1)
        if type(count) is not int or count < 1:
            raise ValueError("sample_count должен быть положительным целым числом.")
        timestamp = row["timestamp"]
        available_at = row.get("available_at", timestamp)
        if available_at < timestamp:
            raise ValueError("available_at не может предшествовать timestamp.")
        serialized = {key: row[key] for key in ("turbine_id", *FIELDS)}
        for field in FIELDS:
            _, reason = _measurement(str(serialized[field]), field)
            if reason:
                raise ValueError(f"{field}: {reason}")
        serialized.update(timestamp=iso_utc(timestamp), available_at=iso_utc(available_at), sample_count=count)
        writer.writerow(serialized)
    return output.getvalue()
