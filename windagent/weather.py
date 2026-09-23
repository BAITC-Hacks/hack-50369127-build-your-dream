"""Exact-run weather acquisition with an explicit, unverified availability policy.

This is a research adapter, not certification of historical publication times.
See docs/WEATHER.md. It never falls back to actual weather or stitched forecasts.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

UTC = timezone.utc
ENDPOINT = "https://single-runs-api.open-meteo.com/v1/forecast"
MODEL = "ecmwf_ifs"
ARCHIVE_START = datetime(2024, 3, 14, tzinfo=UTC)
MAX_RESPONSE_BYTES = 5_000_000
AVAILABILITY_BASIS = "assumed_conservative_lag_not_historical_publication_evidence"


class WeatherError(RuntimeError):
    """Missing, invalid, unavailable, or unauditable weather input."""


def _iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _aware_hour(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise WeatherError("as_of must be a timezone-aware datetime")
    value = value.astimezone(UTC)
    if value.minute or value.second or value.microsecond:
        raise WeatherError("as_of must be aligned to a UTC hour")
    return value


def select_run(as_of: datetime, availability_lag_hours: float = 12) -> datetime:
    """Latest 00/06/12/18 UTC run permitted by the assumed publication lag."""
    issue = _aware_hour(as_of)
    if (isinstance(availability_lag_hours, bool)
            or not isinstance(availability_lag_hours, (int, float))
            or not math.isfinite(availability_lag_hours)
            or not 0 <= availability_lag_hours <= 168):
        raise WeatherError("availability_lag_hours must be a finite number from 0 to 168")
    cutoff = issue - timedelta(hours=availability_lag_hours)
    run = cutoff.replace(hour=(cutoff.hour // 6) * 6, minute=0, second=0, microsecond=0)
    if run < ARCHIVE_START:
        raise WeatherError("Exact ECMWF run archive starts on 2024-03-14; no fallback is allowed")
    return run


def _target(value: dict) -> tuple[str, float, float]:
    if not isinstance(value, dict) or not isinstance(value.get("id"), str) or not value["id"].strip():
        raise WeatherError("Each turbine must have a nonempty string id")
    coordinates = []
    for name, lower, upper in (("latitude", -90, 90), ("longitude", -180, 180)):
        number = value.get(name)
        if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number):
            raise WeatherError(f"Turbine {value['id']}: {name} must be a finite number")
        if not lower <= number <= upper:
            raise WeatherError(f"Turbine {value['id']}: {name} is outside its valid range")
        coordinates.append(float(number))
    return value["id"], coordinates[0], coordinates[1]


def _atomic_write(path: Path, content: bytes) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(content)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _download(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "WindAgent-Hackathon/1.0", "Accept": "application/json"})
    for attempt in range(3):
        try:
            with urlopen(request, timeout=20) as response:
                raw = response.read(MAX_RESPONSE_BYTES + 1)
            if len(raw) > MAX_RESPONSE_BYTES:
                raise WeatherError("Weather response exceeds the allowed size")
            return raw
        except HTTPError as exc:
            if exc.code not in (408, 429, 500, 502, 503, 504) or attempt == 2:
                detail = ""
                if exc.fp is not None:
                    try:
                        reason = json.loads(exc.read(4096)).get("reason")
                        if isinstance(reason, str):
                            detail = ": " + reason[:300]
                    except (ValueError, OSError, AttributeError):
                        pass
                exc.close()
                raise WeatherError(f"Archived weather API returned HTTP {exc.code}{detail}; no substitute weather was used") from exc
            exc.close()
        except (URLError, TimeoutError, OSError) as exc:
            if attempt == 2:
                raise WeatherError(f"Cannot retrieve archived weather: {exc}") from exc
        time.sleep(0.5 * (2 ** attempt))
    raise WeatherError("Archived weather request failed")


def _preserve_snapshot(raw: bytes, cache_dir: Path) -> None:
    """Keep earlier source bytes available after a refresh changes the response."""
    directory = cache_dir / "raw"
    directory.mkdir(parents=True, exist_ok=True)
    snapshot = directory / (hashlib.sha256(raw).hexdigest() + ".json")
    if snapshot.exists():
        if snapshot.read_bytes() != raw:
            raise WeatherError("Immutable weather snapshot integrity check failed")
    else:
        _atomic_write(snapshot, raw)


def _response(url: str, cache_dir: Path, refresh: bool) -> tuple[bytes, str, bool]:
    key = hashlib.sha256(url.encode("utf-8")).hexdigest()
    raw_path = cache_dir / (key + ".json")
    meta_path = cache_dir / (key + ".meta.json")
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        if not refresh and (raw_path.exists() or meta_path.exists()):
            if not raw_path.exists() or not meta_path.exists():
                raise WeatherError("Incomplete weather cache; use refresh to download the exact run again")
            raw = raw_path.read_bytes()
            if len(raw) > MAX_RESPONSE_BYTES:
                raise WeatherError("Cached weather response exceeds the allowed size")
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if (not isinstance(meta, dict) or meta.get("url") != url
                    or meta.get("sha256") != hashlib.sha256(raw).hexdigest()):
                raise WeatherError("Weather cache integrity check failed; use refresh to download again")
            retrieved = datetime.fromisoformat(meta["retrieved_at"].replace("Z", "+00:00"))
            if retrieved.tzinfo is None:
                raise WeatherError("Weather cache has an invalid retrieval timestamp")
            _preserve_snapshot(raw, cache_dir)
            return raw, _iso(retrieved), True
        raw = _download(url)
        # Reject API errors / malformed JSON before making them persistent.
        _decode(raw)
        retrieved_at = _iso(datetime.now(UTC))
        meta = {"url": url, "sha256": hashlib.sha256(raw).hexdigest(), "retrieved_at": retrieved_at}
        _preserve_snapshot(raw, cache_dir)
        _atomic_write(raw_path, raw)
        _atomic_write(meta_path, json.dumps(meta, indent=2, ensure_ascii=False).encode("utf-8"))
        return raw, retrieved_at, False
    except WeatherError:
        raise
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        raise WeatherError(f"Cannot use weather cache: {exc}") from exc


def _decode(raw: bytes) -> dict:
    try:
        value = json.loads(raw)
    except (ValueError, UnicodeError) as exc:
        raise WeatherError("Archived weather API did not return valid JSON") from exc
    if not isinstance(value, dict) or value.get("error"):
        raise WeatherError("Archived weather API returned an error or unexpected response")
    return value


def _rows(payload: dict, turbine_id: str, valid_times: list[datetime]) -> list[dict]:
    if payload.get("utc_offset_seconds") != 0:
        raise WeatherError("Weather response must explicitly use UTC (utc_offset_seconds=0)")
    units = payload.get("hourly_units", {})
    if not isinstance(units, dict) or units.get("wind_speed_100m") != "m/s" or units.get("temperature_2m") != "°C":
        raise WeatherError("Weather units must be wind m/s at 100m and temperature °C at 2m")
    if units.get("time") != "iso8601":
        raise WeatherError("Weather time format must be ISO8601")
    hourly = payload.get("hourly", {})
    if not isinstance(hourly, dict):
        raise WeatherError("Weather response lacks hourly data")
    arrays = [hourly.get(key) for key in ("time", "wind_speed_100m", "temperature_2m")]
    if not all(isinstance(array, list) for array in arrays) or not arrays[0] or len({len(a) for a in arrays}) != 1:
        raise WeatherError("Weather hourly arrays are empty, missing, or unequal in length")
    by_time = {}
    for stamp, wind, temperature in zip(*arrays):
        try:
            if not isinstance(stamp, str):
                raise ValueError("timestamp is not a string")
            parsed = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=UTC)  # response offset explicitly checked above
            parsed = _aware_hour(parsed)
        except (ValueError, TypeError) as exc:
            raise WeatherError("Weather response contains an invalid hourly timestamp") from exc
        if parsed in by_time:
            raise WeatherError("Weather response contains duplicate hourly timestamps")
        by_time[parsed] = (wind, temperature)
    result = []
    for valid in valid_times:
        if valid not in by_time:
            raise WeatherError(f"Archived forecast is missing required hour {_iso(valid)} for {turbine_id}")
        wind, temperature = by_time[valid]
        for key, number, lower, upper in (("wind_speed", wind, 0, 100), ("temperature", temperature, -100, 70)):
            if (isinstance(number, bool) or not isinstance(number, (int, float))
                    or not math.isfinite(number) or not lower <= number <= upper):
                raise WeatherError(f"Invalid or missing {key} for {turbine_id} at {_iso(valid)}")
        result.append({"turbine_id": turbine_id, "valid_time": _iso(valid), "wind_speed": float(wind), "temperature": float(temperature)})
    return result


def fetch_weather(as_of: datetime, targets: list[dict], hours: int, cache_dir: Path,
                  refresh: bool = False, availability_lag_hours: float = 12) -> dict:
    """Retrieve one exact model run per turbine, returning as_of+1h ... +hours.

    available_at is an assumption, clearly labelled in every provenance record.
    competition_ready is always False until operational-run/publication evidence
    has been independently supplied; a URL with run= alone is not that evidence.
    """
    issue = _aware_hour(as_of)
    if isinstance(hours, bool) or not isinstance(hours, int) or not 1 <= hours <= 48:
        raise WeatherError("hours must be an integer from 1 to 48")
    if not isinstance(targets, list) or not targets:
        raise WeatherError("At least one turbine is required")
    checked = [_target(target) for target in targets]
    if len({target[0] for target in checked}) != len(checked):
        raise WeatherError("Turbine ids must be unique")
    run = select_run(issue, availability_lag_hours)
    assumed_available = run + timedelta(hours=availability_lag_hours)
    valid_times = [issue + timedelta(hours=index) for index in range(1, hours + 1)]
    if valid_times[-1] >= run + timedelta(days=7):
        raise WeatherError("Requested hours exceed this adapter's seven-day exact-run window; reduce the assumed publication lag")
    warnings = [
        f"Доступность погоды в прошлом НЕ подтверждена: assumed available_at = run + {availability_lag_hours:g} ч. Это допущение, а не архивный журнал публикации; competition_ready=false.",
        "В документации ранний архив ECMWF назван hindcasts. Происхождение конкретного запуска (операционный прогноз или ретропрогноз) требует подтверждения источника.",
        "Прогноз ветра относится к высоте 100 м, температуры — к 2 м. Высота датчика/ступицы ВЭС не подтверждена; требуется калибровка на архивных прогнозах.",
    ]
    if availability_lag_hours < 12:
        warnings.append("Указанная задержка меньше принятого консервативного значения 12 ч и увеличивает риск утечки будущей информации.")
    rows, provenance = [], []
    for turbine_id, latitude, longitude in checked:
        query = {
            "latitude": latitude, "longitude": longitude,
            "models": MODEL, "run": run.strftime("%Y-%m-%dT%H:%M"),
            "hourly": "wind_speed_100m,temperature_2m", "wind_speed_unit": "ms",
            "temperature_unit": "celsius", "timezone": "UTC", "timeformat": "iso8601",
            # Single Runs rejects start_date/end_date. Fetch the run's window
            # and select only the requested issue-relative hours locally.
            "forecast_days": 7,
        }
        url = ENDPOINT + "?" + urlencode(query)
        raw, retrieved_at, cache_hit = _response(url, Path(cache_dir), refresh)
        payload = _decode(raw)
        rows.extend(_rows(payload, turbine_id, valid_times))
        provenance.append({
            "turbine_id": turbine_id, "source": "Open-Meteo Single Runs API", "model": MODEL,
            "run_time": _iso(run), "available_at": _iso(assumed_available),
            "availability_basis": AVAILABILITY_BASIS, "availability_verified": False,
            "availability_lag_hours": availability_lag_hours,
            "competition_ready": False, "operational_run_verified": False,
            "url": url, "sha256": hashlib.sha256(raw).hexdigest(), "retrieved_at": retrieved_at,
            "raw_path": str(Path(cache_dir) / "raw" / (hashlib.sha256(raw).hexdigest() + ".json")),
            "cache_hit": cache_hit, "wind_height_m": 100, "temperature_height_m": 2,
            "requested_latitude": latitude, "requested_longitude": longitude,
            "grid_latitude": payload.get("latitude"), "grid_longitude": payload.get("longitude"),
        })
    return {"rows": rows, "provenance": provenance, "warnings": warnings}
