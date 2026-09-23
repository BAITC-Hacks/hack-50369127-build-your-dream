from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import zipfile
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from time import sleep
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

from .settings import TurbineSettings, WeatherSettings

SINGLE_RUN_ENDPOINT = "https://single-runs-api.open-meteo.com/v1/forecast"
DOCUMENTATION_URL = "https://open-meteo.com/en/docs/single-runs-api"
ARCHIVE_START = datetime(2024, 3, 14, tzinfo=UTC)
RUN_CYCLES = (0, 6, 12, 18)


class WeatherUnavailableError(RuntimeError):
    """No eligible archived forecast covers the requested forecast window."""


@dataclass(frozen=True)
class WeatherRunCheck:
    turbine_id: str
    run: str
    request_url: str
    ok: bool
    status: int | None
    message: str
    rows: int
    raw_output_json: str | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _utc(value: datetime, name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value.astimezone(UTC)


def _parse_run(run: str) -> datetime:
    parsed = datetime.strptime(run, "%Y-%m-%dT%H:%M").replace(tzinfo=UTC)
    if parsed.strftime("%Y-%m-%dT%H:%M") != run:
        raise ValueError("Run must use the exact YYYY-MM-DDTHH:MM UTC format")
    if parsed.minute or parsed.hour not in RUN_CYCLES:
        raise ValueError("ECMWF IFS runs must use 00, 06, 12 or 18 UTC")
    if parsed < ARCHIVE_START:
        raise ValueError("The ECMWF IFS Single Runs archive starts on 2024-03-14")
    return parsed


def _validate_settings(settings: WeatherSettings) -> None:
    # A model allowlist prevents silent use of ERA5 or stitched, lead-zero weather.
    if settings.endpoint.rstrip("/") != SINGLE_RUN_ENDPOINT:
        raise ValueError("Historical replay requires the Open-Meteo Single Runs endpoint")
    if settings.model != "ecmwf_ifs":
        raise ValueError("For February 2026 replay use the archived ecmwf_ifs model")
    if settings.timezone != "UTC":
        raise ValueError("Weather requests must use UTC")
    if settings.model_latency_hours < 6:
        raise ValueError("Use at least 6 hours for the assumed global-model publication latency")
    if not 1 <= settings.forecast_hours <= 240:
        raise ValueError("forecast_hours must be in 1..240")
    if not 1 <= settings.retry_attempts <= 10 or not 1 <= settings.fallback_runs <= 40:
        raise ValueError("retry_attempts must be 1..10 and fallback_runs 1..40")
    if not 0 <= settings.retry_backoff_seconds <= 30:
        raise ValueError("retry_backoff_seconds must be in 0..30")
    if not 0 < settings.request_timeout_seconds <= 120:
        raise ValueError("request_timeout_seconds must be in 1..120")


def build_single_run_url(settings: WeatherSettings, turbine: TurbineSettings, run: str) -> str:
    _validate_settings(settings)
    _parse_run(run)
    params = {
        "latitude": turbine.latitude,
        "longitude": turbine.longitude,
        "hourly": ",".join(settings.hourly_variables),
        "models": settings.model,
        "run": run,
        "forecast_hours": settings.forecast_hours,
        "timezone": "UTC",
        "timeformat": "iso8601",
        "wind_speed_unit": "ms",
        "temperature_unit": "celsius",
    }
    return f"{settings.endpoint.rstrip('/')}?{urlencode(params)}"


def eligible_runs(settings: WeatherSettings, calculation_time: datetime) -> list[str]:
    """Newest first, using an explicit *assumption* about historical publication delay.

    Initialization is not availability. Open-Meteo documents 4–6 h for global
    models and 10 min extra for server consistency. The example uses 8 h.
    Historical per-run publication timestamps are not supplied by this API.
    """
    _validate_settings(settings)
    latest = _utc(calculation_time, "calculation_time") - timedelta(
        hours=settings.model_latency_hours
    )
    latest = latest.replace(hour=(latest.hour // 6) * 6, minute=0, second=0, microsecond=0)
    return [
        candidate.strftime("%Y-%m-%dT%H:%M")
        for offset in range(settings.fallback_runs)
        if (candidate := latest - timedelta(hours=6 * offset)) >= ARCHIVE_START
    ]


def select_run_for_calculation_time(settings: WeatherSettings, calculation_time: datetime) -> str:
    candidates = eligible_runs(settings, calculation_time)
    if not candidates:
        raise WeatherUnavailableError("Calculation predates the Single Runs archive")
    return candidates[0]


def select_run_for_calculation_date(settings: WeatherSettings, calculation_date: date) -> str:
    """Compatibility helper; new replay code passes an explicit aware instant."""
    moment = datetime.combine(
        calculation_date, time(hour=settings.daily_calculation_hour_utc), tzinfo=UTC
    )
    return select_run_for_calculation_time(settings, moment)


def fetch_single_run_payload(
    settings: WeatherSettings, turbine: TurbineSettings, run: str
) -> tuple[dict[str, object], str, int]:
    url = build_single_run_url(settings, turbine, run)
    for attempt in range(settings.retry_attempts):
        try:
            with urlopen(url, timeout=settings.request_timeout_seconds) as response:
                status = response.status
                payload = json.loads(response.read().decode("utf-8"))
            if not isinstance(payload, dict) or payload.get("error"):
                raise ValueError(f"Open-Meteo returned an invalid/error payload: {payload!r}")
            return payload, url, status
        except HTTPError as exc:
            if exc.code != 429 and not 500 <= exc.code <= 599:
                raise
            if attempt + 1 == settings.retry_attempts:
                raise
        except (URLError, TimeoutError, ConnectionError):
            if attempt + 1 == settings.retry_attempts:
                raise
        sleep(min(30, settings.retry_backoff_seconds * 2**attempt))
    raise AssertionError("Unreachable retry state")


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _payload_hash(payload: dict) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _write_cache(output_dir: Path, envelope: dict) -> Path:
    params = envelope["request"]["params"]
    key = hashlib.sha256(_canonical_json(envelope["request"]).encode()).hexdigest()
    run_key = _parse_run(params["run"]).strftime("%Y%m%dT%H%M")
    path = output_dir / f"run_{run_key}_{key}.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(_canonical_json(envelope), encoding="utf-8")
    temporary.replace(path)
    return path


def _validate_request(request: dict) -> dict:
    if not isinstance(request, dict) or request.get("endpoint") != SINGLE_RUN_ENDPOINT:
        raise ValueError("Cache provenance must identify the exact Single Runs endpoint")
    params = request.get("params", {})
    if not isinstance(params, dict):
        raise TypeError("Cache request params must be an object")
    if params.get("models") != "ecmwf_ifs" or params.get("timezone") != "UTC":
        raise ValueError("Cache must identify model ecmwf_ifs and UTC")
    if params.get("wind_speed_unit") != "ms":
        raise ValueError("Cache wind speed unit must be ms")
    if params.get("temperature_unit", "celsius") != "celsius":
        raise ValueError("Cache temperature unit must be celsius")
    _parse_run(params["run"])
    if not -90 <= float(params["latitude"]) <= 90:
        raise ValueError("Invalid cache latitude")
    if not -180 <= float(params["longitude"]) <= 180:
        raise ValueError("Invalid cache longitude")
    return params


def _validate_payload(payload: dict, run: str, required_variables: tuple[str, ...]):
    import pandas as pd

    if not isinstance(payload, dict) or payload.get("error"):
        raise ValueError("Weather response is not a successful object")
    if payload.get("utc_offset_seconds") != 0:
        raise ValueError("Weather response must explicitly declare utc_offset_seconds=0")
    hourly = payload.get("hourly")
    if not isinstance(hourly, dict) or not hourly.get("time"):
        raise ValueError("Weather response has no hourly timestamps")
    length = len(hourly["time"])
    if any(not isinstance(values, list) or len(values) != length for values in hourly.values()):
        raise ValueError("Weather hourly columns have inconsistent lengths")
    times = pd.DatetimeIndex(pd.to_datetime(hourly["time"], utc=True, errors="raise"))
    if times.hasnans or times.has_duplicates or not times.is_monotonic_increasing:
        raise ValueError("Weather timestamps must be unique, finite and increasing")
    if not times.equals(pd.date_range(times[0], periods=length, freq="h")):
        raise ValueError("Weather timestamps must be contiguous hourly samples")
    if times[0] != pd.Timestamp(_parse_run(run)):
        raise ValueError("Weather payload must start at the requested UTC initialization")
    if any(variable not in hourly for variable in required_variables):
        raise ValueError("Weather response is missing a required variable")
    frame = pd.DataFrame(hourly)
    frame["timestamp"] = times
    units = payload.get("hourly_units", {})
    for variable in required_variables:
        expected_unit = "m/s" if variable.startswith("wind_speed") else "°C"
        if units.get(variable) != expected_unit:
            raise ValueError(f"Unexpected or absent unit for {variable}: {units.get(variable)!r}")
    return frame


def import_weather_archive(archive_path: Path, output_dir: Path) -> dict[str, object]:
    """Import cached data only; never extract or execute code from a submitted archive.

    The old SQLite checksum proves consistency with the supplied cache, not
    authenticity or historical public availability. Preserve that distinction.
    """
    archive_path, output_dir = Path(archive_path), Path(output_dir)
    archive_hash = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    imported = 0
    runs = set()
    with zipfile.ZipFile(archive_path) as archive:
        members = [info for info in archive.infolist() if info.filename.endswith("/weather.sqlite")]
        if len(members) != 1:
            raise ValueError("Expected exactly one weather.sqlite cache in the archive")
        member = members[0]
        if member.file_size > 256 * 1024 * 1024:
            raise ValueError("Weather cache exceeds the 256 MiB import limit")
        blob = bytearray(archive.read(member))
    if not blob.startswith(b"SQLite format 3\x00"):
        raise ValueError("Archive weather cache is not a SQLite database")
    # deserialize cannot open a WAL-mode database without a WAL file. Changing
    # these two header flags on this private in-memory copy makes it read-only
    # rollback mode. No source file or archive content is modified.
    blob[18] = blob[19] = 1
    connection = sqlite3.connect(":memory:")
    try:
        connection.deserialize(bytes(blob))
        connection.execute("PRAGMA query_only=ON")
        connection.execute("PRAGMA trusted_schema=OFF")
        rows = connection.execute(
            "SELECT request_json,payload_json,payload_sha256,fetched_at FROM weather_cache"
        )
        for request_json, payload_json, original_hash, fetched_at in rows:
            if hashlib.sha256(payload_json.encode("utf-8")).hexdigest() != original_hash:
                raise ValueError("Archive payload SHA256 verification failed")
            request, payload = json.loads(request_json), json.loads(payload_json)
            params = _validate_request(request)
            _validate_payload(payload, params["run"], ("wind_speed_100m", "temperature_2m"))
            _utc(datetime.fromisoformat(fetched_at), "fetched_at")
            envelope = {
                "schema_version": 1,
                "request": request,
                "request_url": f"{request['endpoint']}?{urlencode(params)}",
                "payload": payload,
                "payload_sha256": _payload_hash(payload),
                "fetched_at": fetched_at,
                "source": "provided_archive_cache",
                "provenance": {
                    "archive_path": str(archive_path.resolve()),
                    "archive_sha256": archive_hash,
                    "archive_member": member.filename,
                    "original_payload_sha256": original_hash,
                    "original_checksum_verified": True,
                    "historical_publication_verified": False,
                    "as_issued_authenticity_verified": False,
                },
            }
            _write_cache(output_dir, envelope)
            runs.add(params["run"])
            imported += 1
    finally:
        connection.close()
    return {
        "imported": imported,
        "source_archive": str(archive_path.resolve()),
        "archive_sha256": archive_hash,
        "output_dir": str(output_dir),
        "earliest_run": min(runs) if runs else None,
        "latest_run": max(runs) if runs else None,
        "historical_publication_verified": False,
    }


def _cache_candidates(output_dir: Path, run: str):
    key = _parse_run(run).strftime("%Y%m%dT%H%M")
    return sorted(output_dir.glob(f"run_{key}_*.json"))


def get_weather_for_forecast(
    settings: WeatherSettings,
    turbine: TurbineSettings,
    as_of: datetime,
    target_times,
    output_dir: Path,
    *,
    offline: bool = False,
    refresh: bool = False,
):
    """Return complete UTC feature rows and honest run-level provenance.

    In offline mode, select the newest valid eligible cache within fallback_runs.
    Online refresh probes each eligible run newest first and can use a previous
    valid cache after a network failure. Failed/incomplete runs never contribute
    partially filled values to a forecast.
    """
    from urllib.parse import parse_qsl, urlparse

    import numpy as np
    import pandas as pd

    _validate_settings(settings)
    as_of = _utc(as_of, "as_of")
    targets = pd.DatetimeIndex(target_times)
    if targets.tz is None:
        raise ValueError("target_times must be timezone-aware")
    targets = targets.tz_convert("UTC")
    if len(targets) == 0 or targets.hasnans or targets.has_duplicates:
        raise ValueError("target_times must contain unique, finite timestamps")
    if not targets.equals(pd.date_range(targets[0], periods=len(targets), freq="h")):
        raise ValueError("target_times must be contiguous increasing hours")
    if targets[0] <= pd.Timestamp(as_of):
        raise ValueError("Every target time must be strictly after the calculation instant")
    if targets[0].minute or targets[0].second or targets[0].microsecond:
        raise ValueError("Target times must fall on full UTC hours")
    if offline and refresh:
        raise ValueError("offline and refresh are mutually exclusive")
    attempts = []
    required = (settings.wind_speed_variable, settings.temperature_variable)
    output_dir = Path(output_dir)

    def validate_envelope(envelope, run, path):
        if envelope.get("schema_version") != 1:
            raise ValueError("Unsupported weather cache schema")
        params = _validate_request(envelope["request"])
        if params["run"] != run:
            raise ValueError("Cache run differs from requested run")
        for name, expected in (("latitude", turbine.latitude), ("longitude", turbine.longitude)):
            if not math.isclose(float(params[name]), expected, rel_tol=0, abs_tol=1e-8):
                return None
        payload = envelope["payload"]
        if _payload_hash(payload) != envelope["payload_sha256"]:
            raise ValueError("Weather cache payload SHA256 mismatch")
        frame = _validate_payload(payload, run, required).set_index("timestamp")
        if not targets.isin(frame.index).all():
            raise ValueError("Weather run does not cover every requested target hour")
        result = frame.loc[targets, list(required)].copy()
        result.columns = ["wind_speed", "temperature"]
        result = result.apply(pd.to_numeric, errors="raise")
        if not np.isfinite(result.to_numpy(dtype=float)).all():
            raise ValueError("Weather run contains missing or non-finite forecast values")
        if (result["wind_speed"] < 0).any():
            raise ValueError("Weather run contains negative wind speed")
        available_at = _parse_run(run) + timedelta(hours=settings.model_latency_hours)
        if available_at > as_of:
            raise ValueError("Weather run was not eligible at the calculation instant")
        metadata = {
            "run": run,
            "model": settings.model,
            "payload_sha256": envelope["payload_sha256"],
            "request_url": f"{envelope['request']['endpoint']}?{urlencode(params)}",
            "assumed_available_at": available_at.isoformat(),
            "publication_latency_hours": settings.model_latency_hours,
            "historical_publication_verified": False,
            "availability_basis": "initialization_plus_configured_latency_assumption",
            "documentation_url": DOCUMENTATION_URL,
            "retrieved_at": envelope["fetched_at"],
            "source": envelope["source"],
            "cache_file": str(path),
            "provenance": envelope.get("provenance", {}),
            "grid_latitude": payload.get("latitude"),
            "grid_longitude": payload.get("longitude"),
            "as_of": as_of.isoformat(),
            "attempts": list(attempts),
            "fallback_used": run != eligible_runs(settings, as_of)[0],
        }
        return result.rename_axis("timestamp").reset_index(), metadata

    def load_cached(run):
        entries = []
        for path in _cache_candidates(output_dir, run):
            try:
                envelope = json.loads(path.read_text(encoding="utf-8"))
                fetched = _utc(datetime.fromisoformat(envelope["fetched_at"]), "fetched_at")
                entries.append((fetched, path, envelope))
            except (KeyError, TypeError, ValueError, OSError) as exc:
                attempts.append({"run": run, "action": "cache_rejected", "error": str(exc)})
        for _, path, envelope in sorted(entries, key=lambda item: item[0], reverse=True):
            try:
                result = validate_envelope(envelope, run, path)
                if result is not None:
                    return result
            except (KeyError, TypeError, ValueError, OSError) as exc:
                attempts.append({"run": run, "action": "cache_rejected", "error": str(exc)})
        return None

    for run in eligible_runs(settings, as_of):
        if not refresh:
            result = load_cached(run)
            if result is not None:
                return result
        if not offline:
            try:
                payload, url, status = fetch_single_run_payload(settings, turbine, run)
                if not 200 <= status < 300:
                    raise ValueError(f"HTTP {status}")
                parts = urlparse(url)
                request = {
                    "endpoint": f"{parts.scheme}://{parts.netloc}{parts.path}",
                    "params": dict(parse_qsl(parts.query)),
                }
                envelope = {
                    "schema_version": 1,
                    "request": request,
                    "request_url": url,
                    "payload": payload,
                    "payload_sha256": _payload_hash(payload),
                    "fetched_at": datetime.now(UTC).isoformat(),
                    "source": "open_meteo_single_run",
                    "provenance": {"historical_publication_verified": False},
                }
                # Validate full target coverage before making this entry reusable.
                result = validate_envelope(envelope, run, output_dir)
                path = _write_cache(output_dir, envelope)
                result[1]["cache_file"] = str(path)
                return result
            except (HTTPError, URLError, TimeoutError, ValueError, ConnectionError) as exc:
                attempts.append({"run": run, "action": "fetch_failed", "error": str(exc)})
                # Authentication/configuration failures affect every run; don't
                # hammer the provider. An existing valid cache is still usable.
                if isinstance(exc, HTTPError) and exc.code in (401, 403):
                    offline = True
        if refresh:
            result = load_cached(run)
            if result is not None:
                return result
        attempts.append({"run": run, "action": "no_usable_cached_run"})
    raise WeatherUnavailableError(
        f"No eligible archived weather for {turbine.id} at {as_of.isoformat()}; "
        f"attempts={json.dumps(attempts, ensure_ascii=False)}"
    )


def check_single_run(
    settings: WeatherSettings, turbine: TurbineSettings, run: str, output_dir: Path
) -> WeatherRunCheck:
    url = build_single_run_url(settings, turbine, run)
    try:
        payload, _, status = fetch_single_run_payload(settings, turbine, run)
        frame = _validate_payload(
            payload, run, (settings.wind_speed_variable, settings.temperature_variable)
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{turbine.id}_{run.replace(':', '')}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return WeatherRunCheck(turbine.id, run, url, True, status, "ok", len(frame), str(path))
    except (HTTPError, URLError, TimeoutError, ValueError) as exc:
        return WeatherRunCheck(
            turbine.id, run, url, False, getattr(exc, "code", None), str(exc), 0, None
        )
