"""Original NOAA GFS forecasts, with per-object historical availability evidence.

Only the operational 0.25 degree pgrb2 archive is accepted. HTTP Last-Modified
must precede the simulated issue time; later archive copies fail closed. Local
caches contain extracted points and source hashes, never substitute analyses.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path
from threading import Lock
from time import sleep
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import uuid4

from .settings import TurbineSettings, WeatherSettings

BUCKET_URL = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
DOCUMENTATION_URL = "https://registry.opendata.aws/noaa-gfs-bdp-pds/"
MAX_INDEX_BYTES = 256 * 1024
MAX_GRIB_BYTES = 5_000_000
MAX_WORKERS = 4
_DECODE_LOCK = Lock()
_FIELDS = {
    "u": ("UGRD", "100 m above ground", ("100u", "u"), 100, "m s**-1"),
    "v": ("VGRD", "100 m above ground", ("100v", "v"), 100, "m s**-1"),
    "t": ("TMP", "2 m above ground", ("2t", "t"), 2, "K"),
}


class GFSUnavailableError(RuntimeError):
    """No complete original operational forecast is proven available by issue time."""


class GFSDependencyError(RuntimeError):
    """The optional ecCodes package or native library is unavailable."""


def _utc(value: datetime, name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value.astimezone(UTC)


def _iso(value: datetime) -> str:
    return _utc(value, "timestamp").isoformat()


def _json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _validate_settings(settings: WeatherSettings) -> None:
    if settings.model not in ("gfs", "gfs_0p25", "noaa_gfs"):
        raise ValueError("NOAA provider requires model=gfs_0p25")
    if settings.timezone != "UTC":
        raise ValueError("NOAA weather uses UTC")
    if settings.model_latency_hours < 8:
        raise ValueError("NOAA GFS replay requires a conservative latency of at least 8 hours")
    if settings.wind_speed_variable != "wind_speed_100m":
        raise ValueError("NOAA provider supports wind_speed_100m")
    if settings.temperature_variable != "temperature_2m":
        raise ValueError("NOAA provider supports temperature_2m")
    if not 1 <= settings.fallback_runs <= 40 or not 1 <= settings.retry_attempts <= 10:
        raise ValueError("fallback_runs must be 1..40 and retry_attempts 1..10")
    if not 0 < settings.request_timeout_seconds <= 120:
        raise ValueError("request_timeout_seconds must be in 1..120")
    if not 0 <= settings.retry_backoff_seconds <= 30:
        raise ValueError("retry_backoff_seconds must be in 0..30")


def eligible_runs(settings: WeatherSettings, as_of: datetime) -> list[datetime]:
    _validate_settings(settings)
    latest = _utc(as_of, "as_of") - timedelta(hours=settings.model_latency_hours)
    latest = latest.replace(hour=6 * (latest.hour // 6), minute=0, second=0, microsecond=0)
    return [latest - timedelta(hours=6 * offset) for offset in range(settings.fallback_runs)]


def build_gfs_url(run: datetime, lead: int) -> str:
    run = _utc(run, "run")
    if run.hour % 6 or run.minute or run.second or run.microsecond:
        raise ValueError("GFS initialization must be on a six-hour UTC cycle")
    if isinstance(lead, bool) or not isinstance(lead, int) or not 1 <= lead <= 120:
        raise ValueError("Only original hourly GFS forecast leads 1..120 are supported")
    return (
        f"{BUCKET_URL}/gfs.{run:%Y%m%d}/{run:%H}/atmos/"
        f"gfs.t{run:%H}z.pgrb2.0p25.f{lead:03d}"
    )


def _available(modified: str, run: datetime, as_of: datetime) -> datetime:
    try:
        value = datetime.fromisoformat(modified)
    except ValueError:
        value = parsedate_to_datetime(modified)
    value = _utc(value, "Last-Modified")
    if value < run or value > as_of:
        raise ValueError("Source Last-Modified is not between initialization and issue time")
    return value


def _http_bytes(
    url: str, settings: WeatherSettings, *, limit: int, byte_range: tuple[int, int | None] | None,
) -> tuple[bytes, dict]:
    """Read only bounded responses. A server ignoring Range is never downloaded."""
    headers = {"Accept-Encoding": "identity", "User-Agent": "wind-power-forecast/0.1"}
    if byte_range is not None:
        start, end = byte_range
        if start < 0 or (end is not None and not start <= end < start + limit):
            raise ValueError("GRIB byte range exceeds the download limit")
        headers["Range"] = f"bytes={start}-{'' if end is None else end}"
    for attempt in range(settings.retry_attempts):
        try:
            with urlopen(Request(url, headers=headers), timeout=settings.request_timeout_seconds) as response:
                if response.geturl() != url:
                    raise ValueError("NOAA archive response redirected to an unexpected URL")
                expected_status = 206 if byte_range is not None else 200
                if response.status != expected_status:
                    raise ValueError(f"Expected HTTP {expected_status}; refused unbounded archive response")
                if response.headers.get("Content-Encoding", "identity") != "identity":
                    raise ValueError("Compressed archive transport is not supported")
                modified = response.headers.get("Last-Modified")
                if not modified:
                    raise ValueError("NOAA response is missing Last-Modified availability evidence")
                length = int(response.headers.get("Content-Length", "-1"))
                if not 0 < length <= limit:
                    raise ValueError("Missing Content-Length or response exceeds the download limit")
                content_range = response.headers.get("Content-Range")
                if byte_range is not None:
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range or "")
                    if not match:
                        raise ValueError("Range response has no valid Content-Range")
                    first, last, total = map(int, match.groups())
                    if first != start or last >= total or last - first + 1 != length:
                        raise ValueError("Range response does not match requested bytes")
                    if (end is not None and last != end) or (end is None and last != total - 1):
                        raise ValueError("Range response does not match requested end")
                blob = response.read(length + 1)
                if len(blob) != length:
                    raise ValueError("Truncated or oversized NOAA response")
                return blob, {
                    "url": url, "last_modified": _iso(parsedate_to_datetime(modified)),
                    "sha256": hashlib.sha256(blob).hexdigest(),
                    "content_range": content_range, "etag": response.headers.get("ETag"),
                }
        except HTTPError as exc:
            exc.close()
            if (exc.code != 429 and not 500 <= exc.code <= 599) or attempt + 1 == settings.retry_attempts:
                raise
        except (URLError, TimeoutError, ConnectionError):
            if attempt + 1 == settings.retry_attempts:
                raise
        sleep(min(30, settings.retry_backoff_seconds * 2**attempt))
    raise AssertionError("Unreachable retry state")


def _index_ranges(blob: bytes, run: datetime, lead: int) -> dict[str, tuple[int, int | None]]:
    if len(blob) > MAX_INDEX_BYTES:
        raise ValueError("GFS index exceeds the 256 KiB limit")
    rows = []
    for line in blob.decode("ascii").splitlines():
        parts = line.split(":")
        if len(parts) < 7:
            raise ValueError("Malformed GFS inventory line")
        offset = int(parts[1])
        if offset < 0 or (rows and offset <= rows[-1][0]):
            raise ValueError("GFS inventory offsets must be strictly increasing")
        rows.append((offset, parts))
    found = {}
    for position, (offset, parts) in enumerate(rows):
        for key, (variable, level, _, _, _) in _FIELDS.items():
            if parts[3:5] != [variable, level]:
                continue
            if key in found:
                raise ValueError("Duplicate required field in GFS inventory")
            if parts[2] != f"d={run:%Y%m%d%H}" or parts[5] != f"{lead} hour fcst":
                raise ValueError("GFS inventory initialization or forecast lead does not match")
            end = rows[position + 1][0] - 1 if position + 1 < len(rows) else None
            if end is not None and end - offset + 1 > MAX_GRIB_BYTES:
                raise ValueError("Selected GRIB field exceeds the 5 MB limit")
            found[key] = (offset, end)
    if set(found) != set(_FIELDS):
        raise ValueError("GFS inventory lacks UGRD/VGRD at 100 m or TMP at 2 m")
    return found


def _nearest_grid(latitude: float, longitude: float) -> tuple[float, float]:
    if not math.isfinite(latitude) or not -90 <= latitude <= 90:
        raise ValueError("Invalid turbine latitude")
    if not math.isfinite(longitude) or not -180 <= longitude <= 180:
        raise ValueError("Invalid turbine longitude")
    # Ties choose the northern/eastern grid node; both choices are equally near.
    return math.floor(latitude * 4 + 0.5) / 4, (math.floor((longitude % 360) * 4 + 0.5) / 4) % 360


def _validate_grib_metadata(metadata: dict, field: str, run: datetime, lead: int) -> None:
    _, _, short_names, level, units = _FIELDS[field]
    expected = {
        "edition": 2, "dataDate": int(run.strftime("%Y%m%d")),
        "dataTime": run.hour * 100, "forecastTime": lead, "stepUnits": 1,
        "stepType": "instant", "typeOfLevel": "heightAboveGround", "level": level,
        "gridType": "regular_ll", "Ni": 1440, "Nj": 721,
        "iDirectionIncrementInDegrees": 0.25, "jDirectionIncrementInDegrees": 0.25,
        "latitudeOfFirstGridPointInDegrees": 90.0,
        "longitudeOfFirstGridPointInDegrees": 0.0,
        "validityDate": int((run + timedelta(hours=lead)).strftime("%Y%m%d")),
        "validityTime": int((run + timedelta(hours=lead)).strftime("%H%M")),
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"GRIB {key} differs from the requested operational forecast: {metadata.get(key)!r}")
    if metadata.get("shortName") not in short_names or metadata.get("units") != units:
        raise ValueError("GRIB variable or units differ from the requested field")
    if str(metadata.get("centre")) not in ("kwbc", "7"):
        raise ValueError("GRIB centre is not NOAA/NCEP")
    if metadata.get("typeOfGeneratingProcess") != 2:
        raise ValueError("GRIB must describe a forecast, not an analysis or hindcast substitute")


def _decode_grib(blob: bytes, field: str, run: datetime, lead: int, grid: tuple[float, float]) -> dict:
    if len(blob) < 20 or blob[:4] != b"GRIB" or blob[7] != 2 or blob[-4:] != b"7777":
        raise ValueError("Range is not a complete GRIB2 message")
    if int.from_bytes(blob[8:16], "big") != len(blob):
        raise ValueError("GRIB2 message length does not match the selected range")
    try:
        from .native import prepare_eccodes_native

        prepare_eccodes_native()
        import eccodes
    except (ImportError, RuntimeError, OSError) as exc:
        raise GFSDependencyError("Install the optional eccodes Python package and native ecCodes library") from exc
    keys = (
        "edition", "dataDate", "dataTime", "forecastTime", "stepUnits", "stepType",
        "typeOfLevel", "level", "shortName", "units", "centre", "typeOfGeneratingProcess",
        "gridType", "Ni", "Nj", "iDirectionIncrementInDegrees", "jDirectionIncrementInDegrees",
        "latitudeOfFirstGridPointInDegrees", "longitudeOfFirstGridPointInDegrees",
        "validityDate", "validityTime",
    )
    # Network requests run concurrently; each native decoder handle is isolated
    # and serialized for ecCodes builds without native thread support.
    with _DECODE_LOCK:
        handle = eccodes.codes_new_from_message(blob)
        try:
            metadata = {key: eccodes.codes_get(handle, key) for key in keys}
            _validate_grib_metadata(metadata, field, run, lead)
            nearest = eccodes.codes_grib_find_nearest(handle, grid[0], grid[1])[0]
            latitude, longitude, value = float(nearest["lat"]), float(nearest["lon"]), float(nearest["value"])
            if not math.isclose(latitude, grid[0], abs_tol=1e-7) or not math.isclose(longitude % 360, grid[1], abs_tol=1e-7):
                raise ValueError("Decoded grid node differs from the requested nearest 0.25 degree node")
            if not math.isfinite(value) or value == eccodes.codes_get(handle, "missingValue"):
                raise ValueError("GFS point contains a missing/non-finite forecast value")
            return {"value": value, "grid_latitude": latitude, "grid_longitude": longitude % 360,
                    "grib_metadata": metadata}
        finally:
            eccodes.codes_release(handle)


def _cache_path(output_dir: Path, run: datetime, lead: int, grid: tuple[float, float]) -> Path:
    return output_dir / "gfs_points" / f"{run:%Y%m%d%H}_f{lead:03d}_{grid[0]:.2f}_{grid[1]:.2f}.json"


def _validate_point(payload: dict, run: datetime, lead: int, grid: tuple[float, float], as_of: datetime) -> None:
    if payload.get("provider") != "noaa-gfs" or payload.get("run") != _iso(run) or payload.get("lead") != lead:
        raise ValueError("GFS point cache does not match the requested operational run/lead")
    if payload.get("grid") != list(grid) or payload.get("valid_time") != _iso(run + timedelta(hours=lead)):
        raise ValueError("GFS point cache has a different grid or target time")
    url = build_gfs_url(run, lead)
    index = payload["index"]
    if index.get("url") != url + ".idx" or not re.fullmatch(r"[0-9a-f]{64}", index.get("sha256", "")):
        raise ValueError("GFS point cache lacks NOAA inventory provenance")
    _available(index["last_modified"], run, as_of)
    if set(payload["fields"]) != set(_FIELDS):
        raise ValueError("GFS point cache is missing required fields")
    object_versions = set()
    for field, extracted in payload["fields"].items():
        source = extracted["source"]
        if source.get("url") != url or not re.fullmatch(r"[0-9a-f]{64}", source.get("sha256", "")):
            raise ValueError("GFS point cache lacks NOAA GRIB provenance")
        if not re.fullmatch(r"bytes \d+-\d+/\d+", source.get("content_range", "")):
            raise ValueError("GFS point cache lacks byte range provenance")
        _available(source["last_modified"], run, as_of)
        if not isinstance(source.get("etag"), str) or not source["etag"]:
            raise ValueError("GFS range response lacks an object ETag")
        object_versions.add((source["etag"], source["last_modified"]))
        _validate_grib_metadata(extracted["grib_metadata"], field, run, lead)
        if [extracted["grid_latitude"], extracted["grid_longitude"]] != list(grid):
            raise ValueError("GFS point cache fields have inconsistent grid coordinates")
        value = float(extracted["value"])
        if not math.isfinite(value) or (field == "t" and not 100 <= value <= 400):
            raise ValueError("GFS point cache contains an invalid physical forecast value")
    if len(object_versions) != 1:
        raise ValueError("GFS fields came from different versions of the same GRIB object")


def _point_for_lead(settings, run, lead, grid, as_of, output_dir, offline, refresh):
    path = _cache_path(output_dir, run, lead, grid)
    if path.exists() and not refresh:
        try:
            if path.stat().st_size > MAX_INDEX_BYTES:
                raise ValueError("GFS point cache exceeds its size limit")
            envelope = json.loads(path.read_text(encoding="utf-8"))
            if envelope.get("schema_version") != 1 or _digest(envelope["payload"]) != envelope["payload_sha256"]:
                raise ValueError("GFS point cache checksum/schema mismatch")
            _validate_point(envelope["payload"], run, lead, grid, as_of)
            return envelope
        except (KeyError, TypeError, ValueError):
            if offline:
                raise
    if offline:
        raise GFSUnavailableError(f"No validated cached NOAA GFS point for {run:%Y-%m-%dT%H:%M} f{lead:03d}")
    url = build_gfs_url(run, lead)
    index_blob, index_source = _http_bytes(url + ".idx", settings, limit=MAX_INDEX_BYTES, byte_range=None)
    _available(index_source["last_modified"], run, as_of)
    ranges = _index_ranges(index_blob, run, lead)
    fields = {}
    for field, byte_range in ranges.items():
        blob, source = _http_bytes(url, settings, limit=MAX_GRIB_BYTES, byte_range=byte_range)
        _available(source["last_modified"], run, as_of)
        fields[field] = {**_decode_grib(blob, field, run, lead, grid), "source": source}
    payload = {
        "provider": "noaa-gfs", "run": _iso(run), "lead": lead,
        "valid_time": _iso(run + timedelta(hours=lead)), "grid": list(grid),
        "index": index_source, "fields": fields,
    }
    _validate_point(payload, run, lead, grid, as_of)
    envelope = {"schema_version": 1, "payload": payload, "payload_sha256": _digest(payload),
                "retrieved_at": _iso(datetime.now(UTC))}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{uuid4().hex}.tmp")
    try:
        temporary.write_text(_json(envelope), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return envelope


def get_weather_for_forecast(
    settings: WeatherSettings, turbine: TurbineSettings, as_of: datetime, target_times,
    output_dir: Path, *, offline: bool = False, refresh: bool = False,
):
    """Return exact hourly UTC features from a single eligible original GFS run.

    Retrospective retrieval is allowed, but every source object must carry a
    Last-Modified at/before as_of. Offline replay repeats all metadata checks.
    A complete run is required: missing leads never mix runs or use observations.
    """
    import pandas as pd

    _validate_settings(settings)
    as_of = _utc(as_of, "as_of")
    if offline and refresh:
        raise ValueError("offline and refresh are mutually exclusive")
    targets = pd.DatetimeIndex(target_times)
    if targets.tz is None or targets.empty or targets.hasnans or targets.has_duplicates:
        raise ValueError("Target times must be unique finite timezone-aware hours")
    targets = targets.tz_convert("UTC")
    if not targets.equals(pd.date_range(targets[0], periods=len(targets), freq="h")):
        raise ValueError("Target times must form a contiguous increasing hourly grid")
    if targets[0] <= pd.Timestamp(as_of) or targets[0].minute or targets[0].second or targets[0].microsecond or targets[0].nanosecond:
        raise ValueError("Target times must be full hours strictly after issue time")
    if len(targets) > 48:
        raise ValueError("At most 48 hourly targets may be requested")
    grid = _nearest_grid(turbine.latitude, turbine.longitude)
    output_dir = Path(output_dir)
    attempts = []
    candidates = eligible_runs(settings, as_of)
    for run in candidates:
        leads = [int((target.to_pydatetime() - run).total_seconds() // 3600) for target in targets]
        if min(leads) < 1 or max(leads) > 120:
            attempts.append({"run": _iso(run), "error": "Outside the exact hourly forecast range"})
            continue
        try:
            points = {}
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = {
                    executor.submit(_point_for_lead, settings, run, lead, grid, as_of,
                                    output_dir, offline, refresh): lead
                    for lead in leads
                }
                try:
                    for future in as_completed(futures):
                        points[futures[future]] = future.result()
                except Exception:
                    for future in futures:
                        future.cancel()
                    raise
            ordered = [points[lead] for lead in leads]
            rows = []
            modified = []
            for target, envelope in zip(targets, ordered):
                payload = envelope["payload"]
                fields = payload["fields"]
                rows.append({
                    "timestamp": target,
                    "wind_speed": math.hypot(fields["u"]["value"], fields["v"]["value"]),
                    "temperature": fields["t"]["value"] - 273.15,
                })
                modified.extend([payload["index"]["last_modified"], *[
                    field["source"]["last_modified"] for field in fields.values()
                ]])
            if any(not math.isfinite(row[column]) for row in rows for column in ("wind_speed", "temperature")):
                raise ValueError("GFS conversion produced non-finite weather features")
            metadata = {
                "run": run.strftime("%Y-%m-%dT%H:%M"), "model": "gfs_0p25", "source": "noaa-gfs",
                "request_url": f"{BUCKET_URL}/gfs.{run:%Y%m%d}/{run:%H}/atmos/",
                "payload_sha256": _digest([point["payload_sha256"] for point in ordered]),
                "assumed_available_at": _iso(run + timedelta(hours=settings.model_latency_hours)),
                "verified_available_at": max(modified),
                "historical_publication_verified": True, "as_issued_authenticity_verified": True,
                "availability_basis": "official_NOAA_S3_Last-Modified_and_original_GRIB_run_metadata",
                "documentation_url": DOCUMENTATION_URL,
                "requested_latitude": turbine.latitude, "requested_longitude": turbine.longitude,
                "grid_latitude": grid[0], "grid_longitude": grid[1],
                "sampling": "nearest_0.25_degree_grid_node_without_temporal_interpolation",
                "wind_speed_unit": "m/s", "temperature_unit": "celsius",
                "as_of": _iso(as_of), "attempts": attempts, "fallback_used": run != candidates[0],
                "retrieved_at": max(point["retrieved_at"] for point in ordered),
                "cache_files": [str(_cache_path(output_dir, run, lead, grid)) for lead in leads],
                "point_payload_sha256": [point["payload_sha256"] for point in ordered],
            }
            return pd.DataFrame(rows), metadata
        except GFSDependencyError:
            raise
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
            attempts.append({"run": _iso(run), "error": f"{type(exc).__name__}: {exc}"})
    raise GFSUnavailableError("No original NOAA GFS run meets issue-time checks: " + _json(attempts))
