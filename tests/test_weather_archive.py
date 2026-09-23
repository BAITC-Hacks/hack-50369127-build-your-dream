import hashlib
import io
import json
import sqlite3
import sys
import zipfile
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError, URLError

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wind_power_forecast.settings import TurbineSettings, WeatherSettings
from wind_power_forecast.weather import (
    SINGLE_RUN_ENDPOINT,
    WeatherUnavailableError,
    _payload_hash,
    _write_cache,
    build_single_run_url,
    eligible_runs,
    fetch_single_run_payload,
    get_weather_for_forecast,
    import_weather_archive,
    select_run_for_calculation_time,
)


@pytest.fixture
def weather():
    return WeatherSettings(
        endpoint=SINGLE_RUN_ENDPOINT,
        model="ecmwf_ifs",
        forecast_hours=168,
        timezone="UTC",
        daily_calculation_hour_utc=18,
        model_latency_hours=8,
        wind_speed_variable="wind_speed_100m",
        temperature_variable="temperature_2m",
        hourly_variables=("wind_speed_100m", "temperature_2m"),
        retry_backoff_seconds=0,
    )


@pytest.fixture
def turbine():
    return TurbineSettings("turbine_1", 43.64515, 78.535604, None, None, "self")


def payload(run, periods=168):
    times = pd.date_range(run, periods=periods, freq="h")
    return {
        "utc_offset_seconds": 0,
        "latitude": 43.620384,
        "longitude": 78.47891,
        "hourly_units": {"wind_speed_100m": "m/s", "temperature_2m": "°C"},
        "hourly": {
            "time": times.strftime("%Y-%m-%dT%H:%M").tolist(),
            "wind_speed_100m": [5.0] * periods,
            "temperature_2m": [-2.0] * periods,
        },
    }


def cache(tmp_path, turbine, run="2026-01-30T12:00", body=None, fetched=None):
    body = payload(run) if body is None else body
    request = {
        "endpoint": SINGLE_RUN_ENDPOINT,
        "params": {
            "latitude": turbine.latitude,
            "longitude": turbine.longitude,
            "models": "ecmwf_ifs",
            "timezone": "UTC",
            "run": run,
            "wind_speed_unit": "ms",
            "temperature_unit": "celsius",
            "forecast_days": 7,
            "hourly": "wind_speed_100m,temperature_2m",
        },
    }
    envelope = {
        "schema_version": 1,
        "request": request,
        "payload": body,
        "payload_sha256": _payload_hash(body),
        "fetched_at": fetched or "2026-09-23T00:00:00+00:00",
        "source": "provided_archive_cache",
        "provenance": {"historical_publication_verified": False},
    }
    return _write_cache(tmp_path, envelope), envelope


def targets():
    return pd.date_range("2026-02-01", periods=48, freq="h", tz="Asia/Almaty")


AS_OF = datetime(2026, 1, 31, 18, tzinfo=UTC)


def test_run_selection_uses_aware_instant_and_publication_latency(weather):
    assert select_run_for_calculation_time(weather, AS_OF) == "2026-01-31T06:00"
    assert (
        select_run_for_calculation_time(
            weather, pd.Timestamp("2026-01-31T23:00", tz="Asia/Almaty").to_pydatetime()
        )
        == "2026-01-31T06:00"
    )
    assert eligible_runs(weather, AS_OF)[-1] == "2026-01-30T12:00"
    just_before_release = datetime(2026, 1, 31, 13, 59, 59, tzinfo=UTC)
    assert select_run_for_calculation_time(weather, just_before_release) == "2026-01-31T00:00"
    assert (
        select_run_for_calculation_time(weather, just_before_release + timedelta(seconds=1))
        == "2026-01-31T06:00"
    )
    with pytest.raises(ValueError, match="timezone-aware"):
        select_run_for_calculation_time(weather, datetime(2026, 1, 31, 18))  # noqa: DTZ001


@pytest.mark.parametrize(
    "changes,run",
    [
        ({"model": "era5"}, "2026-01-31T12:00"),
        ({"model": "gfs_global"}, "2026-01-31T12:00"),
        ({"endpoint": "https://archive-api.open-meteo.com/v1/archive"}, "2026-01-31T12:00"),
        ({"timezone": "Asia/Almaty"}, "2026-01-31T12:00"),
        ({"model_latency_hours": 0}, "2026-01-31T12:00"),
        ({}, "2026-01-31T03:00"),
        ({}, "2026-01-31T06:30"),
        ({}, "2023-01-31T12:00"),
    ],
)
def test_disallows_reanalysis_unsupported_cycles_and_non_utc(weather, turbine, changes, run):
    with pytest.raises(ValueError):
        build_single_run_url(replace(weather, **changes), turbine, run)


def test_offline_replay_falls_back_only_to_eligible_run(weather, turbine, tmp_path):
    # Later initialized runs cannot be selected just because they are now cached.
    cache(tmp_path, turbine, "2026-01-31T12:00")
    cache(tmp_path, turbine)
    with patch("wind_power_forecast.weather.urlopen") as network:
        frame, meta = get_weather_for_forecast(
            weather, turbine, AS_OF, targets(), tmp_path, offline=True
        )
    network.assert_not_called()
    assert len(frame) == 48
    assert frame["timestamp"].iloc[0] == targets()[0]
    assert str(frame["timestamp"].dt.tz) == "UTC"
    assert meta["run"] == "2026-01-30T12:00"
    assert meta["assumed_available_at"] == "2026-01-30T20:00:00+00:00"
    assert meta["historical_publication_verified"] is False
    assert meta["fallback_used"] is True
    assert len(meta["attempts"]) == 3


@pytest.mark.parametrize(
    "damage", ["hash", "gap", "duplicate", "null", "infinity", "unit", "offset"]
)
def test_rejects_corrupt_incomplete_and_nonfinite_weather(weather, turbine, tmp_path, damage):
    body = payload("2026-01-30T12:00")
    if damage == "gap":
        for values in body["hourly"].values():
            values.pop(35)
    elif damage == "duplicate":
        body["hourly"]["time"][36] = body["hourly"]["time"][35]
    elif damage in ("null", "infinity"):
        body["hourly"]["wind_speed_100m"][35] = None if damage == "null" else float("inf")
    elif damage == "unit":
        body["hourly_units"]["wind_speed_100m"] = "km/h"
    elif damage == "offset":
        body["utc_offset_seconds"] = 18000
    path, envelope = cache(tmp_path, turbine, body=body)
    if damage == "hash":
        envelope["payload"]["hourly"]["wind_speed_100m"][35] = 10.0
        path.write_text(json.dumps(envelope), encoding="utf-8")
    with pytest.raises(WeatherUnavailableError):
        get_weather_for_forecast(weather, turbine, AS_OF, targets(), tmp_path, offline=True)


def test_missing_coverage_and_wrong_coordinates_are_not_used(weather, turbine, tmp_path):
    cache(tmp_path, replace(turbine, latitude=40.0))
    cache(tmp_path, turbine, body=payload("2026-01-30T12:00", periods=36))
    with pytest.raises(WeatherUnavailableError):
        get_weather_for_forecast(weather, turbine, AS_OF, targets(), tmp_path, offline=True)


def test_offline_target_time_guards(weather, turbine, tmp_path):
    for invalid in (
        pd.date_range("2026-02-01", periods=48, freq="h"),
        pd.DatetimeIndex([AS_OF]),
        targets().delete(5),
        pd.date_range("2026-02-01T00:15", periods=48, freq="h", tz="Asia/Almaty"),
    ):
        with pytest.raises(ValueError):
            get_weather_for_forecast(weather, turbine, AS_OF, invalid, tmp_path, offline=True)


def test_online_fetches_own_location_even_if_other_turbine_is_cached(weather, turbine, tmp_path):
    run = "2026-01-31T06:00"
    cache(tmp_path, replace(turbine, latitude=40.0), run)
    body = payload(run)
    with patch(
        "wind_power_forecast.weather.fetch_single_run_payload",
        return_value=(body, build_single_run_url(weather, turbine, run), 200),
    ) as fetch:
        frame, meta = get_weather_for_forecast(weather, turbine, AS_OF, targets(), tmp_path)
    fetch.assert_called_once()
    assert len(frame) == 48
    assert meta["run"] == run
    assert meta["fallback_used"] is False


def test_online_transient_failure_falls_back_to_older_valid_cache(weather, turbine, tmp_path):
    cache(tmp_path, turbine)
    with patch(
        "wind_power_forecast.weather.fetch_single_run_payload", side_effect=URLError("down")
    ):
        frame, meta = get_weather_for_forecast(weather, turbine, AS_OF, targets(), tmp_path)
    assert len(frame) == 48
    assert meta["fallback_used"]
    assert sum(a["action"] == "fetch_failed" for a in meta["attempts"]) == 3


def test_authorization_failure_does_not_repeat_for_all_runs(weather, turbine, tmp_path):
    cache(tmp_path, turbine)
    error = HTTPError(SINGLE_RUN_ENDPOINT, 403, "Forbidden", {}, io.BytesIO())
    with patch("wind_power_forecast.weather.fetch_single_run_payload", side_effect=error) as fetch:
        frame, meta = get_weather_for_forecast(
            weather, turbine, AS_OF, targets(), tmp_path, refresh=True
        )
    fetch.assert_called_once()
    assert len(frame) == 48
    assert meta["run"] == "2026-01-30T12:00"


def test_retry_only_transient_errors(weather, turbine):
    run = "2026-01-31T06:00"
    error = HTTPError(SINGLE_RUN_ENDPOINT, 503, "Busy", {}, io.BytesIO())
    with (
        patch("wind_power_forecast.weather.urlopen", side_effect=error) as fetch,
        pytest.raises(HTTPError),
    ):
        fetch_single_run_payload(weather, turbine, run)
    assert fetch.call_count == weather.retry_attempts
    error = HTTPError(SINGLE_RUN_ENDPOINT, 400, "Invalid run", {}, io.BytesIO())
    with (
        patch("wind_power_forecast.weather.urlopen", side_effect=error) as fetch,
        pytest.raises(HTTPError),
    ):
        fetch_single_run_payload(weather, turbine, run)
    fetch.assert_called_once()


def make_archive(tmp_path, turbine, corrupt=False):
    _, envelope = cache(tmp_path / "seed", turbine)
    connection = sqlite3.connect(":memory:")
    connection.execute(
        "CREATE TABLE weather_cache(request_json TEXT, payload_json TEXT, "
        "payload_sha256 TEXT, fetched_at TEXT)"
    )
    raw = json.dumps(envelope["payload"])
    digest = hashlib.sha256(raw.encode()).hexdigest()
    connection.execute(
        "INSERT INTO weather_cache VALUES(?,?,?,?)",
        (
            json.dumps(envelope["request"]),
            raw,
            "wrong" if corrupt else digest,
            envelope["fetched_at"],
        ),
    )
    connection.commit()
    blob = connection.serialize()
    connection.close()
    path = tmp_path / "submission.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("submission/data/cache/weather.sqlite", blob)
        archive.writestr("submission/../../should_not_extract.py", "raise RuntimeError('unsafe')")
    return path


def test_archive_import_checks_hashes_preserves_provenance_and_never_extracts(
    weather, turbine, tmp_path
):
    archive = make_archive(tmp_path, turbine)
    original = archive.read_bytes()
    output = tmp_path / "cache"
    report = import_weather_archive(archive, output)
    assert report["imported"] == 1
    assert archive.read_bytes() == original
    assert not (tmp_path / "should_not_extract.py").exists()
    _, meta = get_weather_for_forecast(weather, turbine, AS_OF, targets(), output, offline=True)
    assert meta["provenance"]["original_checksum_verified"] is True
    assert meta["provenance"]["historical_publication_verified"] is False
    assert meta["provenance"]["as_issued_authenticity_verified"] is False


def test_archive_import_rejects_modified_payload(turbine, tmp_path):
    archive = make_archive(tmp_path, turbine, corrupt=True)
    with pytest.raises(ValueError, match="SHA256"):
        import_weather_archive(archive, tmp_path / "cache")
