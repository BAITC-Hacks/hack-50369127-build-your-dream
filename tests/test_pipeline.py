from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sqlite3
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
import unittest
from unittest.mock import patch
from typing import Any

import numpy as np
import pandas as pd
import requests

from src.data.build_features import FEATURE_COLUMNS, join_labels, build
from src.data.fetch_weather import WeatherClient, UNITS, VARIABLES, parse_response, select_run
from src.data.preprocess_telemetry import preprocess_telemetry
from src.utils.common import load_config, utc


class FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self.payload


class FakeSession:
    """Synthetic HTTP test double; never used by the application."""
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.calls = 0

    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        self.calls += 1
        return FakeResponse(self.payload)


def weather_payload(times: pd.DatetimeIndex) -> dict[str, Any]:
    values = [5.0, 7.0, 359.0, 1.0, 10.0, 1000.0]
    return {"utc_offset_seconds": 0, "latitude": 43.65, "longitude": 78.54,
            "hourly_units": UNITS,
            "hourly": {"time": times.strftime("%Y-%m-%dT%H:%M").tolist(),
                       **{name: [value] * len(times) for name, value in zip(VARIABLES, values)}}}


class PipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.cfg = load_config(Path(__file__).resolve().parents[1] / "config/config.yaml")
        self.cfg["_root"] = self.root
        self.times = pd.date_range("2026-02-01", periods=48, freq="h", tz="UTC")
        self.issue = utc("2026-01-31T23:00:00Z")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def telemetry_file(self, rows: int = 6) -> Path:
        frame = pd.DataFrame({
            "time": pd.date_range("2026-01-31", periods=rows, freq="10min"),
            "wind_speed_ms": 5.0, "power_norm": np.linspace(0, 1, rows), "temperature_c": 10.0})
        frame = frame.rename(columns=self.cfg["telemetry"]["columns"])
        path = self.root / "test.csv"
        frame.to_csv(path, index=False)
        return path

    def test_full_hour_mean_and_density(self) -> None:
        frame, report = preprocess_telemetry(self.telemetry_file(), "T1", self.cfg["telemetry"])
        self.assertAlmostEqual(frame.loc[0, "power_norm"], 0.5)
        self.assertEqual(report["hours_complete"], 1)
        weather = parse_response(weather_payload(self.times), self.times, VARIABLES)
        self.assertAlmostEqual(weather.loc[0, "air_density"], 100000 / (287.05 * 283.15))

    def test_missing_slot_does_not_create_full_hour_label(self) -> None:
        frame, _ = preprocess_telemetry(self.telemetry_file(5), "T1", self.cfg["telemetry"])
        self.assertTrue(pd.isna(frame.loc[0, "power_norm"]))
        self.assertAlmostEqual(frame.loc[0, "coverage_power"], 5 / 6)

    def test_outlier_is_masked_not_clipped(self) -> None:
        path = self.telemetry_file()
        frame = pd.read_csv(path)
        frame.loc[0, self.cfg["telemetry"]["columns"]["power_norm"]] = 1.5
        frame.to_csv(path, index=False)
        hourly, report = preprocess_telemetry(path, "T1", self.cfg["telemetry"])
        self.assertTrue(pd.isna(hourly.loc[0, "power_norm"]))
        self.assertEqual(report["invalid_measurements"]["power_norm"], 1)

    def test_conflicting_duplicate_rejected(self) -> None:
        path = self.telemetry_file()
        frame = pd.read_csv(path)
        duplicate = frame.iloc[[0]].copy()
        duplicate.iloc[0, 2] = 0.7
        pd.concat([frame, duplicate]).to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "conflicting"):
            preprocess_telemetry(path, "T1", self.cfg["telemetry"])

    def test_timezone_and_interval_end(self) -> None:
        cfg = deepcopy(self.cfg["telemetry"])
        cfg["source_timezone"] = "Etc/GMT-5"
        frame, _ = preprocess_telemetry(self.telemetry_file(), "T1", cfg)
        self.assertEqual(frame.loc[0, "valid_time"], utc("2026-01-30T19:00:00Z"))
        cfg["timestamp_convention"] = "interval_end"
        shifted, _ = preprocess_telemetry(self.telemetry_file(), "T1", cfg)
        self.assertEqual(shifted.loc[0, "valid_time"], utc("2026-01-30T18:00:00Z"))

    def test_release_delay_and_strict_mode(self) -> None:
        run, available, basis = select_run(self.issue, self.cfg)
        self.assertEqual(run, utc("2026-01-31T12:00:00Z"))
        self.assertLessEqual(available, self.issue)
        self.assertEqual(basis, "assumed_delay")
        self.cfg["weather"]["require_verified_availability"] = True
        with self.assertRaisesRegex(ValueError, "Verified"):
            select_run(self.issue, self.cfg)

    def test_manifest_selects_only_published_run(self) -> None:
        pd.DataFrame({"model": ["ecmwf_ifs"] * 2,
                      "run_time": ["2026-01-31T12:00Z", "2026-01-31T18:00Z"],
                      "available_at": ["2026-01-31T20:00Z", "2026-02-01T01:00Z"],
                      "source": ["test-evidence", "test-evidence"]}).to_csv(self.root / "availability.csv", index=False)
        self.cfg["weather"].update(availability_policy="manifest", availability_csv="availability.csv",
                                   require_verified_availability=True)
        run, _, basis = select_run(self.issue, self.cfg)
        self.assertEqual(run.hour, 12)
        self.assertTrue(basis.startswith("manifest:"))

    def test_cache_reuse_and_integrity(self) -> None:
        session = FakeSession(weather_payload(self.times))
        client = WeatherClient(self.cfg, session=session)
        a = client.fetch(self.cfg["turbines"][0], self.issue, self.times)
        b = client.fetch(self.cfg["turbines"][0], self.issue, self.times)
        self.assertEqual(session.calls, 1)
        pd.testing.assert_frame_equal(a, b)
        offline = WeatherClient(self.cfg, offline=True)
        pd.testing.assert_frame_equal(a, offline.fetch(self.cfg["turbines"][0], self.issue, self.times))
        offline.close()
        with sqlite3.connect(client.cache) as db:
            db.execute("UPDATE weather_cache SET payload_json='{}'")
        with self.assertRaisesRegex(ValueError, "Corrupt"):
            client.fetch(self.cfg["turbines"][0], self.issue, self.times)

    def test_cache_miss_offline(self) -> None:
        client = WeatherClient(self.cfg, offline=True)
        try:
            with self.assertRaises(FileNotFoundError):
                client.fetch(self.cfg["turbines"][0], self.issue, self.times)
        finally:
            client.close()

    def test_missing_weather_and_wrong_units_rejected(self) -> None:
        payload = weather_payload(self.times)
        payload["hourly"]["wind_speed_100m"][0] = None
        with self.assertRaisesRegex(ValueError, "missing hours"):
            parse_response(payload, self.times, VARIABLES)
        payload = deepcopy(weather_payload(self.times))
        payload["hourly_units"]["wind_speed_10m"] = "km/h"
        with self.assertRaisesRegex(ValueError, "unit"):
            parse_response(payload, self.times, VARIABLES)

    def test_future_labels_and_observations_are_excluded(self) -> None:
        client = WeatherClient(self.cfg, session=FakeSession(weather_payload(self.times)))
        weather = client.fetch(self.cfg["turbines"][0], self.issue, self.times)
        telemetry = pd.DataFrame({"turbine_id": "T1", "valid_time": self.times,
                                  "power_norm": 0.7, "coverage_power": 1.0})
        features = join_labels(weather, telemetry, self.cfg)
        self.assertTrue(features["target_power_norm"].isna().all())
        self.assertNotIn("power_norm", FEATURE_COLUMNS)
        self.assertNotIn("wind_speed_ms", FEATURE_COLUMNS)
        self.assertLess(abs(features.loc[0, "wind_direction_100m_sin"] -
                            features.loc[0, "wind_direction_10m_sin"]), 0.04)
        weather["available_at"] = self.issue + pd.Timedelta(hours=1)
        with self.assertRaisesRegex(ValueError, "unavailable"):
            join_labels(weather, telemetry, self.cfg)

    def test_end_to_end_two_turbines_and_offline_replay(self) -> None:
        path = self.telemetry_file()
        for turbine in self.cfg["turbines"]:
            turbine["telemetry_file"] = str(path)
        self.cfg["dataset"]["issue_end_date"] = "2026-01-31"
        session = FakeSession(weather_payload(self.times))
        client = WeatherClient(self.cfg, session=session)
        with patch("src.data.build_features.WeatherClient", return_value=client):
            manifest = build(self.cfg)
        self.assertEqual(manifest["rows"], 96)
        self.assertEqual(manifest["labels_present"], 0)
        output = self.root / manifest["dataset"]
        content = output.read_bytes()
        offline_manifest = build(self.cfg, offline=True)
        self.assertEqual(offline_manifest["rows"], 96)
        self.assertEqual(content, output.read_bytes())
        self.assertEqual(session.calls, 2)

    def test_retry_on_transient_http_error(self) -> None:
        class Handler(BaseHTTPRequestHandler):
            calls = 0

            def do_GET(self) -> None:
                Handler.calls += 1
                self.send_response(503 if Handler.calls == 1 else 200)
                self.end_headers()
                self.wfile.write(b"ok")

            def log_message(self, format: str, *args: Any) -> None:
                return None

        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        client = WeatherClient(self.cfg)
        try:
            client.session.mount("http://", client.session.get_adapter("https://"))
            response = client.session.get(f"http://127.0.0.1:{server.server_port}", timeout=5)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(Handler.calls, 2)
        finally:
            client.close()
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == "__main__":
    unittest.main()
