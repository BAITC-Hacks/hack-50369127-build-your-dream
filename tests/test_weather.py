"""No-network tests of temporal boundaries, provenance, and corruption handling."""

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlparse

from windagent.weather import WeatherError, _download, fetch_weather, select_run

UTC = timezone.utc
ISSUE = datetime(2026, 1, 31, 18, tzinfo=UTC)
TARGETS = [{"id": "T1", "latitude": 43.6451388889, "longitude": 78.5356111111}]


def fixture(hours=48):
    return {
        "latitude": 43.65, "longitude": 78.54, "utc_offset_seconds": 0,
        "hourly_units": {"time": "iso8601", "wind_speed_100m": "m/s", "temperature_2m": "°C"},
        "hourly": {
            "time": [(ISSUE + timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M") for i in range(1, hours + 1)],
            "wind_speed_100m": [7.5] * hours,
            "temperature_2m": [-8.0] * hours,
        },
    }


class WeatherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.cache = Path(self.temp.name)

    def fetch(self, payload=None, **overrides):
        args = {"as_of": ISSUE, "targets": TARGETS, "hours": 48, "cache_dir": self.cache}
        args.update(overrides)
        raw = json.dumps(fixture() if payload is None else payload).encode("utf-8")
        with patch("windagent.weather._download", return_value=raw) as download:
            result = fetch_weather(**args)
        return result, download

    def test_select_run_enforces_available_boundary_and_utc_conversion(self):
        self.assertEqual(select_run(ISSUE), datetime(2026, 1, 31, 6, tzinfo=UTC))
        local = ISSUE.astimezone(timezone(timedelta(hours=5)))
        self.assertEqual(select_run(local), select_run(ISSUE))
        self.assertEqual(select_run(ISSUE - timedelta(hours=1)), datetime(2026, 1, 31, 0, tzinfo=UTC))
        self.assertEqual(select_run(ISSUE, 12.5), datetime(2026, 1, 31, 0, tzinfo=UTC))

    def test_naive_fractional_and_pre_archive_times_fail(self):
        for stamp in (ISSUE.replace(tzinfo=None), ISSUE.replace(minute=1), datetime(2024, 3, 14, tzinfo=UTC)):
            with self.subTest(stamp=stamp), self.assertRaises(WeatherError):
                select_run(stamp)
        for lag in (-1, float("nan"), float("inf"), True, 169):
            with self.subTest(lag=lag), self.assertRaises(WeatherError):
                select_run(ISSUE, lag)

    def test_exact_run_units_hour_window_and_uncertified_provenance(self):
        result, download = self.fetch()
        self.assertEqual(len(result["rows"]), 48)
        self.assertEqual(result["rows"][0]["valid_time"], "2026-01-31T19:00:00Z")
        self.assertEqual(result["rows"][-1]["valid_time"], "2026-02-02T18:00:00Z")
        provenance = result["provenance"][0]
        query = parse_qs(urlparse(download.call_args.args[0]).query)
        self.assertEqual(query["run"], ["2026-01-31T06:00"])
        self.assertEqual(query["models"], ["ecmwf_ifs"])
        self.assertEqual(query["wind_speed_unit"], ["ms"])
        self.assertEqual(query["start_date"], ["2026-01-31"])
        self.assertEqual(query["end_date"], ["2026-02-02"])
        self.assertFalse(provenance["competition_ready"])
        self.assertFalse(provenance["availability_verified"])
        self.assertFalse(provenance["operational_run_verified"])
        self.assertIn("assumed", provenance["availability_basis"])
        self.assertEqual(provenance["available_at"], "2026-01-31T18:00:00Z")
        self.assertEqual(len(provenance["sha256"]), 64)
        self.assertGreaterEqual(len(result["warnings"]), 3)

    def test_cache_reuses_exact_bytes_and_refresh_retrieves_again(self):
        first, _ = self.fetch()
        first_snapshot = Path(first["provenance"][0]["raw_path"])
        first_bytes = first_snapshot.read_bytes()
        second, download = self.fetch()
        download.assert_not_called()
        self.assertTrue(second["provenance"][0]["cache_hit"])
        self.assertEqual(first["provenance"][0]["sha256"], second["provenance"][0]["sha256"])
        self.assertEqual(first["provenance"][0]["retrieved_at"], second["provenance"][0]["retrieved_at"])
        changed = fixture()
        changed["hourly"]["wind_speed_100m"][0] = 9.0
        refreshed, download = self.fetch(changed, refresh=True)
        download.assert_called_once()
        self.assertNotEqual(first["provenance"][0]["sha256"], refreshed["provenance"][0]["sha256"])
        self.assertEqual(refreshed["rows"][0]["wind_speed"], 9.0)
        self.assertEqual(first_snapshot.read_bytes(), first_bytes)
        self.assertTrue(Path(refreshed["provenance"][0]["raw_path"]).exists())
        self.assertNotEqual(first["provenance"][0]["raw_path"], refreshed["provenance"][0]["raw_path"])

    def test_cache_corruption_and_partial_cache_fail_closed(self):
        self.fetch()
        raw_path = next(path for path in self.cache.glob("*.json") if not path.name.endswith(".meta.json"))
        raw_path.write_bytes(b"{}")
        with self.assertRaisesRegex(WeatherError, "integrity"):
            self.fetch()
        raw_path.unlink()
        with self.assertRaisesRegex(WeatherError, "Incomplete"):
            self.fetch()

    def test_missing_null_invalid_units_duplicates_and_offset_fail(self):
        variants = []
        payload = fixture(); payload["hourly"]["wind_speed_100m"][10] = None; variants.append(payload)
        payload = fixture(); payload["hourly"]["wind_speed_100m"][10] = float("nan"); variants.append(payload)
        payload = fixture(); payload["hourly"]["temperature_2m"][10] = True; variants.append(payload)
        payload = fixture(); payload["hourly_units"]["wind_speed_100m"] = "km/h"; variants.append(payload)
        payload = fixture(); payload["utc_offset_seconds"] = 18000; variants.append(payload)
        payload = fixture(); payload["hourly"]["time"][1] = payload["hourly"]["time"][0]; variants.append(payload)
        payload = fixture(); payload["hourly"]["time"][1] = "2026-01-01T00:00"; variants.append(payload)
        payload = fixture(); payload["hourly"]["temperature_2m"].pop(); variants.append(payload)
        for index, payload in enumerate(variants):
            with self.subTest(index=index), self.assertRaises(WeatherError):
                self.fetch(payload, refresh=True)

    def test_validates_inputs_before_request(self):
        variants = [{"targets": []}, {"targets": TARGETS * 2}, {"hours": 0}, {"hours": 49},
                    {"targets": [{"id": "T1", "latitude": float("nan"), "longitude": 78.5}]}]
        for args in variants:
            with self.subTest(args=args), self.assertRaises(WeatherError):
                self.fetch(**args)

    def test_two_turbines_have_distinct_audit_records(self):
        other = {"id": "T2", "latitude": 43.6431944444, "longitude": 78.5388333333}
        result, download = self.fetch(targets=TARGETS + [other])
        self.assertEqual(len(result["rows"]), 96)
        self.assertEqual(download.call_count, 2)
        self.assertEqual({row["turbine_id"] for row in result["rows"]}, {"T1", "T2"})
        self.assertNotEqual(result["provenance"][0]["url"], result["provenance"][1]["url"])

    def test_no_fallback_and_bounded_network_retries(self):
        with patch("windagent.weather.urlopen", side_effect=URLError("offline")) as request, patch("windagent.weather.time.sleep"):
            with self.assertRaises(WeatherError):
                _download("https://single-runs-api.open-meteo.com/v1/forecast?run=2026-01-31T06:00")
            self.assertEqual(request.call_count, 3)
        error = HTTPError("https://example.invalid", 400, "invalid run", {}, None)
        with patch("windagent.weather.urlopen", side_effect=error) as request:
            with self.assertRaisesRegex(WeatherError, "HTTP 400"):
                _download("https://single-runs-api.open-meteo.com/v1/forecast?run=2026-01-31T06:00")
            request.assert_called_once()


if __name__ == "__main__":
    unittest.main()
