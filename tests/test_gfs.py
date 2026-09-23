import json
import re
import sys
import unittest
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch
from urllib.error import HTTPError

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wind_power_forecast import gfs
from wind_power_forecast.settings import TurbineSettings, WeatherSettings


def grib_metadata(field, run, lead):
    valid = run + timedelta(hours=lead)
    return {
        "edition": 2, "dataDate": int(run.strftime("%Y%m%d")), "dataTime": run.hour * 100,
        "forecastTime": lead, "stepUnits": 1, "stepType": "instant",
        "typeOfLevel": "heightAboveGround", "level": 2 if field == "t" else 100,
        "shortName": {"u": "100u", "v": "100v", "t": "2t"}[field],
        "units": "K" if field == "t" else "m s**-1", "centre": "kwbc",
        "typeOfGeneratingProcess": 2, "gridType": "regular_ll", "Ni": 1440, "Nj": 721,
        "iDirectionIncrementInDegrees": 0.25, "jDirectionIncrementInDegrees": 0.25,
        "latitudeOfFirstGridPointInDegrees": 90.0, "longitudeOfFirstGridPointInDegrees": 0.0,
        "validityDate": int(valid.strftime("%Y%m%d")), "validityTime": valid.hour * 100,
    }


def grib_blob(field):
    # A bounded synthetic GRIB2 envelope. Native decoding is mocked below;
    # this is deliberately not presented as a real meteorological data fixture.
    return b"GRIB\x00\x00\x00\x02" + (32).to_bytes(8, "big") + field.encode() * 12 + b"7777"


def inventory(run, lead):
    stamp = run.strftime("%Y%m%d%H")
    fields = [("UGRD", "100 m above ground"), ("VGRD", "100 m above ground"),
              ("TMP", "2 m above ground"), ("HGT", "surface")]
    return "\n".join(
        f"{i + 1}:{i * 32}:d={stamp}:{name}:{level}:{lead} hour fcst:"
        for i, (name, level) in enumerate(fields)
    ).encode("ascii")


class Response:
    def __init__(self, url, body, status=200, modified=None, content_range=None):
        self.url, self.body, self.status = url, body, status
        self.headers = {"Content-Length": str(len(body)), "ETag": '"fixture"'}
        if modified is not None:
            self.headers["Last-Modified"] = format_datetime(modified, usegmt=True)
        if content_range is not None:
            self.headers["Content-Range"] = content_range
        self.read_called = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def geturl(self):
        return self.url

    def read(self, size=-1):
        self.read_called = True
        return self.body if size == -1 else self.body[:size]


class OriginalGFSProviderTest(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.settings = WeatherSettings(
            endpoint=gfs.BUCKET_URL, model="gfs_0p25", forecast_hours=120,
            timezone="UTC", daily_calculation_hour_utc=18, model_latency_hours=8,
            wind_speed_variable="wind_speed_100m", temperature_variable="temperature_2m",
            hourly_variables=("wind_speed_100m", "temperature_2m"), retry_attempts=1,
            retry_backoff_seconds=0, fallback_runs=1,
        )
        self.turbine = TurbineSettings(
            id="turbine_1", latitude=43.645150, longitude=78.535604,
            data_csv=None, capacity_mw=None, model_source="self",
        )
        self.as_of = datetime(2026, 1, 31, 18, tzinfo=UTC)
        self.run = datetime(2026, 1, 31, 6, tzinfo=UTC)
        self.targets = pd.date_range("2026-01-31T19:00Z", periods=2, freq="h")

    def http_fixture(self, request, timeout):
        url = request.full_url
        match = re.search(r"gfs\.(\d{8})/(\d{2})/atmos/.*\.f(\d{3})", url)
        run = datetime.strptime(match[1] + match[2], "%Y%m%d%H").replace(tzinfo=UTC)
        lead = int(match[3])
        modified = run + timedelta(hours=3)
        if url.endswith(".idx"):
            return Response(url, inventory(run, lead), modified=modified)
        start, end = map(int, request.get_header("Range").removeprefix("bytes=").split("-"))
        field = {0: "u", 32: "v", 64: "t"}[start]
        return Response(url, grib_blob(field), status=206, modified=modified,
                        content_range=f"bytes {start}-{end}/128")

    def decoder_fixture(self, blob, field, run, lead, grid):
        return {
            "value": {"u": 3.0, "v": 4.0, "t": 280.0}[field],
            "grid_latitude": grid[0], "grid_longitude": grid[1],
            "grib_metadata": grib_metadata(field, run, lead),
        }

    def fetch(self, **kwargs):
        return gfs.get_weather_for_forecast(
            self.settings, self.turbine, self.as_of, self.targets, self.root, **kwargs
        )

    def test_cycle_selection_and_exact_original_product_url(self):
        self.assertEqual(gfs.eligible_runs(self.settings, self.as_of), [self.run])
        self.assertEqual(gfs.build_gfs_url(self.run, 13),
                         gfs.BUCKET_URL + "/gfs.20260131/06/atmos/gfs.t06z.pgrb2.0p25.f013")
        with self.assertRaisesRegex(ValueError, "1..120"):
            gfs.build_gfs_url(self.run, 0)

    def test_full_hourly_forecast_and_shared_grid_cache_for_both_turbines(self):
        self.targets = pd.date_range("2026-01-31T19:00Z", periods=48, freq="h")
        with patch.object(gfs, "urlopen", side_effect=self.http_fixture) as network, patch.object(
            gfs, "_decode_grib", side_effect=self.decoder_fixture
        ):
            frame, metadata = self.fetch()
            self.assertEqual(network.call_count, 48 * 4)
            self.assertEqual(len(frame), 48)
            self.assertEqual(frame["timestamp"].tolist(), self.targets.tolist())
            np.testing.assert_allclose(frame["wind_speed"], 5.0)
            np.testing.assert_allclose(frame["temperature"], 6.85)
            self.assertEqual(metadata["run"], "2026-01-31T06:00")
            self.assertTrue(metadata["historical_publication_verified"])
            self.assertTrue(metadata["as_issued_authenticity_verified"])
            self.assertEqual(metadata["grid_latitude"], 43.75)
            self.assertEqual(metadata["grid_longitude"], 78.5)
            self.turbine = replace(self.turbine, id="turbine_2", latitude=43.643198, longitude=78.538828)
            second, cached = self.fetch(offline=True)
            self.assertEqual(network.call_count, 48 * 4)
            pd.testing.assert_frame_equal(frame, second)
            self.assertEqual(metadata["payload_sha256"], cached["payload_sha256"])
            self.assertEqual(len(list((self.root / "gfs_points").glob("*.json"))), 48)

    def test_unchanged_refresh_has_stable_payload_hash(self):
        with patch.object(gfs, "urlopen", side_effect=self.http_fixture), patch.object(
            gfs, "_decode_grib", side_effect=self.decoder_fixture
        ):
            _, first = self.fetch()
            _, refreshed = self.fetch(refresh=True)
            self.assertEqual(first["payload_sha256"], refreshed["payload_sha256"])

    def test_later_copied_archive_object_is_rejected(self):
        def copied(request, timeout):
            response = self.http_fixture(request, timeout)
            response.headers["Last-Modified"] = format_datetime(self.as_of + timedelta(hours=1), usegmt=True)
            return response

        with patch.object(gfs, "urlopen", side_effect=copied), self.assertRaisesRegex(
            gfs.GFSUnavailableError, "Last-Modified"
        ):
            self.fetch()
        self.assertEqual(list(self.root.rglob("*.json")), [])

    def test_one_late_grib_field_is_rejected_even_when_index_is_old(self):
        def copied_field(request, timeout):
            response = self.http_fixture(request, timeout)
            if request.get_header("Range") == "bytes=32-63":
                response.headers["Last-Modified"] = format_datetime(self.as_of + timedelta(hours=1), usegmt=True)
            return response

        with patch.object(gfs, "urlopen", side_effect=copied_field), patch.object(
            gfs, "_decode_grib", side_effect=self.decoder_fixture
        ), self.assertRaisesRegex(gfs.GFSUnavailableError, "Last-Modified"):
            self.fetch()

    def test_grib_object_changed_between_field_requests_is_rejected(self):
        def mixed_object(request, timeout):
            response = self.http_fixture(request, timeout)
            if request.get_header("Range") == "bytes=32-63":
                response.headers["ETag"] = '"other-object-version"'
            return response

        with patch.object(gfs, "urlopen", side_effect=mixed_object), patch.object(
            gfs, "_decode_grib", side_effect=self.decoder_fixture
        ), self.assertRaisesRegex(gfs.GFSUnavailableError, "different versions"):
            self.fetch()

    def test_missing_latest_run_falls_back_as_a_whole_run(self):
        self.settings = replace(self.settings, fallback_runs=2)

        def missing_latest(request, timeout):
            if "/06/" in request.full_url:
                raise HTTPError(request.full_url, 404, "No run", {}, None)
            return self.http_fixture(request, timeout)

        with patch.object(gfs, "urlopen", side_effect=missing_latest), patch.object(
            gfs, "_decode_grib", side_effect=self.decoder_fixture
        ):
            frame, metadata = self.fetch()
            self.assertEqual(len(frame), 2)
            self.assertEqual(metadata["run"], "2026-01-31T00:00")
            self.assertTrue(metadata["fallback_used"])
            for path in metadata["cache_files"]:
                point = json.loads(Path(path).read_text())
                self.assertEqual(point["payload"]["run"], "2026-01-31T00:00:00+00:00")

    def test_offline_cache_checks_integrity_and_historical_availability_again(self):
        with patch.object(gfs, "urlopen", side_effect=self.http_fixture), patch.object(
            gfs, "_decode_grib", side_effect=self.decoder_fixture
        ):
            self.fetch()
        path = next((self.root / "gfs_points").glob("*.json"))
        envelope = json.loads(path.read_text())
        envelope["payload"]["fields"]["t"]["value"] = 350
        path.write_text(json.dumps(envelope))
        with patch.object(gfs, "urlopen") as network, self.assertRaisesRegex(
            gfs.GFSUnavailableError, "checksum"
        ):
            self.fetch(offline=True)
        network.assert_not_called()
        envelope["payload"]["fields"]["u"]["source"]["last_modified"] = "2026-02-01T00:00:00+00:00"
        envelope["payload_sha256"] = gfs._digest(envelope["payload"])
        path.write_text(json.dumps(envelope))
        with self.assertRaisesRegex(gfs.GFSUnavailableError, "Last-Modified"):
            self.fetch(offline=True)

    def test_range_ignored_by_server_is_rejected_without_reading_full_grib(self):
        url = gfs.build_gfs_url(self.run, 13)
        response = Response(url, b"large-full-file", status=200, modified=self.run)
        with patch.object(gfs, "urlopen", return_value=response), self.assertRaisesRegex(
            ValueError, "Expected HTTP 206"
        ):
            gfs._http_bytes(url, self.settings, limit=gfs.MAX_GRIB_BYTES, byte_range=(0, 31))
        self.assertFalse(response.read_called)

    def test_response_limits_and_wrong_range_are_enforced_before_read(self):
        url = gfs.build_gfs_url(self.run, 13)
        response = Response(url, b"tiny", status=206, modified=self.run, content_range="bytes 10-13/128")
        with patch.object(gfs, "urlopen", return_value=response), self.assertRaisesRegex(
            ValueError, "requested bytes"
        ):
            gfs._http_bytes(url, self.settings, limit=gfs.MAX_GRIB_BYTES, byte_range=(0, 3))
        self.assertFalse(response.read_called)
        response = Response(url + ".idx", b"tiny", modified=self.run)
        response.headers["Content-Length"] = str(gfs.MAX_INDEX_BYTES + 1)
        with patch.object(gfs, "urlopen", return_value=response), self.assertRaisesRegex(
            ValueError, "download limit"
        ):
            gfs._http_bytes(url + ".idx", self.settings, limit=gfs.MAX_INDEX_BYTES, byte_range=None)
        self.assertFalse(response.read_called)

    def test_inventory_wrong_run_lead_missing_variable_and_oversized_fields_fail(self):
        good = inventory(self.run, 13)
        bad = [
            good.replace(b"d=2026013106", b"d=2026013100"),
            good.replace(b"13 hour fcst", b"12 hour fcst"),
            good.replace(b"100 m above ground", b"10 m above ground"),
            good.replace(b"2:32:", b"2:5000001:"),
        ]
        for blob in bad:
            with self.subTest(index=blob), self.assertRaises(ValueError):
                gfs._index_ranges(blob, self.run, 13)

    def test_decoder_checks_original_run_lead_variable_level_and_units(self):
        metadata = grib_metadata("u", self.run, 13)
        fake = SimpleNamespace(
            codes_new_from_message=Mock(return_value=1),
            codes_get=lambda handle, key: metadata[key] if key != "missingValue" else 9999,
            codes_grib_find_nearest=Mock(return_value=[{"lat": 43.75, "lon": 78.5, "value": 3.0}]),
            codes_release=Mock(),
        )
        with patch.dict(sys.modules, {"eccodes": fake}):
            decoded = gfs._decode_grib(grib_blob("u"), "u", self.run, 13, (43.75, 78.5))
            self.assertEqual(decoded["value"], 3.0)
            for key, incorrect in (
                ("dataDate", 20260201), ("dataTime", 0), ("forecastTime", 12),
                ("shortName", "v"), ("level", 10), ("units", "K"),
                ("stepType", "avg"), ("typeOfGeneratingProcess", 0),
            ):
                original = metadata[key]
                metadata[key] = incorrect
                with self.subTest(key=key), self.assertRaises(ValueError):
                    gfs._decode_grib(grib_blob("u"), "u", self.run, 13, (43.75, 78.5))
                metadata[key] = original
            self.assertEqual(fake.codes_release.call_count, 9)

    def test_naive_gapped_or_nonfuture_targets_fail_without_network(self):
        invalid = [pd.date_range("2026-02-01", periods=2, freq="h"),
                   pd.DatetimeIndex([self.targets[0], self.targets[0] + timedelta(hours=2)]),
                   pd.DatetimeIndex([self.as_of])]
        with patch.object(gfs, "urlopen") as network:
            for targets in invalid:
                self.targets = targets
                with self.assertRaises(ValueError):
                    self.fetch()
        network.assert_not_called()


if __name__ == "__main__":
    unittest.main()
