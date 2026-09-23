"""Offline project-adapter tests; CatBoost is not a test dependency."""
from __future__ import annotations

import contextlib
import copy
import csv
from datetime import timedelta
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import subprocess
import tempfile
import types
import unittest
from unittest.mock import patch
import zipfile

from windagent.agent import ForecastAgent
from windagent.common import digest, iso, load_config, read_json, save_json, utc
from windagent.project import (ASSETS, ENDPOINT, FEATURE_NAMES, VARIABLES,
                               ProjectEngine, audit_cache, audit_january,
                               install_archive, make_features, sha_file)
from windagent.telemetry import history_to_csv


def json_text(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def csv_text(fields, rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


class ProjectFixtureTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.folder = self.base / "project"
        self.targets = load_config()["turbines"]
        self.issue = utc("2026-01-31T23:00:00Z")
        self.run = utc("2026-01-31T12:00:00Z")
        for asset in ASSETS:
            (self.folder / asset).parent.mkdir(parents=True, exist_ok=True)
        self.database = self.folder / "data/cache/weather.sqlite"
        self._build_cache()
        (self.folder / "artifacts/catboost.cbm").write_bytes(b"native-model-fixture-not-executed")
        (self.folder / "config/config.yaml").write_text("timezone: UTC\n", encoding="utf-8")
        raw_header = "ID,Статистическое время,Средняя скорость ветра(m/s),Нормализованная активная мощность,Средняя температура окружающей среды(°C)\n"
        for turbine in ("1", "2"):
            (self.folder / f"data/raw/turbine_{turbine}.csv").write_text(
                raw_header + "".join(f"{i + 1},2026-01-31 21:{i * 10:02d}:00,6,0.{turbine},-10\n" for i in range(6)),
                encoding="utf-8")
        self.january_fields = ["turbine_id", "issue_time", "valid_time", "lead_hours", "target_power_norm", "prediction"]
        self.january = [
            {"turbine_id": "T1", "issue_time": "2025-12-31T23:00:00Z", "valid_time": "2026-01-01T00:00:00Z", "lead_hours": 1, "target_power_norm": .3, "prediction": .4},
            {"turbine_id": "T2", "issue_time": "2025-12-31T23:00:00Z", "valid_time": "2026-01-01T00:00:00Z", "lead_hours": 1, "target_power_norm": .2, "prediction": .1},
        ]
        self.report = {
            "final_training_asof": iso(self.issue), "rows": {"final_fit": 2},
            "calibration_january": {f"T{t}/{bucket}": .1 for t in (1, 2) for bucket in ("01-24", "25-48")},
            "january_metrics": [
                {"model": "selected_model", "turbine_id": turbine, "horizon": horizon,
                 "n": n, "mae": .1, "rmse": .1, "bias": bias}
                for turbine, horizon, n, bias in (("ALL", "ALL", 2, 0), ("T1", "01-24", 1, .1), ("T2", "01-24", 1, -.1))
            ],
        }
        save_json(self.folder / "reports/training_report.json", self.report)
        self._write_january(self.january)
        self.info = {"folder": str(self.folder), "archive_sha256": "fixture-archive-version-A",
                     "training_cutoff": iso(self.issue), "asset_hashes": {}, "weather_audit": audit_cache(self.database)}
        for asset in ("artifacts/catboost.cbm", "reports/training_report.json"):
            self.info["asset_hashes"][asset] = sha_file(self.folder / asset)
        weather = ProjectEngine(self.info, self.targets).weather(self.issue, 48)
        self.saved_fields = ["turbine_id", "issue_time", "valid_time", "run_time", "weather_sha256", "power_norm", "lower_80", "upper_80"]
        self.saved_rows = []
        for row in weather["rows"]:
            provenance = next(p for p in weather["provenance"] if p["turbine_id"] == row["turbine_id"])
            point = .2 if row["turbine_id"] == "1" else .4
            self.saved_rows.append({"turbine_id": "T" + row["turbine_id"], "issue_time": iso(self.issue),
                "valid_time": row["valid_time"], "run_time": row["run_time"], "weather_sha256": provenance["sha256"],
                "power_norm": point, "lower_80": point - .1, "upper_80": point + .1})
        saved_text = csv_text(self.saved_fields, self.saved_rows)
        for asset in ("outputs/forecast_all_issues.csv", "outputs/submission_february.csv"):
            (self.folder / asset).write_text(saved_text, encoding="utf-8", newline="")
        self.info["asset_hashes"] = {asset: sha_file(self.folder / asset) for asset in ASSETS}

    def _build_cache(self, run=None):
        run = run or self.run
        with contextlib.closing(sqlite3.connect(self.database)) as database:
            database.execute("CREATE TABLE IF NOT EXISTS weather_cache(cache_key TEXT PRIMARY KEY, request_json TEXT, payload_json TEXT, payload_sha256 TEXT, fetched_at TEXT)")
            database.execute("DELETE FROM weather_cache")
            for target in self.targets:
                request = {"endpoint": ENDPOINT, "params": {"models": "ecmwf_ifs", "run": run.strftime("%Y-%m-%dT%H"),
                    "latitude": target["latitude"], "longitude": target["longitude"], "hourly": ",".join(VARIABLES),
                    "timezone": "UTC", "wind_speed_unit": "ms"}}
                times = [self.issue - timedelta(hours=24) + timedelta(hours=index) for index in range(121)]
                hourly = {"time": [at.strftime("%Y-%m-%dT%H:%M") for at in times]}
                hourly.update({name: [value] * len(times) for name, value in {
                    "wind_speed_10m": 5.0, "wind_direction_10m": 180.0, "wind_direction_100m": 90.0,
                    "temperature_2m": -10.0, "surface_pressure": 900.0}.items()})
                hourly["wind_speed_100m"] = [7 + int(target["id"]) + index / 100 for index in range(len(times))]
                payload = {"utc_offset_seconds": 0, "latitude": target["latitude"], "longitude": target["longitude"],
                    "hourly": hourly, "hourly_units": dict(zip(VARIABLES, ("m/s", "m/s", "°", "°", "°C", "hPa")))}
                request_string, body = json_text(request), json_text(payload)
                database.execute("INSERT INTO weather_cache VALUES (?,?,?,?,?)", (
                    hashlib.sha256(request_string.encode()).hexdigest(), request_string, body,
                    hashlib.sha256(body.encode()).hexdigest(), "2026-09-23T00:00:00Z"))
            database.commit()

    def _write_january(self, rows):
        (self.folder / "reports/january_predictions.csv").write_text(csv_text(self.january_fields, rows), encoding="utf-8", newline="")

    def _replace_payload(self, transform, update_hash=True, refresh_index=False):
        with contextlib.closing(sqlite3.connect(self.database)) as database:
            key, body = database.execute("SELECT cache_key,payload_json FROM weather_cache ORDER BY cache_key LIMIT 1").fetchone()
            payload = json.loads(body)
            transform(payload)
            changed = json_text(payload)
            if update_hash:
                database.execute("UPDATE weather_cache SET payload_json=?,payload_sha256=? WHERE cache_key=?",
                                 (changed, hashlib.sha256(changed.encode()).hexdigest(), key))
            else:
                database.execute("UPDATE weather_cache SET payload_json=? WHERE cache_key=?", (changed, key))
            database.commit()
        if refresh_index:
            self.info["weather_audit"] = audit_cache(self.database)

    def _engine(self):
        return ProjectEngine(self.info, self.targets)

    def _agent(self):
        config = {**load_config(), "training_cutoff": iso(self.issue), "timezone_offset_hours": 0}
        agent = ForecastAgent(self.base / "runtime", config)
        source = self.base / "canonical.csv"
        text = history_to_csv([{"turbine_id": target["id"], "timestamp": self.issue - timedelta(hours=2),
            "available_at": self.issue - timedelta(hours=1), "sample_count": 6, "wind_speed": 6,
            "temperature": -10, "power": .1 * int(target["id"])} for target in self.targets])
        source.write_text(text, encoding="utf-8", newline="")
        agent.state.update(project=self.info, dataset={"kind": "project_archive", "demo": False,
                           "path": str(source), "sha256": digest(text)}, model={"kind": "catboost_archive"})
        return agent

    def _archive(self, corrupt=None, traversal=False):
        archive = self.base / "source.zip"
        content = {asset: (self.folder / asset).read_bytes() for asset in ASSETS}
        content["src/untrusted_script.py"] = b'raise RuntimeError("Archive code must never execute")\n'
        manifest = "\n".join(hashlib.sha256(body).hexdigest() + "  ./" + name for name, body in content.items())
        with zipfile.ZipFile(archive, "w") as output:
            output.writestr("wind-forecast/SHA256SUMS.txt", manifest)
            for name, body in content.items():
                output.writestr("wind-forecast/" + name, body + b"changed" if name == corrupt else body)
            if traversal:
                output.writestr("../outside.txt", b"not allowed")
        return archive

    def test_archive_verifies_manifest_and_extracts_only_data_assets(self):
        result = install_archive(self._archive(), self.base / "installed")
        folder = Path(result["folder"])
        self.assertEqual(result["manifest_files_verified"], len(ASSETS) + 1)
        self.assertTrue(all((folder / asset).is_file() for asset in ASSETS))
        self.assertFalse((folder / "src/untrusted_script.py").exists())
        self.assertTrue(result["january_audit"]["arithmetic_verified"])
        self.assertFalse(result["january_audit"]["independent_retraining_verified"])
        self.assertFalse(result["weather_audit"]["availability_verified"])

    def test_archive_rejects_corruption_and_path_traversal_before_extracting(self):
        with self.assertRaises(ValueError):
            install_archive(self._archive(corrupt="artifacts/catboost.cbm"), self.base / "bad-checksum")
        self.assertFalse((self.base / "bad-checksum").exists())
        with self.assertRaises(ValueError):
            install_archive(self._archive(traversal=True), self.base / "bad-path")
        self.assertFalse((self.base / "outside.txt").exists())

    def test_cache_hashes_detect_payload_changes_and_checked_index_staleness(self):
        self.assertEqual(audit_cache(self.database)["records"], 2)
        self._replace_payload(lambda p: p["hourly"]["wind_speed_100m"].__setitem__(25, 99), update_hash=False)
        with self.assertRaises(ValueError):
            audit_cache(self.database)
        self._build_cache()
        self._replace_payload(lambda p: p["hourly"]["wind_speed_100m"].__setitem__(25, 99), update_hash=True)
        self.assertEqual(audit_cache(self.database)["records"], 2)
        with self.assertRaises(ValueError):
            self._engine().weather(self.issue, 24)

    def test_tampered_index_cannot_relabel_future_run_as_available(self):
        self._build_cache(run=utc("2026-01-31T18:00:00Z"))
        self.info["weather_audit"] = audit_cache(self.database)
        for entry in self.info["weather_audit"]["index"]:
            entry["request"]["params"]["run"] = "2026-01-31T12"
        with self.assertRaises(ValueError):
            self._engine().weather(self.issue, 24)

    def test_weather_selection_respects_exact_assumed_publication_boundary(self):
        engine = self._engine()
        at_publication = self.run + timedelta(hours=8)
        result = engine.weather(at_publication, 24)
        self.assertEqual(len(result["rows"]), 48)
        for record in result["provenance"]:
            self.assertEqual(utc(record["available_at"]), at_publication)
            self.assertFalse(record["availability_verified"])
        self.assertTrue(all(utc(row["valid_time"]) > at_publication for row in result["rows"]))
        with self.assertRaises(ValueError):
            engine.weather(at_publication - timedelta(hours=1), 24)

    def test_weather_rejects_bad_units_duplicates_missing_hours_and_nonfinite_values(self):
        def wrong_units(payload):
            payload["hourly_units"]["wind_speed_100m"] = "km/h"
        def duplicate(payload):
            payload["hourly"]["time"][25] = payload["hourly"]["time"][26]
        def missing(payload):
            for values in payload["hourly"].values():
                values.pop(25)
        def nonfinite(payload):
            payload["hourly"]["wind_speed_100m"][25] = float("nan")
        for transform in (wrong_units, duplicate, missing, nonfinite):
            with self.subTest(problem=transform.__name__):
                self._build_cache()
                self._replace_payload(transform, refresh_index=True)
                with self.assertRaises(ValueError):
                    self._engine().weather(self.issue, 24)

    def test_features_are_forecast_only_and_turbine_histories_are_independent(self):
        weather = self._engine().weather(self.issue, 24)["rows"]
        rows = [row for row in weather if utc(row["valid_time"]) <= self.issue + timedelta(hours=3)]
        first = make_features(list(reversed(rows)), self.issue)
        poisoned = copy.deepcopy(rows)
        for row in poisoned:
            row.update(target_power_norm=999, actual_wind_speed=-999, power=float("nan"))
        self.assertEqual(first, make_features(poisoned, self.issue))
        values = [dict(zip(FEATURE_NAMES, vector)) for vector in first]
        for turbine in ("T1", "T2"):
            own = [value for value in values if value["turbine_id"] == turbine]
            self.assertEqual(own[0]["forecast_wind_delta"], 0)
            self.assertEqual(own[0]["forecast_wind_mean3"], own[0]["wind_speed_100m"])
            self.assertAlmostEqual(own[2]["forecast_wind_mean3"], sum(v["wind_speed_100m"] for v in own) / 3)
        self.assertEqual(values[0]["lead_hours"], 1)
        self.assertEqual(values[0]["weather_lead_hours"], 12)
        self.assertAlmostEqual(values[0]["air_density"], 90000 / (287.05 * 263.15))
        self.assertAlmostEqual(values[0]["wind100_u"], -values[0]["wind_speed_100m"])
        changed = copy.deepcopy(rows)
        changed[2]["wind_speed_100m"] += 10
        future_changed = make_features(changed, self.issue)
        self.assertEqual(first[:2], future_changed[:2])
        self.assertEqual(first[3:], future_changed[3:])

    def test_saved_fallback_is_explicit_and_forbids_pretraining_origins(self):
        engine = self._engine()
        weather = engine.weather(self.issue, 24)
        result, execution, comparison = engine.predict(self.issue, weather, execute=False)
        self.assertEqual(execution, "saved_archive")
        self.assertIsNone(comparison)
        self.assertEqual(len(result), 48)
        self.assertEqual({row["power_pred"] for row in result}, {.2, .4})
        with self.assertRaises(ValueError):
            engine.predict(self.issue - timedelta(hours=1), weather, execute=False)

    def test_saved_forecast_requires_unchanged_file_and_matching_weather_lineage(self):
        weather = self._engine().weather(self.issue, 24)
        weather["provenance"][0]["sha256"] = "another-weather-payload"
        with self.assertRaises(ValueError):
            self._engine().saved(self.issue, weather)
        file = self.folder / "outputs/forecast_all_issues.csv"
        with file.open("a", encoding="utf-8") as stream:
            stream.write("changed")
        with self.assertRaises(ValueError):
            self._engine().saved(self.issue, self._engine().weather(self.issue, 24))

    def test_native_predictions_keep_row_identity_when_weather_is_shuffled(self):
        engine = self._engine()
        weather = engine.weather(self.issue, 24)
        weather["rows"].reverse()
        def fake_worker(command, **kwargs):
            request = json.loads(kwargs["input"])
            position = request["feature_names"].index("wind_speed_100m")
            predictions = [vector[position] / 100 for vector in request["features"]]
            return types.SimpleNamespace(returncode=0, stdout=json.dumps({"predictions": predictions}).encode(), stderr=b"")
        with patch("windagent.project.model_python", return_value="fixture-python"), \
             patch("windagent.project.subprocess.run", side_effect=fake_worker), \
             patch.object(engine, "saved", side_effect=ValueError("Для этого выпуска нет сохранённого прогноза.")):
            result, execution, _ = engine.predict(self.issue, weather)
        self.assertEqual(execution, "recomputed")
        for row in result:
            self.assertAlmostEqual(row["power_pred"], row["wind_speed_100m"] / 100)

    def test_january_audit_recomputes_metrics_and_rejects_wrong_time_buckets(self):
        path = self.folder / "reports/january_predictions.csv"
        audit = audit_january(path, self.report)
        self.assertEqual(audit["rows"], 2)
        self.assertTrue(audit["arithmetic_verified"])
        self.assertFalse(audit["independent_retraining_verified"])
        for field, value in (("lead_hours", 25), ("lead_hours", 49), ("turbine_id", "T3"),
                             ("valid_time", "2026-02-01T00:00:00Z")):
            with self.subTest(field=field, value=value):
                changed = copy.deepcopy(self.january)
                changed[0][field] = value
                self._write_january(changed)
                with self.assertRaises(ValueError):
                    audit_january(path, self.report)
        self._write_january([])
        with self.assertRaises(ValueError):
            audit_january(path, self.report)

    def test_project_agent_reuses_only_unchanged_model_calibration_and_history(self):
        agent = self._agent()
        with patch("windagent.project.model_python", return_value=None):
            first = agent.forecast(self.issue, 24, "project")
            same = agent.forecast(self.issue, 24, "project")
            self.assertTrue(same["reused"])
            self.assertEqual(first["id"], same["id"])
            self.report["calibration_january"]["T1/01-24"] = .15
            save_json(self.folder / "reports/training_report.json", self.report)
            self.info["asset_hashes"]["reports/training_report.json"] = sha_file(self.folder / "reports/training_report.json")
            changed = agent.forecast(self.issue, 24, "project")
            self.assertNotEqual(first["id"], changed["id"])
            self.assertFalse(changed["reused"])
            with Path(agent.state["dataset"]["path"]).open("a", encoding="utf-8") as stream:
                stream.write("\n")
            with self.assertRaises(ValueError):
                agent.forecast(self.issue, 24, "project")
        self.assertEqual(first["execution"], "saved_archive")
        self.assertFalse(first["eligibility"]["competition_ready"])
        self.assertEqual({row["persistence_pred"] for row in first["rows"]}, {.1, .2})

    def test_january_audit_rejects_nonfinite_saved_metrics(self):
        path = self.folder / "reports/january_predictions.csv"
        for field in ("mae", "rmse", "bias"):
            for value in (float("nan"), float("inf")):
                with self.subTest(field=field, value=value):
                    report = copy.deepcopy(self.report)
                    report["january_metrics"][0][field] = value
                    with self.assertRaisesRegex(ValueError, "метрики"):
                        audit_january(path, report)

    def test_engine_rejects_invalid_calibration_even_with_matching_file_hash(self):
        path = self.folder / "reports/training_report.json"
        for value in (float("nan"), float("inf"), -.1, True):
            with self.subTest(radius=value):
                report = copy.deepcopy(self.report)
                report["calibration_january"]["T1/01-24"] = value
                # Deliberately write invalid JSON numbers accepted by Python's
                # decoder, bypassing the application's safe JSON writer.
                path.write_text(json.dumps(report, allow_nan=True), encoding="utf-8")
                self.info["asset_hashes"]["reports/training_report.json"] = sha_file(path)
                with self.assertRaisesRegex(ValueError, "калибровка"):
                    self._engine()

    def test_native_timeout_is_actionable_and_does_not_silently_use_saved_output(self):
        engine = self._engine()
        weather = engine.weather(self.issue, 24)
        with patch("windagent.project.model_python", return_value="fixture-python"), \
             patch("windagent.project.subprocess.run", side_effect=subprocess.TimeoutExpired("fixture-python", 90)), \
             patch.object(engine, "saved") as saved:
            with self.assertRaisesRegex(RuntimeError, "90 секунд"):
                engine.predict(self.issue, weather)
            saved.assert_not_called()

    def test_project_agent_checks_training_boundary_before_cache_or_weather_access(self):
        agent = self._agent()
        with patch("windagent.project.model_python", return_value=None):
            agent.forecast(self.issue, 24, "project")
            self.report["final_training_asof"] = iso(self.issue + timedelta(hours=1))
            save_json(self.folder / "reports/training_report.json", self.report)
            self.info["asset_hashes"]["reports/training_report.json"] = sha_file(self.folder / "reports/training_report.json")
            with patch.object(ProjectEngine, "weather") as weather:
                with self.assertRaises(ValueError):
                    agent.forecast(self.issue, 24, "project")
                weather.assert_not_called()

    def test_native_worker_checks_feature_order_without_needing_catboost(self):
        from windagent import catboost_worker
        captured = {}
        class FakeModel:
            feature_names_ = FEATURE_NAMES
            tree_count_ = 2
            def load_model(self, path, format):
                captured["load"] = (path, format)
            def predict(self, data, thread_count):
                return types.SimpleNamespace(tolist=lambda: [.25] * len(data))
        def fake_pool(data, feature_names, cat_features):
            captured["categorical"] = cat_features
            return data
        module = types.SimpleNamespace(CatBoostRegressor=FakeModel, Pool=fake_pool)
        request = {"model_path": "fixture.cbm", "feature_names": FEATURE_NAMES,
                   "features": make_features(self._engine().weather(self.issue, 24)["rows"], self.issue)}
        output = io.StringIO()
        with patch.dict("sys.modules", {"catboost": module}), \
             patch("sys.stdin", io.StringIO(json.dumps(request))), patch("sys.stdout", output):
            catboost_worker.main()
        self.assertEqual(captured["load"], ("fixture.cbm", "cbm"))
        self.assertEqual(captured["categorical"], [FEATURE_NAMES.index("turbine_id")])
        self.assertEqual(len(json.loads(output.getvalue())["predictions"]), 48)
        request["feature_names"] = list(reversed(FEATURE_NAMES))
        with patch.dict("sys.modules", {"catboost": module}), \
             patch("sys.stdin", io.StringIO(json.dumps(request))), patch("sys.stdout", io.StringIO()):
            with self.assertRaises(ValueError):
                catboost_worker.main()


if __name__ == "__main__":
    unittest.main()
