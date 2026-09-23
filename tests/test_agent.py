"""Offline integration checks: rolling origins, provenance, cache and leakage."""

from __future__ import annotations

import contextlib
import csv
import io
import tempfile
import unittest
from collections import Counter
from datetime import timedelta
from pathlib import Path

from windagent.agent import ForecastAgent
from windagent.common import digest, iso, load_config, read_json, utc
from windagent.demo import fetch_demo_weather
from windagent.evaluate import evaluate


def fixture_csv(config, hours=120):
    """Short synthetic SCADA fixture, used only within this test suite."""
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(("timestamp", "turbine_id", "wind_speed", "temperature", "power"))
    cutoff = utc(config["training_cutoff"])
    for index in range(hours):
        stamp = cutoff - timedelta(hours=hours - index - 1)
        for turbine in config["turbines"]:
            wind = 2 + ((index * 7) % 24) / 2
            temperature = -10 + index % 20
            power = min(1, max(0, (wind - 3) / 9))
            if turbine["id"] == "2":
                power *= 0.8
            writer.writerow((iso(stamp), turbine["id"], wind, temperature, power))
    return stream.getvalue()


class WeatherFixture:
    def __init__(self):
        self.shift = 0
        self.calls = 0
        self.future_publication = False
        self.bad_value = False

    def __call__(self, **kwargs):
        self.calls += 1
        result = fetch_demo_weather(**kwargs)
        for row in result["rows"]:
            row["wind_speed"] += self.shift
        if self.bad_value:
            result["rows"][0]["temperature"] = float("nan")
        for provenance in result["provenance"]:
            provenance["source"] = "offline_test_fixture"
            provenance["availability_basis"] = "test_fixture_publication_metadata"
            # Deliberately keep this identifier fixed: row content must also
            # participate in the run signature, independently of source hash.
            provenance["sha256"] = "fixture-source-id"
            provenance["retrieved_at"] = f"retrieval-number-{self.calls}"
            if self.future_publication:
                provenance["available_at"] = iso(utc(kwargs["as_of"]) + timedelta(hours=1))
        return result


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.workspace = Path(self.temporary.name)
        self.config = load_config()
        self.provider = WeatherFixture()
        self.agent = ForecastAgent(self.workspace, self.config, self.provider)
        self.cutoff = utc(self.config["training_cutoff"])

    def import_fixture(self):
        self.agent.import_csv(fixture_csv(self.config), "test-fixture.csv")

    def test_demo_full_february_has_29_issues_672_hours_and_1344_submission_rows(self):
        self.agent.demo()
        original_model = self.agent.state["model"]["per_turbine"]
        report = self.agent.backtest(mode="demo", hours=48)
        self.assertEqual(self.provider.calls, 0, "Demo must never call an external weather provider")
        self.assertEqual(report["runs"], 29)
        self.assertEqual(report["coverage_hours"], 672)
        self.assertEqual(report["coverage_by_turbine"], {"1": 672, "2": 672})
        self.assertEqual(report["rows"], 2640)
        self.assertEqual(utc(report["issue_runs"][0]["as_of"]), utc("2026-01-31T23:00:00+05:00"))
        self.assertEqual(utc(report["issue_runs"][-1]["as_of"]), utc("2026-02-28T23:00:00+05:00"))
        start, end = utc(self.config["test_start"]), utc(self.config["test_end"])
        backtest_rows = read_json(self.workspace / "backtest_rows.json")
        self.assertTrue(all(start <= utc(row["valid_time"]) < end for row in backtest_rows))
        self.assertEqual(len({row["model_sha256"] for row in backtest_rows}), 1)
        submission = list(csv.DictReader(io.StringIO(self.agent.export("submission"))))
        self.assertEqual(len(submission), 1344)
        self.assertEqual(Counter(row["turbine_id"] for row in submission), {"1": 672, "2": 672})
        self.assertEqual(len({(row["turbine_id"], row["valid_time"]) for row in submission}), 1344)
        self.assertEqual(min(utc(row["valid_time"]) for row in submission), start)
        self.assertEqual(max(utc(row["valid_time"]) for row in submission), end - timedelta(hours=1))
        for row in submission:
            lead = (utc(row["valid_time"]) - utc(row["forecast_origin"])).total_seconds() / 3600
            self.assertTrue(1 <= lead <= 24)
            self.assertEqual(lead, int(row["lead_hours"]))
        # The final issue is preserved for audit even though it predicts March.
        self.assertTrue(all(utc(row["valid_time"]) >= end for row in self.agent.state["last_run"]["rows"]))
        self.assertEqual(original_model, self.agent.state["model"]["per_turbine"])
        self.assertEqual(utc(self.agent.state["model"]["training_cutoff"]), self.cutoff)
        self.assertFalse(report["competition_ready"])
        self.assertNotIn("mae", report)
        self.assertNotIn("rmse", report)

    def test_signature_reuses_identical_input_and_recalculates_weather_change(self):
        self.import_fixture()
        first = self.agent.forecast(self.cutoff, 24, "archive")
        second = self.agent.forecast(self.cutoff, 24, "archive", refresh=True)
        self.assertFalse(first["reused"])
        self.assertTrue(second["reused"])
        self.assertEqual(first["id"], second["id"])
        self.assertTrue(any(event["stage"] == "reuse" for event in second["events"]))
        self.provider.shift = 1
        third = self.agent.forecast(self.cutoff, 24, "archive", refresh=True)
        self.assertFalse(third["reused"])
        self.assertNotEqual(first["id"], third["id"])
        self.assertEqual(first["model_sha256"], third["model_sha256"])
        self.assertNotEqual([row["power_pred"] for row in first["rows"]],
                            [row["power_pred"] for row in third["rows"]])
        self.assertEqual(self.provider.calls, 3)

    def test_publication_after_issue_is_rejected_and_failure_is_logged(self):
        self.import_fixture()
        self.provider.future_publication = True
        with self.assertRaisesRegex(ValueError, "утечка"):
            self.agent.forecast(self.cutoff, 48, "archive")
        self.assertIsNone(self.agent.state["last_run"])
        failure = read_json(self.workspace / "last_error.json")
        self.assertEqual(failure["events"][-1]["stage"], "error")
        self.assertFalse((self.workspace / "runs").exists())

    def test_imported_february_truth_never_changes_model_or_forecast(self):
        history = fixture_csv(self.config)
        self.agent.import_csv(history)
        first_model = self.agent.state["model"]
        first_run = self.agent.forecast(self.cutoff + timedelta(days=10), 48, "archive")
        future = io.StringIO(newline="")
        writer = csv.writer(future)
        for offset in range(1, 28 * 24 + 1):
            for turbine in self.config["turbines"]:
                writer.writerow((iso(self.cutoff + timedelta(hours=offset)), turbine["id"],
                                 40, 60, int(offset % 2 == 0)))
        self.agent.import_csv(history + future.getvalue(), "history-with-forbidden-february-labels.csv")
        changed_model = self.agent.state["model"]
        second_run = self.agent.forecast(self.cutoff + timedelta(days=10), 48, "archive")
        self.assertEqual(first_model["per_turbine"], changed_model["per_turbine"])
        self.assertEqual(first_model["diagnostics"], changed_model["diagnostics"])
        self.assertEqual(changed_model["excluded_future_rows"], 1344)
        self.assertEqual(first_run["id"], second_run["id"])
        self.assertEqual(first_run["rows"], second_run["rows"])
        self.assertTrue(second_run["reused"])
        self.assertEqual(utc(second_run["model_training_cutoff"]), self.cutoff)

    def test_history_change_invalidates_model_and_run_cache(self):
        self.import_fixture()
        first = self.agent.forecast(self.cutoff, 24, "archive")
        source = Path(self.agent.state["dataset"]["path"])
        records = list(csv.DictReader(io.StringIO(source.read_text(encoding="utf-8"))))
        for row in records:
            row["power"] = str(float(row["power"]) * 0.5)
        with source.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        changed = self.agent.forecast(self.cutoff, 24, "archive")
        self.assertNotEqual(first["model_sha256"], changed["model_sha256"])
        self.assertNotEqual(first["id"], changed["id"])
        self.assertFalse(changed["reused"])

    def test_earlier_origin_retrains_without_later_history(self):
        self.import_fixture()
        earlier = self.cutoff - timedelta(hours=24)
        result = self.agent.forecast(earlier, 48, "archive")
        model = self.agent.state["model"]
        self.assertEqual(utc(model["training_cutoff"]), earlier)
        self.assertEqual(utc(model["training_end"]), earlier)
        self.assertEqual(model["training_rows"], 96 * 2)
        self.assertTrue(all(utc(row["valid_time"]) > utc(model["training_cutoff"]) for row in result["rows"]))

    def test_invalid_modes_hours_times_and_nonfinite_weather_fail(self):
        with self.assertRaises(ValueError):
            self.agent.forecast(self.cutoff)
        self.import_fixture()
        for as_of, hours, mode in ((self.cutoff, 25, "archive"), (self.cutoff, 24, "unknown"),
                                  (self.cutoff + timedelta(minutes=15), 24, "archive"),
                                  (self.cutoff, 24, "demo")):
            with self.subTest(as_of=as_of, hours=hours, mode=mode):
                with self.assertRaises(ValueError):
                    self.agent.forecast(as_of, hours, mode)
        with self.assertRaises(ValueError):
            self.agent.demo()
        self.provider.bad_value = True
        with self.assertRaises(ValueError):
            self.agent.forecast(self.cutoff, 24, "archive")

    def test_posthoc_evaluation_cannot_change_training_state(self):
        self.import_fixture()
        run = self.agent.forecast(self.cutoff, 48, "archive")
        before = digest(self.agent.snapshot())
        predictions = self.workspace / "predictions.csv"
        predictions.write_text(self.agent.export(), encoding="utf-8")
        actuals = self.workspace / "actuals.csv"
        with actuals.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("timestamp", "turbine_id", "wind_speed", "temperature", "power"))
            for row in run["rows"]:
                writer.writerow((row["valid_time"], row["turbine_id"], row["wind_speed"], row["temperature"], row["power_pred"]))
        scores = evaluate(predictions, actuals)
        self.assertEqual(scores["matched"], 96)
        self.assertTrue(scores["complete"])
        self.assertEqual(len(scores["metrics"]), 4)
        self.assertTrue(all(group["mae"] == 0 for group in scores["metrics"].values()))
        self.assertEqual(digest(self.agent.snapshot()), before)
        self.assertEqual(utc(self.agent.state["model"]["training_end"]), self.cutoff)

    def test_cli_invalid_watch_settings_return_actionable_failure(self):
        from windagent.__main__ import main

        errors = io.StringIO()
        with contextlib.redirect_stderr(errors):
            result = main(["--workspace", str(self.workspace), "watch", "--interval", "1", "--iterations", "1"])
        self.assertEqual(result, 1)
        self.assertIn("30", errors.getvalue())

    def test_evaluation_rejects_missing_headers_and_blank_required_fields(self):
        actuals = self.workspace / "actuals.csv"
        actuals.write_text(fixture_csv(self.config), encoding="utf-8")
        predictions = self.workspace / "malformed.csv"
        predictions.write_text("valid_time,power_pred\n2026-01-31T18:00Z,0.5\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "столбцы"):
            evaluate(predictions, actuals)
        predictions.write_text(
            "forecast_origin,valid_time,turbine_id,power_pred,persistence_pred\n"
            "2026-01-31T17:00Z,2026-01-31T18:00Z,1,,0.5\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "строка 2.*пропущено"):
            evaluate(predictions, actuals)

    def test_evaluation_rejects_nonfinite_forecasts_even_without_matching_truth(self):
        actuals = self.workspace / "actuals.csv"
        actuals.write_text(fixture_csv(self.config), encoding="utf-8")
        predictions = self.workspace / "invalid-unmatched.csv"
        predictions.write_text(
            "forecast_origin,valid_time,turbine_id,power_pred,persistence_pred\n"
            "2026-01-31T17:00Z,2026-01-31T18:00Z,1,0.5,0.5\n"
            "2026-01-31T17:00Z,2026-01-31T19:00Z,1,nan,0.5\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "конечных"):
            evaluate(predictions, actuals)


if __name__ == "__main__":
    unittest.main()
